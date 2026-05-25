"""
OpenNeuro-ds002424 全局通道数据读取
====================================
.pt 文件结构（每个被试一个文件）：
    {
        'adjacency':        Tensor (N, N)   - 全局静态邻接矩阵
        'coords':           Tensor (N, 3)   - ROI MNI坐标（可选）
        'community_labels': Tensor (N,)     - Louvain社区标签（可选）
    }

目录结构（标签从文件夹名读取）：
    pt_dir/
        0/          ← HC (健康对照)
            sub-01.pt
            sub-02.pt
            ...
        1/          ← ADHD
            sub-10.pt
            ...

用法：
    python load_global.py --pt_dir /path/to/global_pt
"""

import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# ============================================================
# Dataset
# ============================================================

class GlobalGraphDataset(Dataset):
    """
    全局静态图数据集，仅加载全局通道（无时序）的 .pt 文件。
    标签从父目录名推断：目录名为 '0' 或 '1'。
    """
    def __init__(self, pt_dir: str):
        all_pts = sorted(glob.glob(os.path.join(pt_dir, '**', '*.pt'), recursive=True))
        if len(all_pts) == 0:
            raise RuntimeError(f"在 {pt_dir} 下未找到任何 .pt 文件")

        files, labels = [], []
        for p in all_pts:
            parts = os.path.normpath(p).split(os.sep)
            label = None
            for part in reversed(parts[:-1]):
                if part in ('0', '1'):
                    label = int(part)
                    break
            if label is None:
                print(f"[跳过] {p}：父目录名不是 '0' 或 '1'")
                continue
            files.append(p)
            labels.append(label)

        if len(files) == 0:
            raise RuntimeError(f"未找到任何带标签的 .pt 文件，请确认目录结构为 pt_dir/0/*.pt 和 pt_dir/1/*.pt")

        self.files = files
        self.labels = labels

        # 扫描最大节点数 N，用于 padding
        self.N = max(self._get_N(p) for p in self.files)
        print(f"数据集大小: {len(self.files)} 个样本，最大节点数 N={self.N}")
        print(f"  ADHD(1): {sum(labels)}，HC(0): {len(labels) - sum(labels)}")

    def _get_N(self, path):
        d = torch.load(path, map_location='cpu')
        adj = d['adjacency']
        return int(adj.shape[0]) if adj.dim() == 2 else int(adj.shape[1])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        label = self.labels[idx]
        d = torch.load(p, map_location='cpu')

        # 邻接矩阵
        adj = d['adjacency'].float()
        if adj.dim() == 3:          # 兼容 (1, N, N) 格式
            adj = adj[0]
        N_curr = adj.shape[0]

        # coords：若无则用零填充
        coords = d.get('coords', None)
        if coords is not None:
            coords = coords.float()
        else:
            coords = torch.zeros(N_curr, 3)

        # 社区标签：若无则全为 -1
        comm = d.get('community_labels', None)
        if comm is not None:
            comm = comm.long()
        else:
            comm = torch.full((N_curr,), -1, dtype=torch.long)

        # padding 到统一的 N（若数据集内所有 N 相同则此步骤不改变任何内容）
        if N_curr < self.N:
            pad = self.N - N_curr
            adj    = F.pad(adj,    (0, pad, 0, pad))
            coords = F.pad(coords, (0, 0, 0, pad))
            comm   = F.pad(comm,   (0, pad), value=-1)

        return {
            'adjacency':        adj,           # (N, N)
            'coords':           coords,        # (N, 3)
            'community_labels': comm,          # (N,)
            'label':            torch.tensor(label, dtype=torch.long),
            'file':             p,
        }


def build_dataloaders(pt_dir: str, batch_size: int = 16, seed: int = 42):
    """
    构建训练/验证/测试 DataLoader（60/20/20，分层采样）。
    """
    dataset = GlobalGraphDataset(pt_dir)
    labels = dataset.labels
    idx = list(range(len(dataset)))

    idx_train, idx_tmp, _, y_tmp = train_test_split(
        idx, labels, test_size=0.4, stratify=labels, random_state=seed)
    idx_val, idx_test, _, _ = train_test_split(
        idx_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(dataset, idx_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, idx_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(dataset, idx_test),  batch_size=batch_size, shuffle=False)

    print(f"划分完成 → Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    return train_loader, val_loader, test_loader, dataset.N


# ============================================================
# 节点特征构建（从邻接矩阵 + coords 拼接）
# ============================================================

def build_node_features(adj: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    构建节点特征矩阵。
    输入：
        adj:    (B, N, N)  邻接矩阵
        coords: (B, N, 3)  MNI坐标
    输出：
        feat:   (B, N, N+4)  = [邻接行(N) | 节点度(1) | 坐标(3)]
    """
    deg = adj.abs().sum(dim=-1, keepdim=True)   # (B, N, 1)
    feat = torch.cat([adj, deg, coords], dim=-1) # (B, N, N+4)
    return feat


# ============================================================
# 将邻接矩阵转换为 edge_index + edge_attr（供PyG使用）
# ============================================================

def adj_to_edge_index(adj: torch.Tensor):
    """
    将单张图的邻接矩阵转为 PyG 格式。
    输入：adj (N, N)
    输出：edge_index (2, E)，edge_attr (E, 1)
    """
    mask = adj != 0
    src, dst = mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr  = adj[src, dst].unsqueeze(1)
    return edge_index, edge_attr


# ============================================================
# 示例：验证读取是否正常
# ============================================================

def quick_check(pt_dir: str):
    print("=" * 50)
    print("快速验证数据读取")
    print("=" * 50)

    dataset = GlobalGraphDataset(pt_dir)
    sample = dataset[0]

    print(f"adjacency shape:        {sample['adjacency'].shape}")    # (N, N)
    print(f"coords shape:           {sample['coords'].shape}")       # (N, 3)
    print(f"community_labels shape: {sample['community_labels'].shape}")  # (N,)
    print(f"label:                  {sample['label'].item()}")
    print(f"NaN in adj:             {sample['adjacency'].isnan().any().item()}")
    print(f"非零边数:                {int((sample['adjacency'] != 0).sum().item() // 2)}")
    print(f"社区数量:                {sample['community_labels'].unique().numel()}")

    # 测试 DataLoader batch
    loader, _, _, N = build_dataloaders(pt_dir, batch_size=4)
    batch = next(iter(loader))
    print(f"\nDataLoader batch:")
    print(f"  adjacency:  {batch['adjacency'].shape}")   # (4, N, N)
    print(f"  coords:     {batch['coords'].shape}")      # (4, N, 3)
    print(f"  label:      {batch['label']}")

    # 测试特征构建
    feat = build_node_features(batch['adjacency'], batch['coords'])
    print(f"  node feat:  {feat.shape}")                 # (4, N, N+4)

    print("\n验证通过 ✓")


# ============================================================
# main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', required=True, help='全局通道 .pt 文件目录')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--check', action='store_true', help='只做快速验证，不训练')
    args = parser.parse_args()

    if args.check:
        quick_check(args.pt_dir)
    else:
        train_loader, val_loader, test_loader, N = build_dataloaders(
            args.pt_dir, args.batch_size, args.seed)
        print(f"\n数据加载完成，节点数 N={N}，节点特征维度={N+4}")
        print("可将 train_loader / val_loader / test_loader 传入你的模型训练流程。")
