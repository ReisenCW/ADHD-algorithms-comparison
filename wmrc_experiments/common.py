from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch.utils.data import Dataset


TASK_NAMES = ["SLD", "SLI", "SSD", "SSI", "VLD", "VLI", "VSD", "VSI"]
DEFAULT_NUM_NODES = 116


@dataclass
class GraphRecord:
    path: str
    task: str
    label: int
    subject: str
    x: torch.Tensor
    adj: torch.Tensor
    community_labels: torch.Tensor
    pe: torch.Tensor
    rws: torch.Tensor  # Random Walk Structural Encoding
    node_mask: torch.Tensor


def _load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def list_graph_files(data_root: str | Path, task_filter: str | None = None) -> list[Path]:
    root = Path(data_root)
    if task_filter and task_filter != "all":
        return sorted((root / task_filter).glob("*/*.pt"))
    return sorted(root.glob("*/*/*.pt"))


def subject_from_path(path: Path) -> str:
    return path.stem


def task_from_path(path: Path) -> str:
    return path.parent.parent.name


def label_from_path(path: Path) -> int:
    return int(path.parent.name)


def build_node_features(sample: dict) -> torch.Tensor:
    adj = sample["adjacency"].float()
    coords = sample["coords"].float()
    deg = adj.abs().sum(dim=1, keepdim=True)
    return torch.cat([adj, deg, coords], dim=1)


def build_node_features_mode(sample: dict, feature_mode: str = "full") -> torch.Tensor:
    adj = sample["adjacency"].float()
    coords = sample["coords"].float()
    deg = adj.abs().sum(dim=1, keepdim=True)
    feature_mode = feature_mode.lower()
    if feature_mode == "full":
        return torch.cat([adj, deg, coords], dim=1)
    if feature_mode == "adj_only":
        return adj
    if feature_mode == "adj_degree":
        return torch.cat([adj, deg], dim=1)
    if feature_mode == "adj_coords":
        return torch.cat([adj, coords], dim=1)
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def pad_graph_tensors(
    adj: torch.Tensor,
    x: torch.Tensor,
    comm: torch.Tensor,
    pe: torch.Tensor,
    rws: torch.Tensor | None = None,
    target_nodes: int = DEFAULT_NUM_NODES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    n = adj.size(0)
    if n > target_nodes:
        raise ValueError(f"graph has {n} nodes, exceeds target size {target_nodes}")
    node_mask = torch.zeros(target_nodes, dtype=torch.bool)
    node_mask[:n] = True

    adj_pad = torch.zeros(target_nodes, target_nodes, dtype=adj.dtype)
    adj_pad[:n, :n] = adj

    if x.size(1) >= n and x.size(1) <= n + 4:
        extra = x.size(1) - n
        x_pad = torch.zeros(target_nodes, target_nodes + extra, dtype=x.dtype)
        x_pad[:n, :n] = x[:, :n]
        x_pad[:n, target_nodes : target_nodes + (x.size(1) - n)] = x[:, n:]
    else:
        x_pad = torch.zeros(target_nodes, x.size(1), dtype=x.dtype)
        x_pad[:n] = x

    comm_pad = torch.full((target_nodes,), -1, dtype=comm.dtype)
    comm_pad[:n] = comm

    pe_pad = torch.zeros(target_nodes, pe.size(1), dtype=pe.dtype)
    pe_pad[:n] = pe

    rws_pad = None
    if rws is not None and rws.numel() > 0:
        rws_pad = torch.zeros(target_nodes, rws.size(1), dtype=rws.dtype)
        rws_pad[:n] = rws
    return adj_pad, x_pad, comm_pad, pe_pad, node_mask, rws_pad


def record_from_path(
    path: Path, pe_dim: int = 8, rw_dim: int = 16, feature_mode: str = "full"
) -> GraphRecord:
    sample = _load_pt(path)
    x = build_node_features_mode(sample, feature_mode=feature_mode)
    pe = laplacian_pe(sample["adjacency"], pe_dim=pe_dim)
    rws = random_walk_se(sample["adjacency"], rw_dim=rw_dim)
    adj, x, comm, pe, node_mask, rws = pad_graph_tensors(
        sample["adjacency"].float(),
        x,
        sample["community_labels"].long(),
        pe,
        rws=rws,
    )
    return GraphRecord(
        path=str(path),
        task=task_from_path(path),
        label=label_from_path(path),
        subject=subject_from_path(path),
        x=x,
        adj=adj,
        community_labels=comm,
        pe=pe,
        rws=rws,
        node_mask=node_mask,
    )


def normalized_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    a = adj.abs().float()
    if a.dim() == 2:
        if add_self_loops:
            a = a + torch.eye(a.size(0), device=a.device, dtype=a.dtype)
        deg = a.sum(dim=1)
        inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        return inv_sqrt.unsqueeze(1) * a * inv_sqrt.unsqueeze(0)
    if a.dim() != 3:
        raise ValueError(f"expected 2D or 3D adjacency, got shape {tuple(a.shape)}")
    if add_self_loops:
        eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).unsqueeze(0)
        a = a + eye
    deg = a.sum(dim=-1)
    inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def signed_normalized_adjacency(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    a = adj.float()
    if a.dim() == 2:
        if add_self_loops:
            a = a + torch.eye(a.size(0), device=a.device, dtype=a.dtype)
        deg = a.abs().sum(dim=1)
        inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        return inv_sqrt.unsqueeze(1) * a * inv_sqrt.unsqueeze(0)
    if a.dim() != 3:
        raise ValueError(f"expected 2D or 3D adjacency, got shape {tuple(a.shape)}")
    if add_self_loops:
        eye = torch.eye(a.size(-1), device=a.device, dtype=a.dtype).unsqueeze(0)
        a = a + eye
    deg = a.abs().sum(dim=-1)
    inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def laplacian_pe(adj: torch.Tensor, pe_dim: int = 8) -> torch.Tensor:
    """Compute Laplacian positional encoding (eigenvectors of normalized Laplacian).

    Uses the GPS paper's approach: eigenvectors of L_norm = I - D^{-1/2} A D^{-1/2},
    with random sign flip for augmentation stability.
    """
    a = adj.abs().float()
    n = a.size(0)
    if pe_dim <= 0:
        return torch.zeros(n, 0, dtype=a.dtype)
    deg = a.sum(dim=1)
    inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
    eye = torch.eye(n, dtype=a.dtype)
    lap = eye - inv_sqrt.unsqueeze(1) * a * inv_sqrt.unsqueeze(0)
    evals, evecs = torch.linalg.eigh(lap)
    order = torch.argsort(evals)
    evecs = evecs[:, order]
    start = 1 if n > 1 else 0
    pe = evecs[:, start : start + pe_dim].clone()
    if pe.size(1) < pe_dim:
        pad = torch.zeros(n, pe_dim - pe.size(1), dtype=a.dtype)
        pe = torch.cat([pe, pad], dim=1)
    # Sign-flip to a consistent canonical orientation per eigenvector
    for i in range(pe.size(1)):
        col = pe[:, i]
        idx = torch.argmax(col.abs())
        if col[idx] < 0:
            pe[:, i] = -col
    return pe


def random_walk_se(adj: torch.Tensor, rw_dim: int = 16) -> torch.Tensor:
    """Compute Random Walk Structural Encoding (RWSE).

    From GPS paper: diagonal of the m-step random walk matrix.
    For each step k (1..rw_dim), computes diag((D^{-1} A)^k).
    This encodes local structural information (e.g. if a node
    is part of a k-cycle, its return probability after k steps differs).
    """
    a = adj.abs().float()
    n = a.size(0)
    if rw_dim <= 0:
        return torch.zeros(n, 0, dtype=a.dtype)
    deg = a.sum(dim=1).clamp_min(1.0)
    deg_inv = 1.0 / deg
    # Transition matrix: P = D^{-1} A
    p = deg_inv.unsqueeze(1) * a
    se = []
    pk = torch.eye(n, dtype=a.dtype, device=a.device)
    for _ in range(rw_dim):
        pk = pk @ p
        se.append(torch.diag(pk))
    return torch.stack(se, dim=1)  # [N, rw_dim]


class WMRCGraphDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        task_filter: str | None = None,
        max_samples: int | None = None,
        pe_dim: int = 8,
        rw_dim: int = 16,
        feature_mode: str = "full",
    ) -> None:
        self.data_root = Path(data_root)
        self.task_filter = task_filter or "all"
        self.pe_dim = pe_dim
        self.rw_dim = rw_dim
        self.feature_mode = feature_mode
        files = list_graph_files(self.data_root, self.task_filter)
        if max_samples is not None:
            files = files[:max_samples]
        self.records: list[GraphRecord] = []
        for path in files:
            self.records.append(record_from_path(
                path, pe_dim=pe_dim, rw_dim=rw_dim, feature_mode=feature_mode
            ))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> GraphRecord:
        return self.records[idx]


def collate_graphs(batch: Sequence[GraphRecord]) -> dict:
    return {
        "x": torch.stack([item.x for item in batch], dim=0),
        "adj": torch.stack([item.adj for item in batch], dim=0),
        "pe": torch.stack([item.pe for item in batch], dim=0),
        "rws": torch.stack([item.rws for item in batch], dim=0),
        "comm": torch.stack([item.community_labels for item in batch], dim=0),
        "mask": torch.stack([item.node_mask for item in batch], dim=0),
        "y": torch.tensor([item.label for item in batch], dtype=torch.long),
        "path": [item.path for item in batch],
        "task": [item.task for item in batch],
        "subject": [item.subject for item in batch],
    }


def split_by_subject(
    dataset: WMRCGraphDataset,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")

    subjects_by_label: dict[int, list[str]] = defaultdict(list)
    subject_to_indices: dict[str, list[int]] = defaultdict(list)
    subject_to_label: dict[str, int] = {}

    for idx, record in enumerate(dataset.records):
        subject_to_indices[record.subject].append(idx)
        subject_to_label.setdefault(record.subject, record.label)
        if subject_to_label[record.subject] != record.label:
            raise ValueError(
                f"inconsistent labels for subject {record.subject}: "
                f"{subject_to_label[record.subject]} vs {record.label}"
            )

    for subject, label in subject_to_label.items():
        subjects_by_label[label].append(subject)

    rng = random.Random(seed)
    split_indices = {"train": [], "val": [], "test": []}
    split_subjects = {"train": [], "val": [], "test": []}

    for label, subjects in subjects_by_label.items():
        subjects = subjects[:]
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = max(1, round(n * train_ratio)) if n >= 3 else max(1, n - 1)
        n_val = max(1, round(n * val_ratio)) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
            n_val = 1 if n >= 3 else 0
        n_test = n - n_train - n_val
        if n_test <= 0 and n >= 3:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
        buckets = {
            "train": subjects[:n_train],
            "val": subjects[n_train : n_train + n_val],
            "test": subjects[n_train + n_val :],
        }
        for split, subject_list in buckets.items():
            for subject in subject_list:
                split_subjects[split].append(subject)
                split_indices[split].extend(subject_to_indices[subject])

    for split in split_indices:
        split_indices[split].sort()
        split_subjects[split].sort()
    return {"indices": split_indices, "subjects": split_subjects}


def make_loader(dataset: WMRCGraphDataset, indices: list[int], batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs,
        drop_last=False,
    )


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def f1_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    tp = ((pred == 1) & (y == 1)).sum().item()
    fp = ((pred == 1) & (y == 0)).sum().item()
    fn = ((pred == 0) & (y == 1)).sum().item()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


def macro_f1_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    scores = []
    for cls in [0, 1]:
        tp = ((pred == cls) & (y == cls)).sum().item()
        fp = ((pred == cls) & (y != cls)).sum().item()
        fn = ((pred != cls) & (y == cls)).sum().item()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        scores.append(2 * precision * recall / (precision + recall + 1e-12))
    return sum(scores) / len(scores)


def balanced_accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    recalls = []
    for cls in [0, 1]:
        tp = ((pred == cls) & (y == cls)).sum().item()
        fn = ((pred != cls) & (y == cls)).sum().item()
        recalls.append(tp / (tp + fn + 1e-12))
    return sum(recalls) / len(recalls)


def save_predictions_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
