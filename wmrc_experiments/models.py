from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import normalized_adjacency, signed_normalized_adjacency


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class GraphClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = signed_normalized_adjacency(adj)
        return self.lin(torch.bmm(a, x))


class GCNModel(nn.Module):
    def __init__(self, in_dim: int = 120, hidden_dim: int = 64, num_layers: int = 3, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(d_in, hidden_dim))
            d_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.head = GraphClassifierHead(hidden_dim, num_classes=num_classes, dropout=0.5)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = masked_mean(x, mask)
        return self.head(g)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, concat: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.dropout = dropout
        self.lin = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim * heads if concat else out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, n, _ = x.shape
        h = self.lin(x).view(bsz, n, self.heads, self.out_dim).transpose(1, 2)
        src = (h * self.attn_src[None, :, None, :]).sum(dim=-1)
        dst = (h * self.attn_dst[None, :, None, :]).sum(dim=-1)
        e = self.leaky_relu(src.unsqueeze(-1) + dst.unsqueeze(-2))
        edge_mask = adj.abs() > 0
        eye = torch.eye(n, device=adj.device, dtype=torch.bool).unsqueeze(0)
        if mask is not None:
            node_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            edge_mask = (edge_mask | eye) & node_mask
        else:
            edge_mask = edge_mask | eye
        edge_mask = edge_mask.unsqueeze(1)
        e = e.masked_fill(~edge_mask, -1e9)
        alpha = torch.softmax(e, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.matmul(alpha, h)
        if self.concat:
            out = out.transpose(1, 2).contiguous().view(bsz, n, self.heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        return out + self.bias


class GATModel(nn.Module):
    """GAT模型，支持可配置层数"""
    def __init__(self, in_dim: int = 120, hidden_dim: int = 64, heads: int = 4, num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.layers.append(GATLayer(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))

        # Hidden layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.layers.append(GATLayer(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout))

        # Last layer
        self.layers.append(GATLayer(hidden_dim * heads if num_layers > 1 else in_dim, hidden_dim, heads=1, concat=False, dropout=dropout))

        self.head = GraphClassifierHead(hidden_dim, num_classes=num_classes, dropout=0.5)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.elu(layer(x, adj, mask=mask))
            if i < len(self.layers) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        g = masked_mean(x, mask)
        return self.head(g)


def build_hyperedges(adj: torch.Tensor, comm: torch.Tensor | None = None, topk: int = 8, node_mask: torch.Tensor | None = None) -> torch.Tensor:
    n = int(node_mask.sum().item()) if node_mask is not None else adj.size(0)
    adj = adj[:n, :n]
    comm = comm[:n] if comm is not None else None
    a = adj.abs().clone()
    a.fill_diagonal_(0)
    edges: list[list[int]] = []
    for i in range(n):
        k = min(topk, n - 1)
        nbrs = torch.topk(a[i], k=k).indices.tolist()
        nodes = sorted(set([i] + nbrs))
        if len(nodes) >= 2:
            edges.append(nodes)
    if comm is not None:
        unique_comm = torch.unique(comm)
        for c in unique_comm.tolist():
            nodes = torch.nonzero(comm == c, as_tuple=False).view(-1).tolist()
            if len(nodes) >= 2:
                edges.append(sorted(set(nodes)))
    if not edges:
        edges = [list(range(n))]
    h = torch.zeros(n, len(edges), dtype=adj.dtype, device=adj.device)
    for e_idx, nodes in enumerate(edges):
        h[nodes, e_idx] = 1.0
    return h


class HGNNLayer(nn.Module):
    """高效向量化的HGNN层"""
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], h: [B, N, E]
        # 向量化计算: D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} X
        de = h.sum(dim=1).clamp_min(1.0)  # [B, E]
        dv = h.sum(dim=2).clamp_min(1.0)  # [B, N]
        dv_inv_sqrt = dv.pow(-0.5).unsqueeze(-1)  # [B, N, 1]
        de_inv = de.reciprocal().unsqueeze(1)  # [B, 1, E]
        norm_h = dv_inv_sqrt * h * de_inv  # [B, N, E]
        prop = torch.bmm(norm_h, h.transpose(1, 2))  # [B, N, N]
        prop = dv_inv_sqrt * prop * dv_inv_sqrt.transpose(1, 2)
        return self.lin(torch.bmm(prop, x))


class HGNNModel(nn.Module):
    def __init__(self, in_dim: int = 120, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.3, topk: int = 8) -> None:
        super().__init__()
        self.topk = topk
        self.layer1 = HGNNLayer(in_dim, hidden_dim)
        self.layer2 = HGNNLayer(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.head = GraphClassifierHead(hidden_dim, num_classes=num_classes, dropout=0.5)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hs = []
        max_n = adj.size(1)
        for i, (ab, cb) in enumerate(zip(adj, comm if comm is not None else [None] * adj.size(0))):
            hb = build_hyperedges(ab, cb, topk=self.topk, node_mask=mask[i] if mask is not None else None)
            hpad = torch.zeros(max_n, hb.size(1), dtype=hb.dtype, device=hb.device)
            hpad[: hb.size(0)] = hb
            hs.append(hpad)
        max_e = max(h.size(1) for h in hs)
        h = torch.stack([F.pad(h, (0, max_e - h.size(1))) for h in hs], dim=0)
        x = F.relu(self.layer1(x, h))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.layer2(x, h))
        g = masked_mean(x, mask)
        return self.head(g)


class HyperGCNModel(nn.Module):
    def __init__(self, in_dim: int = 120, hidden_dim: int = 64, num_layers: int = 3, num_classes: int = 2, dropout: float = 0.3, topk: int = 8) -> None:
        super().__init__()
        self.topk = topk
        layers = []
        d_in = in_dim
        for _ in range(num_layers):
            layers.append(GCNLayer(d_in, hidden_dim))
            d_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.dropout = dropout
        self.head = GraphClassifierHead(hidden_dim, num_classes=num_classes, dropout=0.5)

    def _clique_adj(self, adj: torch.Tensor, comm: torch.Tensor | None = None, node_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = build_hyperedges(adj, comm, topk=self.topk, node_mask=node_mask)
        a = h @ h.t()
        signs = torch.sign(adj[: a.size(0), : a.size(1)])
        a = a * torch.where(signs >= 0, torch.ones_like(signs), -torch.ones_like(signs))
        a.fill_diagonal_(0)
        return a

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ads = []
        max_n = adj.size(1)
        for i, (ab, cb) in enumerate(zip(adj, comm if comm is not None else [None] * adj.size(0))):
            ac = self._clique_adj(ab, cb, node_mask=mask[i] if mask is not None else None)
            apad = torch.zeros(max_n, max_n, dtype=ac.dtype, device=ac.device)
            apad[: ac.size(0), : ac.size(1)] = ac
            ads.append(apad)
        a = torch.stack(ads, dim=0)
        for layer in self.layers:
            x = layer(x, a)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = masked_mean(x, mask)
        return self.head(g)


class GraphTransformerModel(nn.Module):
    def __init__(
        self,
        in_dim: int = 120,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.3,
        pe_dim: int = 8,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = GraphClassifierHead(hidden_dim, num_classes=num_classes, dropout=0.5)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.lin_in(x)
        if pe is not None and self.pe_proj is not None:
            x = x + self.pe_proj(pe)
        x = self.norm(x)
        key_padding_mask = ~mask if mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = masked_mean(x, mask)
        return self.head(x)


def build_model(
    model_name: str,
    in_dim: int = 120,
    hidden_dim: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    dropout: float = 0.3,
    pe_dim: int = 8,
    topk: int = 8,
    num_classes: int = 2,
) -> nn.Module:
    name = model_name.lower()
    if name == "gcn":
        return GCNModel(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, dropout=dropout)
    if name == "gat":
        return GATModel(in_dim=in_dim, hidden_dim=hidden_dim, heads=heads, num_layers=num_layers, num_classes=num_classes, dropout=dropout)
    if name == "hgnn":
        return HGNNModel(in_dim=in_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout, topk=topk)
    if name == "hypergcn":
        return HyperGCNModel(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_classes=num_classes, dropout=dropout, topk=topk)
    if name in {"gt", "graph_transformer", "graphtransformer"}:
        return GraphTransformerModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=heads,
            num_classes=num_classes,
            dropout=dropout,
            pe_dim=pe_dim,
        )
    raise ValueError(f"unknown model: {model_name}")
