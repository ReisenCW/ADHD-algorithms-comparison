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
    """Graph Convolutional Layer (Kipf & Welling, ICLR 2017)"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        a = signed_normalized_adjacency(adj)
        return self.lin(torch.bmm(a, x))


class GCNModel(nn.Module):
    """GCN: Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)"""
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
        rws: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = masked_mean(x, mask)
        return self.head(g)


class GATLayer(nn.Module):
    """Graph Attention Layer (Veličković et al., ICLR 2018)"""

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
    """GAT: Graph Attention Network (Veličković et al., ICLR 2018)"""
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
        rws: torch.Tensor | None = None,
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
    """Hypergraph Neural Network Layer (Feng et al., AAAI 2019)"""
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
    """HGNN: Hypergraph Neural Network (Feng et al., AAAI 2019)"""
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
        rws: torch.Tensor | None = None,
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
    """HyperGCN: Hypergraph Convolutional Network (Chandra et al., NeurIPS 2020)"""
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
        rws: torch.Tensor | None = None,
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


class GINELayer(nn.Module):
    """GINE (Graph Isomorphism Network with Edge features) layer.

    Used as the MPNN component in GPS layers.
    Handles signed adjacency: edge features encode both connectivity and sign.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_scale = nn.Parameter(torch.ones(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, N, d], adj: [B, N, N] (signed adjacency)
        # Message from neighbors with edge weights as features
        # m_ij = ReLU(x_j * w_ij_sign + edge_scale * |w_ij|)
        # Aggregate: sum over neighbors
        adj_abs = adj.abs()
        adj_sign = torch.sign(adj)
        # Edge feature: sign-aware weight
        e = adj_sign * adj_abs * self.edge_scale  # [B, N, N]
        # Vectorized neighbor message aggregation
        # For each node i: sum_{j} ReLU(x_j + e_ij)
        # We broadcast x and use adjacency as the mask
        bsz, n, d = x.shape
        # Use mask to zero out padded nodes before message passing
        if mask is not None:
            m = mask.unsqueeze(-1).to(dtype=x.dtype)  # [B, N, 1]
            x_masked = x * m
        else:
            x_masked = x
        # Expand for message: x_j + e_ij for all i,j
        # Instead of materializing full [B,N,N,d], compute via masked matmul
        adj_mask = (adj_abs > 0).to(dtype=x.dtype)  # [B, N, N]
        if mask is not None:
            node_mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
            adj_mask = adj_mask * node_mask_2d.to(dtype=adj_mask.dtype)
        # messages: X_j broadcast, masked by adjacency
        # agg_i = sum_j adj_mask_ij * ReLU(x_j * sign_ij + scale * |w_ij|)
        sign_component = x_masked.unsqueeze(2) * adj_sign.unsqueeze(-1)  # [B, N, N, d]
        abs_component = e.unsqueeze(-1)  # [B, N, N, 1]
        msg = F.relu(sign_component + abs_component)  # [B, N, N, d]
        agg = (msg * adj_mask.unsqueeze(-1)).sum(dim=2)  # [B, N, d]
        # GINE update: (1+eps)*x + agg
        out = (1.0 + self.eps) * x_masked + agg
        return self.mlp(out)


class GPSLayer(nn.Module):
    """One GPS layer from Rampášek et al. NeurIPS 2022.

    Each layer combines:
    - Local MPNN (GINE) acting on graph adjacency with edge features
    - Global self-attention (no edge features in attention)
    - 2-layer MLP fusion with residual connections and LayerNorm
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.15) -> None:
        super().__init__()
        # MPNN branch
        self.mpnn = GINELayer(hidden_dim, dropout)
        self.mpnn_norm = nn.LayerNorm(hidden_dim)
        # Global attention branch (no edge features per paper design)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        # 2-layer MLP fusion block
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # --- MPNN branch (local neighborhood) ---
        h_mpnn = self.mpnn(x, adj, mask)
        h_mpnn = F.dropout(h_mpnn, p=self.dropout, training=self.training)
        x_mpnn = self.mpnn_norm(x + h_mpnn)  # residual + norm

        # --- Global attention branch ---
        key_padding_mask = ~mask if mask is not None else None
        h_attn, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        h_attn = F.dropout(h_attn, p=self.dropout, training=self.training)
        x_attn = self.attn_norm(x + h_attn)  # residual + norm

        # --- Fusion: sum + MLP ---
        x_fused = x_mpnn + x_attn
        h_mlp = self.mlp(x_fused)
        h_mlp = F.dropout(h_mlp, p=self.dropout, training=self.training)
        return self.mlp_norm(x_fused + h_mlp)  # residual + norm


class GraphTransformerModel(nn.Module):
    """GPS Graph Transformer following Rampášek et al. NeurIPS 2022.

    Architecture:
    1. Input embedding: node features + Laplacian PE + RWSE → MLP projections
    2. GPS layers: each = MPNN(GINE) + Self-Attention + MLP fusion
    3. Readout: masked mean pooling over nodes
    4. Classification head
    """

    def __init__(
        self,
        in_dim: int = 120,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.15,
        pe_dim: int = 8,
        rw_dim: int = 16,
    ) -> None:
        super().__init__()
        self.pe_dim = pe_dim
        self.rw_dim = rw_dim
        # Input encoders: project each encoding type into hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim) if pe_dim > 0 else None
        self.rw_encoder = nn.Linear(rw_dim, hidden_dim) if rw_dim > 0 else None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # GPS layers (hybrid MPNN + Transformer)
        self.gps_layers = nn.ModuleList(
            [GPSLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.head = GraphClassifierHead(
            hidden_dim, num_classes=num_classes, dropout=0.5
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        pe: torch.Tensor | None = None,
        comm: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        rws: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input encoding: project and sum node + PE + RWSE features
        x = self.node_encoder(x)
        if pe is not None and self.pe_encoder is not None:
            x = x + self.pe_encoder(pe)
        if rws is not None and rws.size(-1) > 0 and self.rw_encoder is not None:
            x = x + self.rw_encoder(rws)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Mask node features for padded positions
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(dtype=x.dtype)

        # GPS layers
        for layer in self.gps_layers:
            x = layer(x, adj, mask)

        # Graph-level readout
        g = masked_mean(x, mask)
        return self.head(g)


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
    rw_dim: int = 16,
) -> nn.Module:
    """根据名称构建模型。

    支持的模型:
        gcn      - GCN: Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling, ICLR 2017)
        gat      - GAT: Graph Attention Network (Veličković et al., ICLR 2018)
        hgnn     - HGNN: Hypergraph Neural Network (Feng et al., AAAI 2019)
        hypergcn - HyperGCN: Hypergraph Convolutional Network (Chandra et al., NeurIPS 2020)
        gt       - Graph Transformer: Generalization of Transformer to Graphs (Dwivedi & Bresson, 2021)
    """
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
            rw_dim=rw_dim,
        )
    raise ValueError(f"unknown model: {model_name}")
