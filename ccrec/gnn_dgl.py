from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class GraphBatch:
    g: "dgl.DGLHeteroGraph"
    user_ids: torch.Tensor
    business_ids: torch.Tensor


class HeteroGAT(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        import dgl.nn as dglnn

        self.conv1 = dglnn.HeteroGraphConv(
            {
                ("user", "ub", "business"): dglnn.GATConv(in_dim, hid_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
                ("business", "bu", "user"): dglnn.GATConv(in_dim, hid_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
                ("user", "uu", "user"): dglnn.GATConv(in_dim, hid_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
            },
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                ("user", "ub", "business"): dglnn.GATConv(hid_dim, out_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
                ("business", "bu", "user"): dglnn.GATConv(hid_dim, out_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
                ("user", "uu", "user"): dglnn.GATConv(hid_dim, out_dim // num_heads, num_heads, feat_drop=dropout, attn_drop=dropout),
            },
            aggregate="sum",
        )
        self.act = nn.ReLU()

    def forward(self, g, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.conv1(g, x_dict)
        h = {k: self.act(v.flatten(1)) for k, v in h.items()}
        h = self.conv2(g, h)
        h = {k: v.flatten(1) for k, v in h.items()}
        return h


class MAGNN_DGL(nn.Module):
    def __init__(
        self,
        num_users: int,
        business_bert_dim: int = 768,
        business_in_dim: int = 802,
        dim: int = 128,
        num_aspects: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_aspects = num_aspects
        self.use_cross_attention = use_cross_attention

        self.user_emb = nn.Embedding(num_users, dim)
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.02)

        self.business_init = nn.Sequential(
            nn.Linear(business_bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim),
        )

        self.gnn = HeteroGAT(in_dim=dim, hid_dim=dim, out_dim=dim, num_heads=num_heads, dropout=dropout)

        self.q = nn.Parameter(torch.empty(num_aspects, dim))
        nn.init.normal_(self.q, mean=0.0, std=0.02)

        self.business_encoder = nn.Sequential(
            nn.Linear(business_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim),
        )

        if use_cross_attention:
            self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=max(1, num_heads), batch_first=True, dropout=dropout)
            self.post_attn = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
            )

        self.scorer = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def graph_forward(
        self,
        g,
        user_idx: torch.Tensor,
        business_bert: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_user = self.user_emb(user_idx)
        x_bus = self.business_init(business_bert)
        h = self.gnn(g, {"user": x_user, "business": x_bus})
        return h["user"], h["business"]

    def user_taste_from_edges(
        self,
        user_h: torch.Tensor,
        business_h: torch.Tensor,
        ub_src: torch.Tensor,
        ub_dst: torch.Tensor,
        ub_aspect: torch.Tensor,
    ) -> torch.Tensor:
        device = user_h.device
        dtype = user_h.dtype
        out = torch.zeros((user_h.shape[0], self.num_aspects, self.dim), device=device, dtype=dtype)
        q = self.q.to(device=device, dtype=dtype)

        for k in range(self.num_aspects):
            mask = ub_aspect == k
            if not mask.any():
                continue
            src = ub_src[mask]
            dst = ub_dst[mask]
            v = business_h.index_select(0, dst)
            scores = (v * q[k]).sum(dim=1)
            exp = torch.exp(scores - scores.max())
            denom = torch.zeros((user_h.shape[0],), device=device, dtype=dtype).index_add_(0, src, exp)
            alpha = exp / (denom.index_select(0, src) + 1e-12)
            contrib = v * alpha.unsqueeze(1)
            out[:, k, :] = out[:, k, :].index_add_(0, src, contrib)

        return out

    def score(
        self,
        user_taste: torch.Tensor,
        w: torch.Tensor,
        business_vec: torch.Tensor,
        distance_feat: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_cross_attention:
            q = user_taste
            k = business_vec.unsqueeze(1)
            v = business_vec.unsqueeze(1)
            attn_out, _ = self.cross_attn(q, k, v, need_weights=False)
            attn_out = self.post_attn(attn_out)
            m = (attn_out * w.unsqueeze(2)).sum(dim=1)
        else:
            m = (user_taste * business_vec.unsqueeze(1) * w.unsqueeze(2)).sum(dim=1)

        x = torch.cat([m, distance_feat.unsqueeze(1)], dim=1)
        return self.scorer(x).squeeze(1)

