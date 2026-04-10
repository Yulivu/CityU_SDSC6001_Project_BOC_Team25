from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FullBatch:
    user_idx: torch.Tensor
    hist_business_idx_by_aspect: List[List[List[int]]]
    w: torch.Tensor
    pos_business_idx: torch.Tensor
    neg_business_idx: torch.Tensor
    distance_pos: torch.Tensor
    distance_neg: torch.Tensor


class FullModel(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_aspects: int = 4,
        hist_bert_dim: int = 768,
        business_in_dim: int = 801,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dim = dim
        self.num_aspects = num_aspects

        self.business_hist_proj = nn.Sequential(
            nn.Linear(hist_bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim),
        )

        self.q = nn.Parameter(torch.empty(num_aspects, dim))
        nn.init.normal_(self.q, mean=0.0, std=0.02)

        self.business_encoder = nn.Sequential(
            nn.Linear(business_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, dim),
        )

        self.matcher = nn.Sequential(
            nn.Linear(dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_business_hist(self, bert: torch.Tensor) -> torch.Tensor:
        return self.business_hist_proj(bert)

    def encode_business(self, x801: torch.Tensor) -> torch.Tensor:
        return self.business_encoder(x801)

    def user_taste(
        self,
        hist_business_emb: torch.Tensor,
        hist_business_idx_by_aspect: List[List[List[int]]],
    ) -> torch.Tensor:
        device = hist_business_emb.device
        dtype = hist_business_emb.dtype
        out = torch.zeros((len(hist_business_idx_by_aspect), self.num_aspects, self.dim), device=device, dtype=dtype)

        q = self.q.to(device=device, dtype=dtype)

        for i, per_aspect in enumerate(hist_business_idx_by_aspect):
            for k in range(self.num_aspects):
                idxs = per_aspect[k]
                if not idxs:
                    continue
                v = hist_business_emb.index_select(0, torch.tensor(idxs, device=device, dtype=torch.long))
                scores = (v * q[k]).sum(dim=1)
                alpha = torch.softmax(scores, dim=0)
                out[i, k] = (alpha.unsqueeze(1) * v).sum(dim=0)

        return out

    def score(
        self,
        user_taste: torch.Tensor,
        w: torch.Tensor,
        business_vec: torch.Tensor,
        distance_feat: torch.Tensor,
    ) -> torch.Tensor:
        b = business_vec.unsqueeze(1)
        m_k = user_taste * b
        m = (m_k * w.unsqueeze(2)).sum(dim=1)
        x = torch.cat([m, distance_feat.unsqueeze(1)], dim=1)
        return self.matcher(x).squeeze(1)


def compute_w_from_var(var: np.ndarray) -> np.ndarray:
    x = -var.astype(np.float32)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)
