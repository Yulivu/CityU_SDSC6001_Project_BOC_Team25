from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


class BusinessEncoder(nn.Module):
    def __init__(self, in_dim: int = 801, out_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MinimalRecModel(nn.Module):
    def __init__(self, business_in_dim: int = 801, dim: int = 128, use_user_embedding: bool = False, num_users: int | None = None):
        super().__init__()
        self.user_emb = None
        if use_user_embedding:
            if num_users is None:
                raise ValueError("num_users is required when use_user_embedding=True")
            emb = nn.Embedding(num_users, dim)
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            self.user_emb = emb
        self.business_encoder = BusinessEncoder(in_dim=business_in_dim, out_dim=dim)

    def encode_business(self, business_x: torch.Tensor) -> torch.Tensor:
        return self.business_encoder(business_x)

    def user_vector(self, hist_vec: torch.Tensor, user_idx: torch.Tensor | None = None) -> torch.Tensor:
        if self.user_emb is None:
            return hist_vec
        if user_idx is None:
            raise ValueError("user_idx is required when use_user_embedding=True")
        return self.user_emb(user_idx) + hist_vec

    def score(self, user_vec: torch.Tensor, business_vec: torch.Tensor) -> torch.Tensor:
        return (user_vec * business_vec).sum(dim=-1)


@dataclass(frozen=True)
class IndexMaps:
    user_to_idx: Dict[str, int]
    idx_to_user: List[str]
    business_to_idx: Dict[str, int]
    idx_to_business: List[str]


def build_index_maps(users: List[str], business_ids: List[str]) -> IndexMaps:
    idx_to_user = list(users)
    user_to_idx = {u: i for i, u in enumerate(idx_to_user)}
    idx_to_business = list(business_ids)
    business_to_idx = {b: i for i, b in enumerate(idx_to_business)}
    return IndexMaps(
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user,
        business_to_idx=business_to_idx,
        idx_to_business=idx_to_business,
    )


def hist_pool_mean(
    business_vecs: torch.Tensor,
    hist_business_indices: List[List[int]],
    device: torch.device,
) -> torch.Tensor:
    lengths = torch.tensor([len(x) for x in hist_business_indices], device=device, dtype=torch.long)
    if (lengths == 0).any():
        raise ValueError("empty history encountered; filter users first")

    flat = torch.tensor([i for sub in hist_business_indices for i in sub], device=device, dtype=torch.long)
    vecs = business_vecs.index_select(0, flat)

    offsets = torch.zeros(len(hist_business_indices) + 1, device=device, dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    out = torch.zeros((len(hist_business_indices), business_vecs.shape[1]), device=device, dtype=business_vecs.dtype)
    for i in range(len(hist_business_indices)):
        start = offsets[i].item()
        end = offsets[i + 1].item()
        out[i] = vecs[start:end].mean(dim=0)
    return out
