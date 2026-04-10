from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class RankingMetrics:
    hr_at_k: float
    ndcg_at_k: float


def _dcg(rank: int) -> float:
    return 1.0 / math.log2(rank + 1.0)


def hr_ndcg_at_k_from_ranks(ranks: Sequence[int], k: int) -> RankingMetrics:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ranks:
        return RankingMetrics(hr_at_k=0.0, ndcg_at_k=0.0)

    hits = 0
    ndcg = 0.0
    for r in ranks:
        if 1 <= r <= k:
            hits += 1
            ndcg += _dcg(r)

    n = len(ranks)
    return RankingMetrics(hr_at_k=hits / n, ndcg_at_k=ndcg / n)


def best_rank_among_positives(sorted_ids: Sequence[int], positive_ids: set[int]) -> int | None:
    for idx, bid in enumerate(sorted_ids, start=1):
        if bid in positive_ids:
            return idx
    return None


def aggregate_ranks(ranks: Iterable[int | None], k: int) -> RankingMetrics:
    ranks_list = [r for r in ranks if r is not None]
    if not ranks_list:
        return RankingMetrics(hr_at_k=0.0, ndcg_at_k=0.0)
    return hr_ndcg_at_k_from_ranks(ranks_list, k=k)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T

