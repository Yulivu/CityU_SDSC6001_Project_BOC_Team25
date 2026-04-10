from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import city_to_business_indices, load_prepared_data
from ccrec.metrics import aggregate_ranks, best_rank_among_positives
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def eval_popularity(data, users, business_by_city, rng, k, negatives):
    business = data.business
    ranks = []
    for user_id in tqdm(users, desc="popularity"):
        for (u, target_city), pos_bids in data.user_city_split.target_city_positives.items():
            if u != user_id:
                continue
            if target_city not in business_by_city:
                continue

            pos_set = set(map(str, pos_bids))
            city_idxs = business_by_city[target_city]
            if len(city_idxs) <= len(pos_set):
                continue

            negs = set()
            for _ in range(negatives * 10):
                bid = business.business_ids[int(rng.choice(city_idxs))]
                if bid not in pos_set:
                    negs.add(bid)
                if len(negs) >= negatives:
                    break
            if len(negs) < negatives:
                continue

            cand_bids = list(pos_set) + list(negs)
            cand_idx = np.asarray([business.business_id_to_idx[b] for b in cand_bids], dtype=np.int64)
            scores = business.review_count[cand_idx]
            order = np.argsort(-scores)
            ordered_ids = cand_idx[order].tolist()
            pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
            r = best_rank_among_positives(ordered_ids, pos_idx_set)
            ranks.append(r)
    return aggregate_ranks(ranks, k=k), len([r for r in ranks if r is not None])


def eval_bert_only(data, users, business_by_city, rng, k, negatives):
    business = data.business
    ranks = []
    for user_id in tqdm(users, desc="bert_only"):
        hist_bids = data.user_city_split.source_history_businesses.get(user_id)
        if not hist_bids:
            continue
        hist_idx = [business.business_id_to_idx.get(b) for b in hist_bids]
        hist_idx = [i for i in hist_idx if i is not None]
        if not hist_idx:
            continue
        user_vec = business.bert[np.asarray(hist_idx, dtype=np.int64)].mean(axis=0, keepdims=True)

        for (u, target_city), pos_bids in data.user_city_split.target_city_positives.items():
            if u != user_id:
                continue
            if target_city not in business_by_city:
                continue

            pos_set = set(map(str, pos_bids))
            city_idxs = business_by_city[target_city]
            if len(city_idxs) <= len(pos_set):
                continue

            negs = set()
            for _ in range(negatives * 10):
                bid = business.business_ids[int(rng.choice(city_idxs))]
                if bid not in pos_set:
                    negs.add(bid)
                if len(negs) >= negatives:
                    break
            if len(negs) < negatives:
                continue

            cand_bids = list(pos_set) + list(negs)
            cand_idx = np.asarray([business.business_id_to_idx[b] for b in cand_bids], dtype=np.int64)
            cand_bert = business.bert[cand_idx]
            u = user_vec / (np.linalg.norm(user_vec, axis=1, keepdims=True) + 1e-12)
            v = cand_bert / (np.linalg.norm(cand_bert, axis=1, keepdims=True) + 1e-12)
            scores = (u @ v.T).reshape(-1)
            order = np.argsort(-scores)
            ordered_ids = cand_idx[order].tolist()
            pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
            r = best_rank_among_positives(ordered_ids, pos_idx_set)
            ranks.append(r)
    return aggregate_ranks(ranks, k=k), len([r for r in ranks if r is not None])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = load_prepared_data()
    business_by_city = city_to_business_indices(data.business)

    users = data.val_users if args.split == "val" else data.test_users
    if args.max_users and args.max_users > 0:
        users = users[: args.max_users]

    pop_m, pop_n = eval_popularity(data, users, business_by_city, rng, k=args.k, negatives=args.negatives)
    bert_m, bert_n = eval_bert_only(data, users, business_by_city, rng, k=args.k, negatives=args.negatives)

    payload = {
        "split": args.split,
        "k": args.k,
        "negatives": args.negatives,
        "popularity": {"hr": pop_m.hr_at_k, "ndcg": pop_m.ndcg_at_k, "n": pop_n},
        "bert_only": {"hr": bert_m.hr_at_k, "ndcg": bert_m.ndcg_at_k, "n": bert_n},
    }

    out_path = Path(args.out) if args.out else (artifacts_dir() / f"baseline_metrics_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
