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
from scripts.train_mf import MF


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=str, default=str(artifacts_dir() / "mf.pt"))
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    prepared = load_prepared_data()
    business = prepared.business
    business_by_city = city_to_business_indices(business)

    ckpt = torch.load(args.ckpt, map_location=device)
    user_ids = list(map(str, ckpt["user_ids"]))
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    model = MF(num_users=len(user_ids), num_items=len(business.business_ids), dim=int(ckpt.get("dim", 64))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    users = prepared.val_users if args.split == "val" else prepared.test_users
    eligible = {u for (u, _city) in prepared.user_city_split.target_city_positives.keys()}
    users = [u for u in users if u in eligible]
    if args.max_users and args.max_users > 0 and len(users) > args.max_users:
        idx = rng.choice(len(users), size=args.max_users, replace=False)
        users = [users[i] for i in idx.tolist()]

    ranks: list[int | None] = []
    with torch.no_grad():
        for user_id in tqdm(users, desc=f"eval_mf {args.split}"):
            ui = user_to_idx.get(user_id)
            if ui is None:
                hist_bids = prepared.user_city_split.source_history_businesses.get(user_id)
                if not hist_bids:
                    continue
                hist_idx = [business.business_id_to_idx.get(str(b)) for b in hist_bids]
                hist_idx = [i for i in hist_idx if i is not None]
                if not hist_idx:
                    continue
                hist_t = torch.tensor(hist_idx, device=device, dtype=torch.long)
                u_vec = model.item(hist_t).mean(dim=0, keepdim=True)
            else:
                u_vec = model.user(torch.tensor([int(ui)], device=device, dtype=torch.long))

            for (u, target_city), pos_bids in prepared.user_city_split.target_city_positives.items():
                if u != user_id:
                    continue
                if target_city not in business_by_city:
                    continue

                pos_set = set(map(str, pos_bids))
                city_idxs = business_by_city[target_city]
                if len(city_idxs) <= len(pos_set):
                    continue

                negs = set()
                for _ in range(args.negatives * 10):
                    bid = business.business_ids[int(rng.choice(city_idxs))]
                    if bid not in pos_set:
                        negs.add(bid)
                    if len(negs) >= args.negatives:
                        break
                if len(negs) < args.negatives:
                    continue

                cand_bids = list(pos_set) + list(negs)
                cand_idx = torch.tensor([business.business_id_to_idx[b] for b in cand_bids], device=device, dtype=torch.long)
                cand_vec = model.item(cand_idx)
                scores = (cand_vec @ u_vec.squeeze(0)).detach().float().cpu().numpy()

                order = np.argsort(-scores)
                ordered_ids = cand_idx.detach().cpu().numpy()[order].tolist()
                pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
                ranks.append(best_rank_among_positives(ordered_ids, pos_idx_set))

    metrics = aggregate_ranks(ranks, k=args.k)
    payload = {
        "split": args.split,
        "k": args.k,
        "negatives": args.negatives,
        "hr": metrics.hr_at_k,
        "ndcg": metrics.ndcg_at_k,
        "n": len([r for r in ranks if r is not None]),
    }

    out_path = Path(args.out) if args.out else (artifacts_dir() / f"mf_metrics_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
