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
from ccrec.models import MinimalRecModel
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=str, default=str(artifacts_dir() / "minimal_model.pt"))
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data = load_prepared_data()
    business = data.business
    x = torch.from_numpy(business.x).to(device=device, dtype=torch.float16)
    business_by_city = city_to_business_indices(business)

    users = data.val_users if args.split == "val" else data.test_users
    if args.max_users and args.max_users > 0:
        users = users[: args.max_users]

    ckpt = torch.load(args.ckpt, map_location=device)
    model = MinimalRecModel(
        business_in_dim=int(ckpt["business_in_dim"]),
        dim=int(ckpt["dim"]),
        use_user_embedding=bool(ckpt.get("use_user_embedding", False)),
        num_users=None,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            all_business_vec = model.encode_business(x).detach()

    ranks = []
    for user_id in tqdm(users, desc=f"eval {args.split}"):
        hist_bids = data.user_city_split.source_history_businesses.get(user_id)
        if not hist_bids:
            continue
        hist_idx = [business.business_id_to_idx.get(b) for b in hist_bids]
        hist_idx = [i for i in hist_idx if i is not None]
        if not hist_idx:
            continue
        hist_idx_t = torch.tensor(hist_idx, device=device, dtype=torch.long)
        hist_mean = all_business_vec.index_select(0, hist_idx_t).mean(dim=0, keepdim=True)
        user_vec = model.user_vector(hist_mean)

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
            cand_vec = all_business_vec.index_select(0, cand_idx)
            scores = (cand_vec * user_vec).sum(dim=1).detach().float().cpu().numpy()
            order = np.argsort(-scores)
            ordered_ids = [cand_idx.cpu().numpy()[i].item() for i in order.tolist()]
            pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
            r = best_rank_among_positives(ordered_ids, pos_idx_set)
            ranks.append(r)

    metrics = aggregate_ranks(ranks, k=args.k)
    payload = {"split": args.split, "k": args.k, "negatives": args.negatives, "hr": metrics.hr_at_k, "ndcg": metrics.ndcg_at_k, "n": len([r for r in ranks if r is not None])}

    out_path = Path(args.out) if args.out else (artifacts_dir() / f"minimal_metrics_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
