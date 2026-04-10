from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import build_full_user_features, city_to_business_indices, load_prepared_data
from ccrec.full_model import FullModel
from ccrec.metrics import aggregate_ranks, best_rank_among_positives
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--max-hist-per-aspect", type=int, default=50)
    p.add_argument("--uniform-w", action="store_true")
    p.add_argument("--no-distance", action="store_true")
    p.add_argument("--no-popularity", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=str, default=str(artifacts_dir() / "full_model.pt"))
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    feat_dtype = torch.float16 if device.type == "cuda" else torch.float32

    prepared = load_prepared_data()
    business = prepared.business
    business_by_city = city_to_business_indices(business)

    ckpt = torch.load(args.ckpt, map_location=device)
    num_aspects = int(ckpt.get("num_aspects", 4))
    business_features = str(ckpt.get("business_features", "full"))
    aspects = ["food", "service", "atmosphere", "price"] if num_aspects == 4 else ["all"]
    user_feat = build_full_user_features(prepared, aspects=aspects, max_hist_per_aspect=args.max_hist_per_aspect)

    users = prepared.val_users if args.split == "val" else prepared.test_users
    users = [u for u in users if u in user_feat.user_to_idx]
    if args.max_users and args.max_users > 0:
        users = users[: args.max_users]

    model = FullModel(
        dim=int(ckpt["dim"]),
        num_aspects=num_aspects,
        business_in_dim=int(ckpt.get("business_in_dim", 801)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if business_features == "full":
        business_x = business.x
    elif business_features == "bert":
        business_x = business.bert
    else:
        business_x = business.x[:, 768:]

    if (args.no_popularity or bool(ckpt.get("no_popularity", False))) and business_features in {"full", "geo"}:
        business_x = business_x[:, :-1]

    x = torch.from_numpy(business_x).to(device=device, dtype=feat_dtype)
    bert = torch.from_numpy(business.bert).to(device=device, dtype=feat_dtype)

    ranks = []
    for user_id in tqdm(users, desc=f"eval_full {args.split}"):
        ui = user_feat.user_to_idx.get(user_id)
        if ui is None:
            continue

        hist_lists = [user_feat.hist_business_idx_by_aspect[ui]]
        if not any(hist_lists[0]):
            continue

        if args.uniform_w:
            w = torch.full((1, num_aspects), 1.0 / num_aspects, device=device, dtype=feat_dtype)
        else:
            w = torch.from_numpy(user_feat.w[ui : ui + 1]).to(device=device, dtype=feat_dtype)

        src_lat = float(user_feat.src_centroid_lat[ui])
        src_lon = float(user_feat.src_centroid_lon[ui])

        flat_hist = [bi for per_aspect in hist_lists[0] for bi in per_aspect]
        unique_hist = sorted(set(flat_hist))
        if not unique_hist:
            continue
        global_to_local = {g: i for i, g in enumerate(unique_hist)}
        local_hist_lists = [[[global_to_local[g] for g in per_aspect] for per_aspect in hist_lists[0]]]

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                hist_bert = bert.index_select(0, torch.tensor(unique_hist, device=device, dtype=torch.long))
                hist_emb = model.encode_business_hist(hist_bert)
                user_taste = model.user_taste(hist_emb, local_hist_lists)

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
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    b_vec = model.encode_business(x.index_select(0, cand_idx))

            tgt_lat = torch.from_numpy(business.latitude[cand_idx.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            tgt_lon = torch.from_numpy(business.longitude[cand_idx.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            if args.no_distance:
                dist = torch.zeros((len(cand_bids),), device=device, dtype=feat_dtype)
            else:
                dist = torch.log1p(torch.sqrt((tgt_lat - src_lat) ** 2 + (tgt_lon - src_lon) ** 2) + 1e-12).to(dtype=feat_dtype)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    scores = model.score(user_taste.repeat(len(cand_bids), 1, 1), w.repeat(len(cand_bids), 1), b_vec, dist).detach().float().cpu().numpy()

            order = np.argsort(-scores)
            ordered_ids = cand_idx.detach().cpu().numpy()[order].tolist()
            pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
            r = best_rank_among_positives(ordered_ids, pos_idx_set)
            ranks.append(r)

    metrics = aggregate_ranks(ranks, k=args.k)
    payload = {
        "split": args.split,
        "k": args.k,
        "negatives": args.negatives,
        "hr": metrics.hr_at_k,
        "ndcg": metrics.ndcg_at_k,
        "n": len([r for r in ranks if r is not None]),
    }

    out_path = Path(args.out) if args.out else (artifacts_dir() / f"full_metrics_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
