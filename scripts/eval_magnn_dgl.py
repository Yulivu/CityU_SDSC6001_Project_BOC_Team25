from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import build_dgl_graph_data, build_full_user_features, city_to_business_indices, load_prepared_data
from ccrec.gnn_dgl import MAGNN_DGL
from ccrec.metrics import aggregate_ranks, best_rank_among_positives
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--max-hist-per-aspect", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ckpt", type=str, default=str(artifacts_dir() / "magnn_dgl.pt"))
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    feat_dtype = torch.float16 if device.type == "cuda" else torch.float32

    import dgl

    ckpt = torch.load(args.ckpt, map_location=device)
    num_aspects = int(ckpt.get("num_aspects", 4))
    no_social = bool(ckpt.get("no_social", False))
    no_aspect = bool(ckpt.get("no_aspect", False))
    use_cross_attn = bool(ckpt.get("use_cross_attn", False))
    uniform_w = bool(ckpt.get("uniform_w", False))
    no_distance = bool(ckpt.get("no_distance", False))

    prepared = load_prepared_data()
    business = prepared.business
    business_by_city = city_to_business_indices(business)
    aspects = ["all"] if (no_aspect or num_aspects == 1) else ["food", "service", "atmosphere", "price"]

    user_feat = build_full_user_features(prepared, aspects=aspects, max_hist_per_aspect=args.max_hist_per_aspect)
    graph_data = build_dgl_graph_data(
        prepared,
        aspects=aspects,
        max_hist_per_aspect=args.max_hist_per_aspect,
        include_social=(not no_social),
        collapse_aspects=bool(no_aspect or num_aspects == 1),
    )

    g = dgl.heterograph(
        {
            ("user", "ub", "business"): (torch.from_numpy(graph_data.ub_src), torch.from_numpy(graph_data.ub_dst)),
            ("business", "bu", "user"): (torch.from_numpy(graph_data.ub_dst), torch.from_numpy(graph_data.ub_src)),
            ("user", "uu", "user"): (
                torch.from_numpy(np.concatenate([graph_data.uu_src, graph_data.uu_dst])),
                torch.from_numpy(np.concatenate([graph_data.uu_dst, graph_data.uu_src])),
            ),
        },
        num_nodes_dict={"user": len(graph_data.user_ids), "business": len(graph_data.business_ids)},
    ).to(device)

    ub_src = torch.from_numpy(graph_data.ub_src).to(device=device, dtype=torch.long)
    ub_dst = torch.from_numpy(graph_data.ub_dst).to(device=device, dtype=torch.long)
    ub_aspect = torch.from_numpy(graph_data.ub_aspect).to(device=device, dtype=torch.long)

    x = torch.from_numpy(business.x).to(device=device, dtype=feat_dtype)
    bert = torch.from_numpy(business.bert).to(device=device, dtype=feat_dtype)

    users = prepared.val_users if args.split == "val" else prepared.test_users
    eligible = {u for (u, _city) in prepared.user_city_split.target_city_positives.keys()}
    users = [u for u in users if u in eligible and u in graph_data.user_to_idx and u in user_feat.user_to_idx]
    if args.max_users and args.max_users > 0 and len(users) > args.max_users:
        idx = rng.choice(len(users), size=args.max_users, replace=False)
        users = [users[i] for i in idx.tolist()]

    model = MAGNN_DGL(
        num_users=len(graph_data.user_ids),
        business_bert_dim=768,
        business_in_dim=int(x.shape[1]),
        dim=int(ckpt.get("dim", 128)),
        num_aspects=num_aspects,
        num_heads=4,
        dropout=0.2,
        use_cross_attention=use_cross_attn,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            all_user_h, all_business_h = model.graph_forward(g, torch.arange(len(graph_data.user_ids), device=device), bert)
            all_user_taste = model.user_taste_from_edges(all_user_h, all_business_h, ub_src, ub_dst, ub_aspect)
            all_b = model.business_encoder(x)

    ranks = []
    for user_id in tqdm(users, desc=f"eval_magnn_dgl {args.split}"):
        ui = graph_data.user_to_idx[user_id]
        u_taste = all_user_taste[ui : ui + 1]

        if uniform_w:
            w = torch.full((1, num_aspects), 1.0 / num_aspects, device=device, dtype=feat_dtype)
        else:
            uf = user_feat.user_to_idx[user_id]
            w = torch.from_numpy(user_feat.w[uf : uf + 1]).to(device=device, dtype=feat_dtype)

        src_lat = float(user_feat.src_centroid_lat[user_feat.user_to_idx[user_id]])
        src_lon = float(user_feat.src_centroid_lon[user_feat.user_to_idx[user_id]])

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
            b_vec = all_b.index_select(0, cand_idx)

            tgt_lat = torch.from_numpy(business.latitude[cand_idx.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            tgt_lon = torch.from_numpy(business.longitude[cand_idx.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            if no_distance:
                dist = torch.zeros((len(cand_bids),), device=device, dtype=feat_dtype)
            else:
                dist = torch.log1p(torch.sqrt((tgt_lat - src_lat) ** 2 + (tgt_lon - src_lon) ** 2) + 1e-12).to(dtype=feat_dtype)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    scores = model.score(u_taste.repeat(len(cand_bids), 1, 1), w.repeat(len(cand_bids), 1), b_vec, dist).detach().float().cpu().numpy()

            order = np.argsort(-scores)
            ordered_ids = cand_idx.detach().cpu().numpy()[order].tolist()
            pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
            r = best_rank_among_positives(ordered_ids, pos_idx_set)
            ranks.append(r)

    metrics = aggregate_ranks(ranks, k=args.k)
    payload = {"split": args.split, "k": args.k, "negatives": args.negatives, "hr": metrics.hr_at_k, "ndcg": metrics.ndcg_at_k, "n": len([r for r in ranks if r is not None])}

    out_path = Path(args.out) if args.out else (artifacts_dir() / f"magnn_dgl_metrics_{args.split}.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
