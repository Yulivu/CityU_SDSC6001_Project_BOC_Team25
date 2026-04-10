from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import build_dgl_graph_data, build_full_user_features, city_to_business_indices, load_prepared_data
from ccrec.gnn_dgl import MAGNN_DGL
from ccrec.paths import artifacts_dir, outputs_dir


ASPECTS = ["food", "service", "atmosphere", "price"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-hist-per-aspect", type=int, default=50)
    p.add_argument("--batch", type=int, default=8192)
    default_ckpt = artifacts_dir() / "magnn_full_cpu_10e_es.pt"
    if not default_ckpt.exists():
        default_ckpt = artifacts_dir() / "magnn_dgl.pt"
    p.add_argument("--ckpt", type=str, default=str(default_ckpt))
    p.add_argument("--candidates", type=str, default=str(Path(__file__).resolve().parent / "candidate_users.json"))
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "case_study_raw.json"))
    return p.parse_args()


def _build_aspect_details(df_reviews: pd.DataFrame, user_id: str, source_city: str, w: np.ndarray) -> list[dict]:
    df_u = df_reviews[(df_reviews["user_id"] == user_id) & (df_reviews["city"] == source_city)]
    out: list[dict] = []
    for i, asp in enumerate(ASPECTS):
        sub = df_u[df_u["aspect"] == asp]["stars"].to_numpy(dtype=np.float32, copy=False)
        if len(sub) >= 2:
            var = float(np.var(sub))
        else:
            var = None
        out.append(
            {
                "aspect": asp,
                "n_reviews": int(len(sub)),
                "mean_stars": (round(float(np.mean(sub)), 3) if len(sub) > 0 else None),
                "variance": (round(float(var), 3) if var is not None else None),
                "w_k": round(float(w[i]), 4),
            }
        )
    return out


def _get_popularity_topk(df_business: pd.DataFrame, city: str, k: int = 10) -> list[dict]:
    city_df = df_business[df_business["city"] == city].copy()
    city_df = city_df.sort_values("review_count", ascending=False).head(k)
    cols = ["business_id", "name", "categories", "stars", "review_count"]
    return city_df[cols].to_dict("records")


def _category_keywords_for_aspect(aspect: str) -> list[str]:
    if aspect == "food":
        return [
            "Restaurants",
            "Pizza",
            "Burgers",
            "Sushi",
            "Chinese",
            "Japanese",
            "Korean",
            "Thai",
            "Vietnamese",
            "Indian",
            "Mexican",
            "Italian",
            "Seafood",
            "Steakhouses",
            "Barbeque",
            "Breakfast",
            "Brunch",
            "Desserts",
            "Bakeries",
            "Coffee",
            "Cafes",
        ]
    if aspect == "price":
        return ["Fast Food", "Food Court", "Buffets"]
    if aspect == "atmosphere":
        return ["Bars", "Nightlife", "Lounges", "Music Venues"]
    if aspect == "service":
        return ["Delivery", "Takeout", "Catering"]
    return []


def _count_topk_matches(rec_list: list[dict], aspect: str, top_k: int = 5) -> dict[str, Any]:
    kws = [k.lower() for k in _category_keywords_for_aspect(aspect)]
    matched: list[dict[str, Any]] = []
    for rec in rec_list[:top_k]:
        cats = str(rec.get("categories") or "")
        cats_l = cats.lower()
        hit = any(k in cats_l for k in kws) if kws else False
        if hit:
            matched.append({"business_id": str(rec.get("business_id")), "name": rec.get("name"), "categories": cats})
    return {"top_k": int(top_k), "aspect": aspect, "n_match": int(len(matched)), "matched": matched}


def _read_json_any_encoding(path: Path) -> Any:
    data = path.read_bytes()
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp936"]:
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    raise SystemExit(f"Failed to decode JSON file with common encodings: {path}")


def _find_gt_positions(rec_list: list[dict], gt_bids: list[str]) -> list[dict]:
    pos: list[dict] = []
    wanted = set(map(str, gt_bids))
    for rank, rec in enumerate(rec_list, start=1):
        bid = str(rec.get("business_id"))
        if bid in wanted:
            pos.append({"business_id": bid, "rank": int(rank)})
    return pos


def _magnn_city_topk(
    model: MAGNN_DGL,
    all_b: torch.Tensor,
    u_taste: torch.Tensor,
    w: torch.Tensor,
    business_lat: np.ndarray,
    business_lon: np.ndarray,
    city_idxs: np.ndarray,
    src_lat: float,
    src_lon: float,
    batch: int,
    device: torch.device,
    feat_dtype: torch.dtype,
    no_distance: bool,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((len(city_idxs),), dtype=np.float32)
    city_idxs_t = torch.from_numpy(city_idxs.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)

    for start in range(0, len(city_idxs), batch):
        end = min(start + batch, len(city_idxs))
        idx_batch = city_idxs_t[start:end]
        b_vec = all_b.index_select(0, idx_batch)

        idx_np = idx_batch.detach().cpu().numpy()
        if no_distance:
            dist = torch.zeros((len(idx_np),), device=device, dtype=feat_dtype)
        else:
            tgt_lat = torch.from_numpy(business_lat[idx_np]).to(device=device, dtype=torch.float32)
            tgt_lon = torch.from_numpy(business_lon[idx_np]).to(device=device, dtype=torch.float32)
            dist = torch.log1p(torch.sqrt((tgt_lat - src_lat) ** 2 + (tgt_lon - src_lon) ** 2) + 1e-12).to(dtype=feat_dtype)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                s = model.score(u_taste.repeat(len(idx_np), 1, 1), w.repeat(len(idx_np), 1), b_vec, dist)
        scores[start:end] = s.detach().float().cpu().numpy().reshape(-1)

    order = np.argsort(-scores)[:k]
    return city_idxs[order], scores[order]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    feat_dtype = torch.float16 if device.type == "cuda" else torch.float32

    try:
        import dgl  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "Missing dependency: dgl. Install the DGL environment first (see requirements-dgl.txt). "
            f"Original error: {e}"
        )

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

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
    aspects = ["all"] if (no_aspect or num_aspects == 1) else ASPECTS

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

    out_dir = outputs_dir()
    required = [
        out_dir / "business_filtered.parquet",
        out_dir / "business_geo_features.parquet",
        out_dir / "business_bert_embeddings.parquet",
        out_dir / "reviews_with_aspects.parquet",
        out_dir / "users_train.parquet",
        out_dir / "users_val.parquet",
        out_dir / "users_test.parquet",
        out_dir / "graph_user_social_edges.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Missing required files (run data_prep pipeline step1~5 to regenerate outputs/):\n" + "\n".join(
            [f"- {p}" for p in missing]
        )
        raise SystemExit(msg)

    df_business = pd.read_parquet(
        out_dir / "business_filtered.parquet",
        columns=["business_id", "name", "city", "categories", "stars", "review_count"],
    )
    df_business["business_id"] = df_business["business_id"].astype(str)
    df_business["city"] = df_business["city"].astype(str)
    df_business["review_count"] = df_business["review_count"].astype(np.float32).fillna(0.0)

    df_reviews = pd.read_parquet(out_dir / "reviews_with_aspects.parquet", columns=["user_id", "city", "stars", "aspect"])
    df_reviews["user_id"] = df_reviews["user_id"].astype(str)
    df_reviews["city"] = df_reviews["city"].astype(str)
    df_reviews["aspect"] = df_reviews["aspect"].astype(str)
    df_reviews["stars"] = df_reviews["stars"].astype(np.float32)

    candidates: list[dict[str, Any]] = list(_read_json_any_encoding(Path(args.candidates)))

    results: list[dict[str, Any]] = []
    for cand in candidates:
        uid = str(cand["user_id"])
        src_city = str(cand["source_city"])
        tgt_city = str(cand["target_city"])

        if tgt_city not in business_by_city:
            continue
        if uid not in graph_data.user_to_idx or uid not in user_feat.user_to_idx:
            continue

        ui = graph_data.user_to_idx[uid]
        u_taste = all_user_taste[ui : ui + 1]

        if uniform_w:
            w_np = np.full((num_aspects,), 1.0 / max(num_aspects, 1), dtype=np.float32)
            w = torch.from_numpy(w_np.reshape(1, -1)).to(device=device, dtype=feat_dtype)
        else:
            uf = user_feat.user_to_idx[uid]
            w_np = user_feat.w[uf].astype(np.float32, copy=True)
            w = torch.from_numpy(w_np.reshape(1, -1)).to(device=device, dtype=feat_dtype)

        if len(aspects) == 1:
            aspect_details = [
                {
                    "aspect": "all",
                    "n_reviews": None,
                    "mean_stars": None,
                    "variance": None,
                    "w_k": round(float(w_np[0]), 4),
                }
            ]
            dominant_aspect = "all"
        else:
            aspect_details = _build_aspect_details(df_reviews, uid, src_city, w_np)
            eligible = [
                a
                for a in aspect_details
                if a.get("n_reviews") is not None and int(a["n_reviews"]) >= 2 and a.get("variance") is not None
            ]
            dominant_aspect = max((eligible or aspect_details), key=lambda x: float(x["w_k"]))["aspect"]

        src_lat = float(user_feat.src_centroid_lat[user_feat.user_to_idx[uid]])
        src_lon = float(user_feat.src_centroid_lon[user_feat.user_to_idx[uid]])
        city_idxs = business_by_city[tgt_city]
        top_idx, top_score = _magnn_city_topk(
            model=model,
            all_b=all_b,
            u_taste=u_taste,
            w=w,
            business_lat=business.latitude,
            business_lon=business.longitude,
            city_idxs=city_idxs,
            src_lat=src_lat,
            src_lon=src_lon,
            batch=int(args.batch),
            device=device,
            feat_dtype=feat_dtype,
            no_distance=no_distance,
            k=10,
        )

        top_bids = [business.business_ids[int(i)] for i in top_idx.tolist()]
        df_top = df_business[df_business["business_id"].isin(set(map(str, top_bids)))].drop_duplicates(subset=["business_id"])
        meta = df_top.set_index("business_id")[["name", "categories", "stars", "review_count"]].to_dict("index")

        magnn_top10: list[dict[str, Any]] = []
        for bid, score in zip(top_bids, top_score.tolist()):
            info = meta.get(str(bid), {})
            magnn_top10.append(
                {
                    "business_id": str(bid),
                    "name": info.get("name"),
                    "categories": info.get("categories"),
                    "stars": (float(info["stars"]) if "stars" in info and info["stars"] is not None else None),
                    "review_count": (float(info["review_count"]) if "review_count" in info and info["review_count"] is not None else None),
                    "score": round(float(score), 6),
                }
            )

        pop_top10 = _get_popularity_topk(df_business, tgt_city, k=10)
        ground_truth = list(cand.get("ground_truth", []))
        gt_bids = [str(g.get("business_id")) for g in ground_truth if "business_id" in g]

        magnn_match = _count_topk_matches(magnn_top10, dominant_aspect, top_k=5)
        pop_match = _count_topk_matches(pop_top10, dominant_aspect, top_k=5)

        results.append(
            {
                "user_id": uid,
                "source_city": src_city,
                "target_city": tgt_city,
                "source_n_reviews": int(cand.get("source_n", 0)),
                "aspect_weights": aspect_details,
                "dominant_aspect": dominant_aspect,
                "alignment": {"magnn": magnn_match, "popularity": pop_match},
                "magnn_top10": magnn_top10,
                "popularity_top10": pop_top10,
                "ground_truth": ground_truth,
                "gt_in_magnn": _find_gt_positions(magnn_top10, gt_bids),
                "gt_in_popularity": _find_gt_positions(pop_top10, gt_bids),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
