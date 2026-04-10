from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.paths import outputs_dir


ASPECTS = ["food", "service", "atmosphere", "price"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--min-source-reviews", type=int, default=20)
    p.add_argument("--min-aspect-reviews", type=int, default=3)
    p.add_argument("--stable-var-threshold", type=float, default=0.8)
    p.add_argument("--unstable-var-threshold", type=float, default=1.8)
    p.add_argument("--min-var-diff", type=float, default=1.2)
    p.add_argument("--min-target-gt", type=int, default=3)
    p.add_argument("--min-target-pool", type=int, default=100)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "candidate_users.json"))
    return p.parse_args()


def _aspect_stats(df_user_src: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for asp in ASPECTS:
        sub = df_user_src[df_user_src["aspect"] == asp]["stars"].to_numpy(dtype=np.float32, copy=False)
        if len(sub) >= 2:
            var = float(np.var(sub))
        else:
            var = None
        rows.append(
            {
                "aspect": asp,
                "n": int(len(sub)),
                "mean_stars": (round(float(np.mean(sub)), 3) if len(sub) > 0 else None),
                "variance": (round(float(var), 3) if var is not None else None),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = outputs_dir()
    case_dir = Path(__file__).resolve().parent
    case_dir.mkdir(parents=True, exist_ok=True)

    required = [
        out_dir / "reviews_with_aspects.parquet",
        out_dir / "users_test.parquet",
        out_dir / "business_filtered.parquet",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "Missing required files:\n" + "\n".join([f"- {p}" for p in missing])
        raise SystemExit(msg)

    df_reviews = pd.read_parquet(out_dir / "reviews_with_aspects.parquet", columns=["user_id", "business_id", "city", "stars", "aspect"])
    df_reviews["user_id"] = df_reviews["user_id"].astype(str)
    df_reviews["business_id"] = df_reviews["business_id"].astype(str)
    df_reviews["city"] = df_reviews["city"].astype(str)
    df_reviews["aspect"] = df_reviews["aspect"].astype(str)
    df_reviews["stars"] = df_reviews["stars"].astype(np.float32)

    test_users = pd.read_parquet(out_dir / "users_test.parquet", columns=["user_id"])["user_id"].astype(str).tolist()
    df_reviews = df_reviews[df_reviews["user_id"].isin(set(test_users))]

    df_business = pd.read_parquet(out_dir / "business_filtered.parquet", columns=["business_id", "name", "city", "categories", "stars", "review_count"])
    df_business["business_id"] = df_business["business_id"].astype(str)
    df_business["city"] = df_business["city"].astype(str)

    city_counts = df_reviews.groupby(["user_id", "city"], sort=False).size().reset_index(name="n")
    city_counts = city_counts.sort_values(["user_id", "n", "city"], ascending=[True, False, True])
    src = city_counts.drop_duplicates(subset=["user_id"], keep="first").rename(columns={"city": "source_city", "n": "source_n"})
    src = src[src["source_n"] >= args.min_source_reviews]

    business_pool_by_city = df_business.groupby("city", sort=False).size().to_dict()

    candidates: list[dict] = []
    for row in src.itertuples(index=False):
        uid = str(row.user_id)
        src_city = str(row.source_city)
        src_reviews = df_reviews[(df_reviews["user_id"] == uid) & (df_reviews["city"] == src_city)]

        stats = _aspect_stats(src_reviews)
        usable = [s for s in stats if s["n"] >= args.min_aspect_reviews and s["variance"] is not None]
        if not usable:
            continue

        stable = [s for s in usable if float(s["variance"]) <= args.stable_var_threshold]
        unstable = [s for s in usable if float(s["variance"]) >= args.unstable_var_threshold]
        if not stable or not unstable:
            continue

        variances = [float(s["variance"]) for s in usable]
        if (max(variances) - min(variances)) < args.min_var_diff:
            continue

        tgt_reviews = df_reviews[(df_reviews["user_id"] == uid) & (df_reviews["city"] != src_city)]
        tgt_city_counts = tgt_reviews.groupby("city", sort=False).size()
        valid_targets = [c for c, n in tgt_city_counts.items() if int(n) >= args.min_target_gt]
        valid_targets = [c for c in valid_targets if int(business_pool_by_city.get(c, 0)) >= args.min_target_pool]
        if not valid_targets:
            continue

        best_target = max(valid_targets, key=lambda c: int(tgt_city_counts[c]))
        gt_bids = tgt_reviews[tgt_reviews["city"] == best_target]["business_id"].astype(str).tolist()
        gt_info = (
            df_business[df_business["business_id"].isin(set(gt_bids))][["business_id", "name", "categories", "stars", "review_count"]]
            .drop_duplicates(subset=["business_id"])
            .to_dict("records")
        )

        candidates.append(
            {
                "user_id": uid,
                "source_city": src_city,
                "target_city": str(best_target),
                "source_n": int(row.source_n),
                "aspect_stats": stats,
                "ground_truth": gt_info,
            }
        )
        if len(candidates) >= args.top_n:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Found {len(candidates)} candidate users.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
