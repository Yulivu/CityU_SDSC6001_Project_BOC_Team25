from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=str, default=str(Path(__file__).resolve().parent / "case_study_raw.json"))
    p.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "best_case_user.json"))
    p.add_argument("--min-dominant-w", type=float, default=0.45)
    p.add_argument("--min-dom-reviews", type=int, default=4)
    p.add_argument("--max-stable-var", type=float, default=1.0)
    p.add_argument("--min-unstable-var", type=float, default=1.5)
    p.add_argument("--min-var-diff", type=float, default=1.0)
    p.add_argument("--require-gt-improve", action="store_true")
    return p.parse_args()


def _best_gt_rank(gt_positions: list[dict[str, Any]]) -> int | None:
    ranks = [int(x["rank"]) for x in gt_positions if "rank" in x and x["rank"] is not None]
    return min(ranks) if ranks else None


def _read_json_any_encoding(path: Path) -> Any:
    data = path.read_bytes()
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp936"]:
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    raise SystemExit(f"Failed to decode JSON file with common encodings: {path}")


def main() -> None:
    args = parse_args()
    payload = _read_json_any_encoding(Path(args.in_path))
    if not isinstance(payload, list):
        raise SystemExit("Input JSON must be a list.")

    scored: list[tuple[float, dict[str, Any]]] = []
    for row in payload:
        aspect_weights = list(row.get("aspect_weights", []))
        if not aspect_weights:
            continue
        eligible = [
            a
            for a in aspect_weights
            if a.get("n_reviews") is not None and int(a["n_reviews"]) >= int(args.min_dom_reviews) and a.get("variance") is not None
        ]
        if len(eligible) < 2:
            continue
        variances = [float(a["variance"]) for a in eligible]
        v_min = min(variances)
        v_max = max(variances)
        if v_min > float(args.max_stable_var) or v_max < float(args.min_unstable_var) or (v_max - v_min) < float(args.min_var_diff):
            continue

        dom = max(eligible, key=lambda x: float(x.get("w_k", 0.0)))
        dom_w = float(dom.get("w_k", 0.0))
        if dom_w < float(args.min_dominant_w):
            continue

        align = row.get("alignment", {})
        magnn_m = float(((align.get("magnn") or {}).get("n_match") or 0))
        pop_m = float(((align.get("popularity") or {}).get("n_match") or 0))
        match_gain = magnn_m - pop_m

        gt_m = _best_gt_rank(list(row.get("gt_in_magnn", [])))
        gt_p = _best_gt_rank(list(row.get("gt_in_popularity", [])))
        gt_gain = 0.0
        if gt_m is not None and gt_p is not None:
            gt_gain = float(gt_p - gt_m)
        elif gt_m is not None and gt_p is None:
            gt_gain = 10.0
        elif gt_m is None and gt_p is not None:
            gt_gain = -10.0
        if bool(args.require_gt_improve) and gt_gain <= 0:
            continue

        score = match_gain * 10.0 + gt_gain + dom_w + (v_max - v_min) * 0.5
        row_out = dict(row)
        row_out["case_score"] = round(float(score), 4)
        row_out["dominant_w"] = round(float(dom_w), 4)
        row_out["match_gain_top5"] = int(match_gain)
        row_out["best_gt_rank_magnn"] = gt_m
        row_out["best_gt_rank_popularity"] = gt_p
        row_out["var_min"] = round(float(v_min), 4)
        row_out["var_max"] = round(float(v_max), 4)
        row_out["var_diff"] = round(float(v_max - v_min), 4)
        scored.append((score, row_out))

    if not scored:
        raise SystemExit("No candidates passed the filters. Try lowering --min-dominant-w.")

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
