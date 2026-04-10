from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ASPECT_ORDER = ["food", "service", "atmosphere", "price"]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json_any_encoding(path: Path) -> Any:
    data = path.read_bytes()
    for enc in ["utf-8", "utf-8-sig", "gbk", "cp936"]:
        try:
            return json.loads(data.decode(enc))
        except Exception:
            continue
    raise SystemExit(f"Failed to decode JSON file with common encodings: {path}")


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _dominant_keywords(aspect: str) -> list[str]:
    if aspect == "food":
        return ["Restaurants", "Seafood", "Sushi", "Pizza", "Chinese", "Japanese", "Korean", "Thai", "Vietnamese", "Indian", "Mexican", "Italian"]
    if aspect == "atmosphere":
        return ["Bars", "Nightlife", "Lounges"]
    if aspect == "service":
        return ["Delivery", "Takeout", "Catering"]
    if aspect == "price":
        return ["Fast Food", "Buffets", "Food Court"]
    return []


def _is_aspect_match(categories: str | None, aspect: str) -> bool:
    if not categories:
        return False
    c = categories.lower()
    return any(k.lower() in c for k in _dominant_keywords(aspect))


def _plot_wk_variance(best: dict[str, Any], out_path: Path, dpi: int) -> None:
    aw = list(best.get("aspect_weights", []))
    by_asp = {str(a.get("aspect")): a for a in aw}
    aspects = [a for a in ASPECT_ORDER if a in by_asp]
    w = [float(by_asp[a].get("w_k", 0.0)) for a in aspects]
    var = [by_asp[a].get("variance") for a in aspects]
    var = [float(v) if v is not None else None for v in var]
    nrev = [int(by_asp[a].get("n_reviews") or 0) for a in aspects]

    x = list(range(len(aspects)))
    fig, ax1 = plt.subplots(1, 1, figsize=(7.8, 4.4))
    bars = ax1.bar(x, w, color="#2c7fb8", alpha=0.9)
    ax1.set_ylim(0.0, max(0.6, max(w) + 0.08))
    ax1.set_ylabel("Aspect Weight (w_k)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.capitalize() for a in aspects])
    ax1.set_title("Case Study: Variance-based Aspect Weights")
    ax1.grid(True, axis="y", alpha=0.25)

    for b, wi, nr in zip(bars, w, nrev):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{wi:.3f}\n(n={nr})", ha="center", va="bottom", fontsize=9)

    ax2 = ax1.twinx()
    v_y = [v if v is not None else float("nan") for v in var]
    ax2.plot(x, v_y, color="#d95f0e", marker="o", linewidth=2.0)
    ax2.set_ylabel("Rating Variance (source city)")

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _emit_top10_tables(best: dict[str, Any], out_dir: Path) -> None:
    dom = str(best.get("dominant_aspect"))
    gt_set = {str(g.get("business_id")) for g in list(best.get("ground_truth", [])) if g.get("business_id") is not None}

    def rows_for(key: str) -> list[dict[str, Any]]:
        recs = list(best.get(key, []))
        out: list[dict[str, Any]] = []
        for i, r in enumerate(recs, start=1):
            bid = str(r.get("business_id"))
            cats = r.get("categories")
            out.append(
                {
                    "rank": i,
                    "business_id": bid,
                    "name": r.get("name"),
                    "categories": cats,
                    "stars": r.get("stars"),
                    "review_count": r.get("review_count"),
                    "score": r.get("score", ""),
                    "is_ground_truth": int(bid in gt_set),
                    "match_dominant_aspect": int(_is_aspect_match(str(cats) if cats is not None else None, dom)),
                }
            )
        return out

    magnn_rows = rows_for("magnn_top10")
    pop_rows = rows_for("popularity_top10")

    header = ["rank", "business_id", "name", "categories", "stars", "review_count", "score", "is_ground_truth", "match_dominant_aspect"]
    _write_csv(out_dir / "top10_magnn.csv", header, magnn_rows)
    _write_csv(out_dir / "top10_popularity.csv", header, pop_rows)


def _plot_gt_rank_compare(best: dict[str, Any], out_path: Path, dpi: int) -> None:
    gt = list(best.get("ground_truth", []))
    gt_bids = [str(g.get("business_id")) for g in gt if g.get("business_id") is not None]
    gt_names = {str(g.get("business_id")): g.get("name") for g in gt if g.get("business_id") is not None}

    magnn_pos = {str(x.get("business_id")): int(x.get("rank")) for x in list(best.get("gt_in_magnn", [])) if x.get("rank") is not None}
    pop_pos = {str(x.get("business_id")): int(x.get("rank")) for x in list(best.get("gt_in_popularity", [])) if x.get("rank") is not None}

    items: list[tuple[str, str, int, int]] = []
    for bid in gt_bids:
        r_m = magnn_pos.get(bid, 11)
        r_p = pop_pos.get(bid, 11)
        if r_m == 11 and r_p == 11:
            continue
        items.append((bid, str(gt_names.get(bid) or bid), r_m, r_p))

    if not items:
        return

    items.sort(key=lambda x: min(x[2], x[3]))
    labels = [x[1] for x in items]
    r_m = [x[2] for x in items]
    r_p = [x[3] for x in items]

    y = list(range(len(items)))
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.6))
    ax.barh([i - 0.18 for i in y], r_m, height=0.35, color="#2c7fb8", label="MA-GNN rank (lower is better)")
    ax.barh([i + 0.18 for i in y], r_p, height=0.35, color="#bdbdbd", label="Popularity rank (lower is better)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Rank in Top-10 (11 = not in Top-10)")
    ax.set_title("Case Study: Ground-truth Rank Comparison (Top-10)")
    ax.set_xlim(0, 11.5)
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_top10_scores(best: dict[str, Any], out_path: Path, dpi: int) -> None:
    dom = str(best.get("dominant_aspect"))
    gt_set = {str(g.get("business_id")) for g in list(best.get("ground_truth", [])) if g.get("business_id") is not None}

    magnn = list(best.get("magnn_top10", []))
    names = [str(r.get("name") or r.get("business_id")) for r in magnn]
    scores = [float(r.get("score") or 0.0) for r in magnn]

    colors = []
    for r in magnn:
        bid = str(r.get("business_id"))
        is_gt = bid in gt_set
        match = _is_aspect_match(str(r.get("categories") or ""), dom)
        if is_gt:
            colors.append("#238b45")
        elif match:
            colors.append("#2c7fb8")
        else:
            colors.append("#bdbdbd")

    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.8))
    y = list(range(len(scores)))[::-1]
    ax.barh(y, scores, color=list(reversed(colors)))
    ax.set_yticks(y)
    ax.set_yticklabels(list(reversed(names)), fontsize=8)
    ax.set_xlabel("MA-GNN Score")
    ax.set_title(f"Case Study: MA-GNN Top-10 Scores (Dominant Aspect: {dom.capitalize()})")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--best", type=str, default=str(Path(__file__).resolve().parent / "best_case_user.json"))
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    report_dir = (root / "report") if not args.out_dir else Path(args.out_dir)
    fig_dir = report_dir / "figures" / "case_study"
    tab_dir = report_dir / "tables" / "case_study"

    best = _read_json_any_encoding(Path(args.best))
    _ensure_dir(fig_dir)
    _ensure_dir(tab_dir)

    _plot_wk_variance(best, fig_dir / "wk_variance.png", dpi=int(args.dpi))
    _plot_top10_scores(best, fig_dir / "magnn_top10_scores.png", dpi=int(args.dpi))
    _plot_gt_rank_compare(best, fig_dir / "gt_rank_compare.png", dpi=int(args.dpi))
    _emit_top10_tables(best, tab_dir)

    (tab_dir / "case_study_selected.json").write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved case study assets under: {report_dir}")


if __name__ == "__main__":
    main()
