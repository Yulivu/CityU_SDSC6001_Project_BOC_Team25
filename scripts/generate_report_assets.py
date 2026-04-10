from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


@dataclass(frozen=True)
class ResultRow:
    model_family: str
    variant: str
    exp_id: str
    split: str
    k: int
    negatives: int
    hr: float
    ndcg: float
    n: int
    best_epoch: int | None
    best_val_ndcg: float | None
    ckpt: str
    history_csv: str
    meta_json: str


def _as_int(x: Any) -> int | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _as_float(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_report_results(path: Path) -> list[ResultRow]:
    rows: list[ResultRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                ResultRow(
                    model_family=str(row.get("model_family", "")).strip(),
                    variant=str(row.get("variant", "")).strip(),
                    exp_id=str(row.get("exp_id", "")).strip(),
                    split=str(row.get("split", "")).strip(),
                    k=int(float(row.get("k", 0) or 0)),
                    negatives=int(float(row.get("negatives", 0) or 0)),
                    hr=float(row.get("hr", 0) or 0),
                    ndcg=float(row.get("ndcg", 0) or 0),
                    n=int(float(row.get("n", 0) or 0)),
                    best_epoch=_as_int(row.get("best_epoch", "")),
                    best_val_ndcg=_as_float(row.get("best_val_ndcg", "")),
                    ckpt=str(row.get("ckpt", "")).strip(),
                    history_csv=str(row.get("history_csv", "")).strip(),
                    meta_json=str(row.get("meta_json", "")).strip(),
                )
            )
    return rows


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _write_markdown_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    _ensure_dir(path.parent)
    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}"


def _select_rows(
    all_rows: list[ResultRow], split: str, exp_order: list[str]
) -> list[ResultRow]:
    lookup = {r.exp_id: r for r in all_rows if r.split == split}
    out: list[ResultRow] = []
    for exp_id in exp_order:
        r = lookup.get(exp_id)
        if r is not None:
            out.append(r)
    return out


def _bold_if(x: str, cond: bool) -> str:
    return f"**{x}**" if cond else x


def _emit_tables(rows: list[ResultRow], report_dir: Path) -> None:
    main_order = [
        "baseline_popularity",
        "baseline_bert_only",
        "mf_10e_es",
        "full_10e_es",
        "magnn_full_cpu_10e_es",
    ]
    ablation_order = [
        "full_no_distance_10e_es",
        "full_single_aspect_10e_es",
        "magnn_no_aspect_cpu_10e_es",
        "magnn_no_social_cpu_10e_es",
    ]

    for split in ["val", "test"]:
        main = _select_rows(rows, split=split, exp_order=main_order)
        abl = _select_rows(rows, split=split, exp_order=ablation_order)

        if main:
            max_hr = max(r.hr for r in main)
            max_ndcg = max(r.ndcg for r in main)
            md_rows: list[list[str]] = []
            csv_rows: list[dict[str, Any]] = []
            for r in main:
                csv_rows.append(
                    {
                        "model": r.variant,
                        "HR@10": r.hr,
                        "NDCG@10": r.ndcg,
                        "n": r.n,
                    }
                )
                md_rows.append(
                    [
                        r.variant,
                        _bold_if(_fmt(r.hr), r.hr == max_hr),
                        _bold_if(_fmt(r.ndcg), r.ndcg == max_ndcg),
                        str(r.n),
                    ]
                )

            _write_csv(
                report_dir / "tables" / f"main_{split}.csv",
                header=["model", "HR@10", "NDCG@10", "n"],
                rows=csv_rows,
            )
            _write_markdown_table(
                report_dir / "tables" / f"main_{split}.md",
                header=["Model", "HR@10", "NDCG@10", "n"],
                rows=md_rows,
            )

        if abl:
            md_rows = []
            csv_rows = []
            for r in abl:
                csv_rows.append(
                    {
                        "model": r.variant,
                        "HR@10": r.hr,
                        "NDCG@10": r.ndcg,
                        "n": r.n,
                    }
                )
                md_rows.append([r.variant, _fmt(r.hr), _fmt(r.ndcg), str(r.n)])

            _write_csv(
                report_dir / "tables" / f"ablations_{split}.csv",
                header=["model", "HR@10", "NDCG@10", "n"],
                rows=csv_rows,
            )
            _write_markdown_table(
                report_dir / "tables" / f"ablations_{split}.md",
                header=["Model", "HR@10", "NDCG@10", "n"],
                rows=md_rows,
            )


def _read_history(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"epoch": [], "train_loss": [], "val_hr": [], "val_ndcg": []}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out["epoch"].append(float(row.get("epoch", 0) or 0))
            out["train_loss"].append(float(row.get("train_loss", 0) or 0))
            out["val_hr"].append(float(row.get("val_hr", 0) or 0))
            out["val_ndcg"].append(float(row.get("val_ndcg", 0) or 0))
    last_reset = 0
    for i in range(1, len(out["epoch"])):
        if out["epoch"][i] <= out["epoch"][i - 1]:
            last_reset = i
    if last_reset > 0:
        for k in list(out.keys()):
            out[k] = out[k][last_reset:]
    return out


def _plot_one_history(exp_id: str, title: str, hist_path: Path, out_path: Path, dpi: int) -> None:
    h = _read_history(hist_path)
    epochs = h["epoch"]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.2), sharex=True)
    fig.suptitle(title)

    axes[0].plot(epochs, h["train_loss"], marker="o", linewidth=1.8)
    axes[0].set_ylabel("Train Loss")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, h["val_hr"], marker="o", linewidth=1.8)
    axes[1].set_ylabel("Val HR@10")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(epochs, h["val_ndcg"], marker="o", linewidth=1.8)
    axes[2].set_ylabel("Val NDCG@10")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.25)

    best_i = max(range(len(epochs)), key=lambda i: h["val_ndcg"][i]) if epochs else None
    if best_i is not None:
        axes[2].scatter([epochs[best_i]], [h["val_ndcg"][best_i]], s=60, zorder=5)

    _ensure_dir(out_path.parent)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _plot_overlay_val_ndcg(
    series: list[tuple[str, str, Path]], out_path: Path, dpi: int
) -> None:
    label_by_exp = {
        "baseline_popularity": "Popularity",
        "baseline_bert_only": "BERT-only",
        "mf_10e_es": "Matrix Factorization",
        "full_10e_es": "Full (-GNN)",
        "full_no_distance_10e_es": "Full (-Geography)",
        "full_single_aspect_10e_es": "Full (-Multi-aspect)",
        "magnn_no_aspect_cpu_10e_es": "MA-GNN (-Aspect)",
        "magnn_no_social_cpu_10e_es": "MA-GNN (-Social)",
        "magnn_full_cpu_10e_es": "MA-GNN Full",
    }

    mf = [t for t in series if t[0] == "mf_10e_es"]
    top = [t for t in series if t[0] != "mf_10e_es"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.4, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    color_by_exp: dict[str, str] = {}
    p_i = 0
    for exp_id, _label, _p in top:
        if exp_id == "magnn_full_cpu_10e_es":
            color_by_exp[exp_id] = "#111111"
        else:
            color_by_exp[exp_id] = palette[p_i % len(palette)]
            p_i += 1

    for exp_id, _label, p in top:
        h = _read_history(p)
        label = label_by_exp.get(exp_id, _label)
        lw = 2.8 if exp_id == "magnn_full_cpu_10e_es" else 1.7
        ms = 5.0 if exp_id == "magnn_full_cpu_10e_es" else 4.0
        ax1.plot(
            h["epoch"],
            h["val_ndcg"],
            marker="o",
            markersize=ms,
            linewidth=lw,
            color=color_by_exp.get(exp_id, None),
            label=label,
        )
    ax1.set_title("Validation NDCG@10 vs Epoch")
    ax1.set_ylabel("Val NDCG@10")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", fontsize=8, ncols=2)

    if mf:
        exp_id, _label, p = mf[0]
        h = _read_history(p)
        ax2.plot(
            h["epoch"],
            h["val_ndcg"],
            marker="o",
            markersize=4.2,
            linewidth=1.8,
            color="#7f7f7f",
            label=label_by_exp.get(exp_id, _label),
        )
        ax2.legend(loc="best", fontsize=8)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val NDCG@10")
    ax2.grid(True, alpha=0.25)
    _ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _emit_summary_barplots_and_deltas(rows: list[ResultRow], report_dir: Path, dpi: int) -> None:
    label_by_exp = {
        "baseline_popularity": "Popularity",
        "baseline_bert_only": "BERT-only",
        "mf_10e_es": "Matrix Factorization",
        "full_10e_es": "Full (-GNN)",
        "full_no_distance_10e_es": "Full (-Geography)",
        "full_single_aspect_10e_es": "Full (-Multi-aspect)",
        "magnn_no_aspect_cpu_10e_es": "MA-GNN (-Aspect)",
        "magnn_no_social_cpu_10e_es": "MA-GNN (-Social)",
        "magnn_full_cpu_10e_es": "MA-GNN Full",
    }
    order = [
        "baseline_popularity",
        "baseline_bert_only",
        "mf_10e_es",
        "full_10e_es",
        "full_no_distance_10e_es",
        "full_single_aspect_10e_es",
        "magnn_no_aspect_cpu_10e_es",
        "magnn_no_social_cpu_10e_es",
        "magnn_full_cpu_10e_es",
    ]

    by_exp_test = {r.exp_id: r for r in rows if r.split == "test"}
    xs = []
    labels = []
    hr = []
    ndcg = []
    colors_baseline = "#c7c7c7"
    colors_ablation = "#9ecae1"
    color_highlight = "#08306b"
    for i, exp_id in enumerate(order):
        r = by_exp_test.get(exp_id)
        if r is None:
            continue
        xs.append(i)
        labels.append(label_by_exp.get(exp_id, exp_id))
        hr.append(float(r.hr))
        ndcg.append(float(r.ndcg))

    if xs:
        def annotate(ax: plt.Axes, bars: Any, vals: list[float]) -> None:
            for b, v in zip(list(bars), vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{v:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=0,
                )

        baseline_names = {"Popularity", "BERT-only", "Matrix Factorization"}
        baseline_idx = [i for i, lbl in enumerate(labels) if lbl in baseline_names]
        ablation_idx = [i for i, lbl in enumerate(labels) if lbl not in baseline_names]

        def plot_metric(values: list[float], ylabel: str, out_name: str, title: str) -> None:
            fig, (ax_b, ax_a) = plt.subplots(
                1,
                2,
                figsize=(12.6, 4.8),
                gridspec_kw={"width_ratios": [1.0, 2.2]},
            )

            b_labels = [labels[i] for i in baseline_idx]
            b_vals = [values[i] for i in baseline_idx]
            a_labels = [labels[i] for i in ablation_idx]
            a_vals = [values[i] for i in ablation_idx]

            b_bars = ax_b.bar(range(len(b_vals)), b_vals, color=colors_baseline)
            ax_b.set_title("Baselines")
            ax_b.set_ylabel(ylabel)
            ax_b.set_xticks(range(len(b_labels)))
            ax_b.set_xticklabels(b_labels, rotation=20, ha="right")
            ax_b.grid(True, axis="y", alpha=0.25)
            if b_vals:
                ax_b.set_ylim(0.0, max(b_vals) + 0.05)
                annotate(ax_b, b_bars, b_vals)

            a_colors = [colors_ablation] * len(a_vals)
            for j, lbl in enumerate(a_labels):
                if lbl == "MA-GNN Full":
                    a_colors[j] = color_highlight
            a_bars = ax_a.bar(range(len(a_vals)), a_vals, color=a_colors)
            ax_a.set_title("Ablation Variants (Zoomed)")
            ax_a.set_xticks(range(len(a_labels)))
            ax_a.set_xticklabels(a_labels, rotation=20, ha="right")
            ax_a.grid(True, axis="y", alpha=0.25)
            if a_vals:
                span = max(a_vals) - min(a_vals)
                pad = max(0.002, span * 0.35)
                ax_a.set_ylim(min(a_vals) - pad, max(a_vals) + pad)
                annotate(ax_a, a_bars, a_vals)

            fig.suptitle(title)
            _ensure_dir((report_dir / "figures").resolve())
            fig.tight_layout()
            fig.savefig(report_dir / "figures" / out_name, dpi=dpi)
            plt.close(fig)

        plot_metric(
            hr,
            ylabel="HR@10",
            out_name="test_hr_bar.png",
            title="Test HR@10: Baselines vs Ablation Variants",
        )
        plot_metric(
            ndcg,
            ylabel="NDCG@10",
            out_name="test_ndcg_bar.png",
            title="Test NDCG@10: Baselines vs Ablation Variants",
        )

    base = by_exp_test.get("magnn_full_cpu_10e_es")
    if base is not None:
        compare = [
            "full_10e_es",
            "full_no_distance_10e_es",
            "full_single_aspect_10e_es",
            "magnn_no_aspect_cpu_10e_es",
            "magnn_no_social_cpu_10e_es",
        ]
        delta_labels = []
        deltas = []
        for exp_id in compare:
            r = by_exp_test.get(exp_id)
            if r is None:
                continue
            delta_labels.append(label_by_exp.get(exp_id, exp_id))
            deltas.append(float(r.ndcg) - float(base.ndcg))

        if deltas:
            up = "#2ca02c"
            down = "#d62728"
            colors = [up if d >= 0 else down for d in deltas]
            fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.6))
            ax.bar(range(len(deltas)), deltas, color=colors)
            ax.axhline(0.0, color="#333333", linewidth=1.0)
            ax.set_title("Ablation Delta vs MA-GNN Full (Test NDCG@10)")
            ax.set_ylabel("Δ NDCG@10 (Model − MA-GNN Full)")
            ax.set_xticks(range(len(delta_labels)))
            ax.set_xticklabels(delta_labels, rotation=25, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(
                handles=[
                    Patch(facecolor=up, label="Higher than MA-GNN Full"),
                    Patch(facecolor=down, label="Lower than MA-GNN Full"),
                ],
                loc="best",
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(report_dir / "figures" / "ablation_delta_ndcg.png", dpi=dpi)
            plt.close(fig)


def _emit_curves(rows: list[ResultRow], root: Path, report_dir: Path, dpi: int) -> None:
    by_exp_val = {r.exp_id: r for r in rows if r.split == "val"}
    items: list[tuple[str, str, Path]] = []

    for exp_id, r in sorted(by_exp_val.items(), key=lambda x: x[0]):
        if not r.history_csv:
            continue
        hist_path = Path(r.history_csv)
        if not hist_path.is_absolute():
            hist_path = root / hist_path
        if not hist_path.exists():
            continue

        out_path = report_dir / "figures" / "curves" / f"{exp_id}.png"
        _plot_one_history(exp_id=exp_id, title=r.variant, hist_path=hist_path, out_path=out_path, dpi=dpi)
        items.append((exp_id, r.variant, hist_path))

    overlay_keep = [
        "mf_10e_es",
        "full_10e_es",
        "full_no_distance_10e_es",
        "full_single_aspect_10e_es",
        "magnn_no_aspect_cpu_10e_es",
        "magnn_no_social_cpu_10e_es",
        "magnn_full_cpu_10e_es",
    ]
    overlay = [(eid, lbl, p) for (eid, lbl, p) in items if eid in set(overlay_keep)]
    if overlay:
        overlay.sort(key=lambda x: overlay_keep.index(x[0]))
        _plot_overlay_val_ndcg(overlay, report_dir / "figures" / "val_ndcg_overlay.png", dpi=dpi)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=str, default="")
    p.add_argument("--report-dir", type=str, default="")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    art = Path(args.artifacts) if args.artifacts else (root / "artifacts")
    report_dir = Path(args.report_dir) if args.report_dir else (root / "report")

    src = art / "report_results.csv"
    rows = _read_report_results(src)
    _emit_tables(rows, report_dir=report_dir)
    _emit_curves(rows, root=root, report_dir=report_dir, dpi=int(args.dpi))
    _emit_summary_barplots_and_deltas(rows, report_dir=report_dir, dpi=int(args.dpi))

    print(f"saved tables/figures under: {report_dir}")


if __name__ == "__main__":
    main()
