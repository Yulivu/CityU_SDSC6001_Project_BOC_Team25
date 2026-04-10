from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_family(exp_id: str) -> str:
    if exp_id.startswith("full"):
        return "Full"
    if exp_id.startswith("magnn"):
        return "MA-GNN"
    if exp_id.startswith("mf"):
        return "MF"
    if exp_id.startswith("baseline"):
        return "Baseline"
    return "Other"


def _variant(exp_id: str) -> str:
    if exp_id == "full_10e_es":
        return "Ours (-GNN)"
    if exp_id == "magnn_full_cpu_10e_es":
        return "Ours (MA-GNN Full)"
    if exp_id == "magnn_no_social_cpu_10e_es":
        return "Ours (MA-GNN -Social)"
    if exp_id == "magnn_no_aspect_cpu_10e_es":
        return "Ours (MA-GNN -Aspect)"
    if exp_id == "full_no_distance_10e_es":
        return "Ours (-Geography)"
    if exp_id == "full_single_aspect_10e_es":
        return "Ours (-Multi-aspect)"
    if exp_id == "mf_10e_es":
        return "Matrix Factorization"
    return exp_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=str, default="")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    art = Path(args.artifacts) if args.artifacts else (root / "artifacts")
    out_path = Path(args.out) if args.out else (art / "report_results.csv")

    rows: list[dict[str, Any]] = []

    for split in ["val", "test"]:
        baseline_path = art / f"baseline_metrics_{split}.json"
        if baseline_path.exists():
            base = _read_json(baseline_path)
            for key, label in [("popularity", "Popularity"), ("bert_only", "BERT-only")]:
                m = base.get(key, {})
                rows.append(
                    {
                        "model_family": "Baseline",
                        "variant": label,
                        "exp_id": f"baseline_{label.replace('-', '_').lower()}",
                        "split": split,
                        "k": base.get("k"),
                        "negatives": base.get("negatives"),
                        "hr": m.get("hr"),
                        "ndcg": m.get("ndcg"),
                        "n": m.get("n"),
                        "best_epoch": "",
                        "best_val_ndcg": "",
                        "meta_json": str(baseline_path.as_posix()),
                        "history_csv": "",
                        "ckpt": "",
                    }
                )

    for path in sorted(art.glob("*_val.json")) + sorted(art.glob("*_test.json")):
        if path.name.startswith("baseline_metrics_"):
            continue
        payload = _read_json(path)
        split = str(payload.get("split", ""))
        stem = path.stem
        if stem.endswith(f"_{split}"):
            exp_id = stem[: -(len(split) + 1)]
        else:
            exp_id = stem

        meta_path = art / f"{exp_id}.meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        hist_path = art / f"{exp_id}_history.csv"
        ckpt_path = art / f"{exp_id}.pt"

        rows.append(
            {
                "model_family": _model_family(exp_id),
                "variant": _variant(exp_id),
                "exp_id": exp_id,
                "split": split,
                "k": payload.get("k"),
                "negatives": payload.get("negatives"),
                "hr": payload.get("hr"),
                "ndcg": payload.get("ndcg"),
                "n": payload.get("n"),
                "best_epoch": meta.get("best_epoch", ""),
                "best_val_ndcg": meta.get("best_val_ndcg", ""),
                "meta_json": str(meta_path.as_posix()) if meta_path.exists() else "",
                "history_csv": str(hist_path.as_posix()) if hist_path.exists() else "",
                "ckpt": str(ckpt_path.as_posix()) if ckpt_path.exists() else "",
            }
        )

    rows.sort(key=lambda r: (str(r["model_family"]), str(r["variant"]), str(r["split"])))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_family",
        "variant",
        "exp_id",
        "split",
        "k",
        "negatives",
        "hr",
        "ndcg",
        "n",
        "best_epoch",
        "best_val_ndcg",
        "ckpt",
        "history_csv",
        "meta_json",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

