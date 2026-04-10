from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.paths import artifacts_dir


@dataclass(frozen=True)
class Experiment:
    name: str
    train_args: List[str]
    eval_args: List[str]


def run(cmd: List[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--negatives-per-positive", type=int, default=1)
    p.add_argument("--negatives", type=int, default=99)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-hist-per-aspect", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-users", type=int, default=0)
    p.add_argument("--out", type=str, default=str(artifacts_dir() / "table1.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    art = artifacts_dir()
    py = sys.executable

    baseline_val = art / "baseline_metrics_val.json"
    baseline_test = art / "baseline_metrics_test.json"
    run([py, "scripts/run_baselines.py", "--split", "val", "--k", str(args.k), "--negatives", str(args.negatives), "--seed", str(args.seed), "--out", str(baseline_val)] + (["--max-users", str(args.max_users)] if args.max_users else []))
    run([py, "scripts/run_baselines.py", "--split", "test", "--k", str(args.k), "--negatives", str(args.negatives), "--seed", str(args.seed), "--out", str(baseline_test)] + (["--max-users", str(args.max_users)] if args.max_users else []))

    exps = []

    def add_full(name: str, extra_train: List[str], extra_eval: List[str]) -> None:
        ckpt = art / f"{name}.pt"
        val_out = art / f"{name}_val.json"
        test_out = art / f"{name}_test.json"
        exps.append(
            (name, ckpt, val_out, test_out, extra_train, extra_eval)
        )

    add_full("full", [], [])
    add_full("full_no_geo", ["--no-distance"], ["--no-distance"])
    add_full("full_uniform_w", ["--uniform-w"], ["--uniform-w"])
    add_full("full_no_popularity", ["--no-popularity"], ["--no-popularity"])
    add_full("full_single_aspect", ["--num-aspects", "1"], [])
    add_full("full_business_bert", ["--business-features", "bert"], [])
    add_full("full_business_geo", ["--business-features", "geo"], [])

    results: Dict[str, Dict] = {
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "negatives_per_positive": args.negatives_per_positive,
            "negatives": args.negatives,
            "k": args.k,
            "max_hist_per_aspect": args.max_hist_per_aspect,
            "seed": args.seed,
        },
        "baselines": {
            "val": read_json(baseline_val),
            "test": read_json(baseline_test),
        },
        "models": {},
    }

    for name, ckpt, val_out, test_out, extra_train, extra_eval in exps:
        train_cmd = [
            py,
            "scripts/train_full.py",
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--negatives-per-positive",
            str(args.negatives_per_positive),
            "--max-hist-per-aspect",
            str(args.max_hist_per_aspect),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--out",
            str(ckpt),
        ] + extra_train
        run(train_cmd)

        eval_common = [
            "--k",
            str(args.k),
            "--negatives",
            str(args.negatives),
            "--max-hist-per-aspect",
            str(args.max_hist_per_aspect),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--ckpt",
            str(ckpt),
        ] + extra_eval
        if args.max_users:
            eval_common += ["--max-users", str(args.max_users)]

        run([py, "scripts/eval_full.py", "--split", "val", "--out", str(val_out)] + eval_common)
        run([py, "scripts/eval_full.py", "--split", "test", "--out", str(test_out)] + eval_common)

        results["models"][name] = {"val": read_json(val_out), "test": read_json(test_out)}

    out_path = Path(args.out)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
