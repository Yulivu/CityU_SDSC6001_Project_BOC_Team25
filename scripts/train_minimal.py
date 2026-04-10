from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import load_prepared_data, make_train_triples
from ccrec.models import MinimalRecModel
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--negatives-per-positive", type=int, default=1)
    p.add_argument("--max-hist", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=str(artifacts_dir() / "minimal_model.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    data = load_prepared_data()
    business = data.business

    x = torch.from_numpy(business.x).to(device=device, dtype=torch.float16)

    triples = make_train_triples(data, rng=rng, negatives_per_positive=args.negatives_per_positive)
    if not triples:
        raise RuntimeError("No training triples were generated.")

    hist_map = {}
    for u, hist_bids in data.user_city_split.source_history_businesses.items():
        idxs = [business.business_id_to_idx.get(b) for b in hist_bids]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            continue
        if len(idxs) > args.max_hist:
            idxs = rng.choice(idxs, size=args.max_hist, replace=False).tolist()
        hist_map[u] = idxs

    model = MinimalRecModel(business_in_dim=x.shape[1], dim=128, use_user_embedding=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()

    model.train()
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(triples)
        total_loss = 0.0
        steps = 0

        for i in tqdm(range(0, len(triples), args.batch_size), desc=f"epoch {epoch}"):
            batch = triples[i : i + args.batch_size]
            users = [u for (u, _tc, _p, _n) in batch]
            pos_bids = [p for (_u, _tc, p, _n) in batch]
            neg_bids = [n for (_u, _tc, _p, n) in batch]

            pos_idx = torch.tensor([business.business_id_to_idx[p] for p in pos_bids], device=device, dtype=torch.long)
            neg_idx = torch.tensor([business.business_id_to_idx[n] for n in neg_bids], device=device, dtype=torch.long)

            hist_lists = [hist_map[u] for u in users]
            flat_hist = torch.tensor([j for sub in hist_lists for j in sub], device=device, dtype=torch.long)
            lengths = torch.tensor([len(s) for s in hist_lists], device=device, dtype=torch.long)
            offsets = torch.zeros(len(hist_lists) + 1, device=device, dtype=torch.long)
            offsets[1:] = torch.cumsum(lengths, dim=0)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                hist_vecs = model.encode_business(x.index_select(0, flat_hist))
                hist_mean = torch.zeros((len(hist_lists), hist_vecs.shape[1]), device=device, dtype=hist_vecs.dtype)
                for r in range(len(hist_lists)):
                    a = offsets[r].item()
                    b = offsets[r + 1].item()
                    hist_mean[r] = hist_vecs[a:b].mean(dim=0)

                user_vec = model.user_vector(hist_mean)
                pos_vec = model.encode_business(x.index_select(0, pos_idx))
                neg_vec = model.encode_business(x.index_select(0, neg_idx))
                s_pos = model.score(user_vec, pos_vec)
                s_neg = model.score(user_vec, neg_vec)
                loss = bpr_loss(s_pos, s_neg)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        avg_loss = total_loss / max(1, steps)
        print(f"epoch={epoch} avg_loss={avg_loss:.6f}")

    elapsed = time.time() - start
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "business_in_dim": int(x.shape[1]),
            "dim": 128,
            "use_user_embedding": False,
            "seed": args.seed,
            "train_triples": len(triples),
            "elapsed_sec": elapsed,
        },
        out_path,
    )
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"out": str(out_path), "elapsed_sec": elapsed, "train_triples": len(triples)}, indent=2))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
