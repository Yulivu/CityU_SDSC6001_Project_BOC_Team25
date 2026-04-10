from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import city_to_business_indices, load_prepared_data, make_train_triples
from ccrec.metrics import aggregate_ranks, best_rank_among_positives
from ccrec.paths import artifacts_dir


class MF(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.user = nn.Embedding(num_users, dim)
        self.item = nn.Embedding(num_items, dim)
        nn.init.normal_(self.user.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item.weight, mean=0.0, std=0.02)

    def score(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return (self.user(u) * self.item(i)).sum(dim=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--negatives-per-positive", type=int, default=4)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-k", type=int, default=10)
    p.add_argument("--eval-negatives", type=int, default=99)
    p.add_argument("--eval-max-users", type=int, default=2000)
    p.add_argument("--log-csv", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default=str(artifacts_dir() / "mf.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    prepared = load_prepared_data()
    business = prepared.business
    business_by_city = city_to_business_indices(business)

    pos_by_user: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (u, target_city), pos_bids in prepared.user_city_split.target_city_positives.items():
        pos_by_user[str(u)].append((str(target_city), list(map(str, pos_bids))))

    triples = make_train_triples(prepared, rng=rng, negatives_per_positive=args.negatives_per_positive)
    if not triples:
        raise RuntimeError("No training triples were generated.")

    user_ids = sorted({str(u) for (u, _tc, _p, _n) in triples})
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    model = MF(num_users=len(user_ids), num_items=len(business.business_ids), dim=int(args.dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def bpr_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        return -torch.log(torch.sigmoid(pos - neg) + 1e-12).mean()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_csv_path = Path(args.log_csv) if args.log_csv else None
    if log_csv_path is not None:
        log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not log_csv_path.exists():
            with log_csv_path.open("w", newline="", encoding="utf-8") as f:
                wtr = csv.writer(f)
                wtr.writerow(
                    [
                        "epoch",
                        "train_loss",
                        "val_hr",
                        "val_ndcg",
                        "val_n",
                        "best_val_ndcg",
                        "elapsed_sec",
                        "lr",
                        "dim",
                    ]
                )

    def evaluate_val() -> tuple[float, float, int]:
        users = [u for u in prepared.val_users if u in pos_by_user]
        if args.eval_max_users and args.eval_max_users > 0 and len(users) > args.eval_max_users:
            idx = rng.choice(len(users), size=args.eval_max_users, replace=False)
            users = [users[i] for i in idx.tolist()]

        model.eval()
        ranks: list[int | None] = []
        with torch.no_grad():
            for user_id in tqdm(users, desc="val", leave=False):
                ui = user_to_idx.get(user_id)
                if ui is None:
                    hist_bids = prepared.user_city_split.source_history_businesses.get(user_id)
                    if not hist_bids:
                        continue
                    hist_idx = [business.business_id_to_idx.get(str(b)) for b in hist_bids]
                    hist_idx = [i for i in hist_idx if i is not None]
                    if not hist_idx:
                        continue
                    hist_t = torch.tensor(hist_idx, device=device, dtype=torch.long)
                    u_vec = model.item(hist_t).mean(dim=0, keepdim=True)
                else:
                    u_vec = model.user(torch.tensor([int(ui)], device=device, dtype=torch.long))

                for target_city, pos_bids in pos_by_user.get(user_id, []):
                    if target_city not in business_by_city:
                        continue

                    pos_set = set(pos_bids)
                    city_idxs = business_by_city[target_city]
                    if len(city_idxs) <= len(pos_set):
                        continue

                    negs = set()
                    for _ in range(args.eval_negatives * 10):
                        bid = business.business_ids[int(rng.choice(city_idxs))]
                        if bid not in pos_set:
                            negs.add(bid)
                        if len(negs) >= args.eval_negatives:
                            break
                    if len(negs) < args.eval_negatives:
                        continue

                    cand_bids = list(pos_set) + list(negs)
                    cand_idx = torch.tensor(
                        [business.business_id_to_idx[b] for b in cand_bids], device=device, dtype=torch.long
                    )
                    cand_vec = model.item(cand_idx)
                    scores = (cand_vec @ u_vec.squeeze(0)).detach().float().cpu().numpy()
                    order = np.argsort(-scores)
                    ordered_ids = cand_idx.detach().cpu().numpy()[order].tolist()
                    pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
                    ranks.append(best_rank_among_positives(ordered_ids, pos_idx_set))

        metrics = aggregate_ranks(ranks, k=args.eval_k)
        n = len([r for r in ranks if r is not None])
        model.train()
        return metrics.hr_at_k, metrics.ndcg_at_k, n

    def save_ckpt(epoch: int, elapsed_sec: float, best_val_ndcg: float | None) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dim": int(args.dim),
                "user_ids": user_ids,
                "seed": args.seed,
                "epoch": epoch,
                "elapsed_sec": elapsed_sec,
                "best_val_ndcg": best_val_ndcg,
            },
            out_path,
        )

    start = time.time()
    best_val_ndcg: float | None = None
    best_epoch = 0
    bad_epochs = 0
    saved = False

    for epoch in range(1, int(args.epochs) + 1):
        rng.shuffle(triples)
        total_loss = 0.0
        steps = 0

        for i in tqdm(range(0, len(triples), int(args.batch_size)), desc=f"epoch {epoch}"):
            batch = triples[i : i + int(args.batch_size)]
            users = [u for (u, _tc, _p, _n) in batch]
            pos_bids = [p for (_u, _tc, p, _n) in batch]
            neg_bids = [n for (_u, _tc, _p, n) in batch]

            u_idx = [user_to_idx.get(str(u)) for u in users]
            keep = [j for j, ui in enumerate(u_idx) if ui is not None]
            if not keep:
                continue

            u_t = torch.tensor([u_idx[j] for j in keep], device=device, dtype=torch.long)
            p_t = torch.tensor(
                [business.business_id_to_idx[pos_bids[j]] for j in keep], device=device, dtype=torch.long
            )
            n_t = torch.tensor(
                [business.business_id_to_idx[neg_bids[j]] for j in keep], device=device, dtype=torch.long
            )

            opt.zero_grad(set_to_none=True)
            pos = model.score(u_t, p_t)
            neg = model.score(u_t, n_t)
            loss = bpr_loss(pos, neg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            steps += 1

        train_loss = total_loss / max(1, steps)
        print(f"epoch={epoch} avg_loss={train_loss:.6f}")

        val_hr: float | None = None
        val_ndcg: float | None = None
        val_n: int | None = None

        do_eval = (args.early_stop or log_csv_path is not None) and int(args.eval_every) > 0 and (epoch % int(args.eval_every) == 0)
        if do_eval:
            hr, ndcg, n = evaluate_val()
            val_hr, val_ndcg, val_n = float(hr), float(ndcg), int(n)
            print(f"val hr@{args.eval_k}={val_hr:.4f} ndcg@{args.eval_k}={val_ndcg:.4f} n={val_n}")

            if args.early_stop:
                improved = best_val_ndcg is None or (val_ndcg > best_val_ndcg + float(args.min_delta))
                if improved:
                    best_val_ndcg = float(val_ndcg)
                    best_epoch = int(epoch)
                    bad_epochs = 0
                    save_ckpt(epoch=epoch, elapsed_sec=time.time() - start, best_val_ndcg=best_val_ndcg)
                    saved = True
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(args.patience):
                        print(f"early_stop: best_epoch={best_epoch} best_val_ndcg={best_val_ndcg:.6f}")
                        if log_csv_path is not None:
                            elapsed_now = time.time() - start
                            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                                wtr = csv.writer(f)
                                wtr.writerow([epoch, train_loss, val_hr, val_ndcg, val_n, best_val_ndcg, elapsed_now, args.lr, args.dim])
                        break

        if log_csv_path is not None:
            elapsed_now = time.time() - start
            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                wtr = csv.writer(f)
                wtr.writerow([epoch, train_loss, val_hr, val_ndcg, val_n, best_val_ndcg, elapsed_now, args.lr, args.dim])

    elapsed = time.time() - start
    if not saved:
        save_ckpt(epoch=epoch, elapsed_sec=elapsed, best_val_ndcg=best_val_ndcg)
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "out": str(out_path),
                "elapsed_sec": elapsed,
                "train_triples": len(triples),
                "best_epoch": best_epoch,
                "best_val_ndcg": best_val_ndcg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
