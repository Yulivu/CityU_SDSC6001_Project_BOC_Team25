from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccrec.data import build_full_user_features, city_to_business_indices, load_prepared_data, make_train_triples
from ccrec.full_model import FullModel
from ccrec.metrics import aggregate_ranks, best_rank_among_positives
from ccrec.paths import artifacts_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--negatives-per-positive", type=int, default=1)
    p.add_argument("--max-hist-per-aspect", type=int, default=50)
    p.add_argument("--num-aspects", type=int, default=4)
    p.add_argument("--uniform-w", action="store_true")
    p.add_argument("--no-distance", action="store_true")
    p.add_argument("--business-features", choices=["full", "bert", "geo"], default="full")
    p.add_argument("--no-popularity", action="store_true")
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
    p.add_argument("--out", type=str, default=str(artifacts_dir() / "full_model.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    feat_dtype = torch.float16 if device.type == "cuda" else torch.float32

    prepared = load_prepared_data()
    business = prepared.business
    business_by_city = city_to_business_indices(business)
    pos_by_user: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for (u, target_city), pos_bids in prepared.user_city_split.target_city_positives.items():
        pos_by_user[str(u)].append((str(target_city), list(map(str, pos_bids))))

    aspects = ["food", "service", "atmosphere", "price"] if args.num_aspects == 4 else ["all"]
    user_feat = build_full_user_features(prepared, aspects=aspects, max_hist_per_aspect=args.max_hist_per_aspect)

    if args.business_features == "full":
        business_x = business.x
    elif args.business_features == "bert":
        business_x = business.bert
    else:
        business_x = business.x[:, 768:]

    if args.no_popularity and args.business_features in {"full", "geo"}:
        business_x = business_x[:, :-1]

    x = torch.from_numpy(business_x).to(device=device, dtype=feat_dtype)
    bert = torch.from_numpy(business.bert).to(device=device, dtype=feat_dtype)

    triples = make_train_triples(prepared, rng=rng, negatives_per_positive=args.negatives_per_positive)
    if not triples:
        raise RuntimeError("No training triples were generated.")

    model = FullModel(dim=128, num_aspects=args.num_aspects, business_in_dim=int(x.shape[1])).to(device)
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
                        "num_aspects",
                        "business_features",
                        "no_distance",
                        "uniform_w",
                        "no_popularity",
                    ]
                )

    def evaluate_val() -> tuple[float, float, int]:
        users = [u for u in prepared.val_users if u in user_feat.user_to_idx and u in pos_by_user]
        if args.eval_max_users and args.eval_max_users > 0 and len(users) > args.eval_max_users:
            idx = rng.choice(len(users), size=args.eval_max_users, replace=False)
            users = [users[i] for i in idx.tolist()]

        ranks: list[int | None] = []
        model.eval()
        for user_id in tqdm(users, desc="val", leave=False):
            ui = user_feat.user_to_idx.get(user_id)
            if ui is None:
                continue

            hist_lists = [user_feat.hist_business_idx_by_aspect[ui]]
            if not any(hist_lists[0]):
                continue

            if args.uniform_w:
                w = torch.full((1, args.num_aspects), 1.0 / args.num_aspects, device=device, dtype=feat_dtype)
            else:
                w = torch.from_numpy(user_feat.w[ui : ui + 1]).to(device=device, dtype=feat_dtype)

            src_lat = float(user_feat.src_centroid_lat[ui])
            src_lon = float(user_feat.src_centroid_lon[ui])

            flat_hist = [bi for per_aspect in hist_lists[0] for bi in per_aspect]
            unique_hist = sorted(set(flat_hist))
            if not unique_hist:
                continue
            global_to_local = {g: i for i, g in enumerate(unique_hist)}
            local_hist_lists = [[[global_to_local[g] for g in per_aspect] for per_aspect in hist_lists[0]]]

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    hist_bert = bert.index_select(0, torch.tensor(unique_hist, device=device, dtype=torch.long))
                    hist_emb = model.encode_business_hist(hist_bert)
                    user_taste = model.user_taste(hist_emb, local_hist_lists)

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
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                        b_vec = model.encode_business(x.index_select(0, cand_idx))

                tgt_lat = torch.from_numpy(business.latitude[cand_idx.detach().cpu().numpy()]).to(
                    device=device, dtype=torch.float32
                )
                tgt_lon = torch.from_numpy(business.longitude[cand_idx.detach().cpu().numpy()]).to(
                    device=device, dtype=torch.float32
                )
                if args.no_distance:
                    dist = torch.zeros((len(cand_bids),), device=device, dtype=feat_dtype)
                else:
                    dist = torch.log1p(torch.sqrt((tgt_lat - src_lat) ** 2 + (tgt_lon - src_lon) ** 2) + 1e-12).to(
                        dtype=feat_dtype
                    )

                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                        scores = (
                            model.score(
                                user_taste.repeat(len(cand_bids), 1, 1),
                                w.repeat(len(cand_bids), 1),
                                b_vec,
                                dist,
                            )
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                        )

                order = np.argsort(-scores)
                ordered_ids = cand_idx.detach().cpu().numpy()[order].tolist()
                pos_idx_set = {business.business_id_to_idx[b] for b in pos_set}
                ranks.append(best_rank_among_positives(ordered_ids, pos_idx_set))

        metrics = aggregate_ranks(ranks, k=args.eval_k)
        n = len([r for r in ranks if r is not None])
        model.train()
        return metrics.hr_at_k, metrics.ndcg_at_k, n

    def save_ckpt(epoch: int, elapsed_sec: float, train_triples: int, best_val_ndcg: float | None) -> None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dim": model.dim,
                "num_aspects": model.num_aspects,
                "business_in_dim": int(x.shape[1]),
                "business_features": args.business_features,
                "uniform_w": bool(args.uniform_w),
                "no_distance": bool(args.no_distance),
                "no_popularity": bool(args.no_popularity),
                "seed": args.seed,
                "train_triples": train_triples,
                "epoch": epoch,
                "elapsed_sec": elapsed_sec,
                "best_val_ndcg": best_val_ndcg,
            },
            out_path,
        )

    model.train()
    start = time.time()
    best_val_ndcg: float | None = None
    best_epoch = 0
    bad_epochs = 0
    saved = False
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(triples)
        total_loss = 0.0
        steps = 0

        for i in tqdm(range(0, len(triples), args.batch_size), desc=f"epoch {epoch}"):
            batch = triples[i : i + args.batch_size]
            users = [u for (u, _tc, _p, _n) in batch]
            pos_bids = [p for (_u, _tc, p, _n) in batch]
            neg_bids = [n for (_u, _tc, _p, n) in batch]

            u_idx = [user_feat.user_to_idx.get(u) for u in users]
            keep = [j for j, ui in enumerate(u_idx) if ui is not None]
            if not keep:
                continue

            u_idx_t = torch.tensor([u_idx[j] for j in keep], device=device, dtype=torch.long)
            pos_idx_t = torch.tensor([business.business_id_to_idx[pos_bids[j]] for j in keep], device=device, dtype=torch.long)
            neg_idx_t = torch.tensor([business.business_id_to_idx[neg_bids[j]] for j in keep], device=device, dtype=torch.long)

            hist_lists = [user_feat.hist_business_idx_by_aspect[int(u_idx[j])] for j in keep]
            if args.uniform_w:
                w = torch.full((len(keep), args.num_aspects), 1.0 / args.num_aspects, device=device, dtype=feat_dtype)
            else:
                w = torch.from_numpy(user_feat.w[u_idx_t.detach().cpu().numpy()]).to(device=device, dtype=feat_dtype)

            src_lat = torch.from_numpy(user_feat.src_centroid_lat[u_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            src_lon = torch.from_numpy(user_feat.src_centroid_lon[u_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)

            tgt_lat_pos = torch.from_numpy(business.latitude[pos_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            tgt_lon_pos = torch.from_numpy(business.longitude[pos_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            tgt_lat_neg = torch.from_numpy(business.latitude[neg_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)
            tgt_lon_neg = torch.from_numpy(business.longitude[neg_idx_t.detach().cpu().numpy()]).to(device=device, dtype=torch.float32)

            if args.no_distance:
                dist_pos = torch.zeros((len(keep),), device=device, dtype=torch.float32)
                dist_neg = torch.zeros((len(keep),), device=device, dtype=torch.float32)
            else:
                dist_pos = torch.log1p(torch.sqrt((src_lat - tgt_lat_pos) ** 2 + (src_lon - tgt_lon_pos) ** 2) + 1e-12)
                dist_neg = torch.log1p(torch.sqrt((src_lat - tgt_lat_neg) ** 2 + (src_lon - tgt_lon_neg) ** 2) + 1e-12)

            flat_hist = []
            for per_user in hist_lists:
                for per_aspect in per_user:
                    flat_hist.extend(per_aspect)
            unique_hist = sorted(set(flat_hist))
            if not unique_hist:
                continue
            global_to_local = {g: i for i, g in enumerate(unique_hist)}
            local_hist_lists = [
                [[global_to_local[g] for g in per_aspect] for per_aspect in per_user] for per_user in hist_lists
            ]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                hist_bert = bert.index_select(0, torch.tensor(unique_hist, device=device, dtype=torch.long))
                hist_emb = model.encode_business_hist(hist_bert)
                user_taste = model.user_taste(hist_emb, local_hist_lists)

                b_pos = model.encode_business(x.index_select(0, pos_idx_t))
                b_neg = model.encode_business(x.index_select(0, neg_idx_t))

                s_pos = model.score(user_taste, w, b_pos, dist_pos.to(dtype=feat_dtype))
                s_neg = model.score(user_taste, w, b_neg, dist_neg.to(dtype=feat_dtype))
                loss = bpr_loss(s_pos, s_neg)

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

        do_eval = (args.early_stop or log_csv_path is not None) and args.eval_every > 0 and (epoch % args.eval_every == 0)
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
                    save_ckpt(
                        epoch=epoch, elapsed_sec=time.time() - start, train_triples=len(triples), best_val_ndcg=best_val_ndcg
                    )
                    saved = True
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(args.patience):
                        print(f"early_stop: best_epoch={best_epoch} best_val_ndcg={best_val_ndcg:.6f}")
                        if log_csv_path is not None:
                            elapsed_now = time.time() - start
                            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                                wtr = csv.writer(f)
                                wtr.writerow(
                                    [
                                        epoch,
                                        train_loss,
                                        val_hr,
                                        val_ndcg,
                                        val_n,
                                        best_val_ndcg,
                                        elapsed_now,
                                        args.lr,
                                        args.num_aspects,
                                        args.business_features,
                                        bool(args.no_distance),
                                        bool(args.uniform_w),
                                        bool(args.no_popularity),
                                    ]
                                )
                        break

        if log_csv_path is not None:
            elapsed_now = time.time() - start
            with log_csv_path.open("a", newline="", encoding="utf-8") as f:
                wtr = csv.writer(f)
                wtr.writerow(
                    [
                        epoch,
                        train_loss,
                        val_hr,
                        val_ndcg,
                        val_n,
                        best_val_ndcg,
                        elapsed_now,
                        args.lr,
                        args.num_aspects,
                        args.business_features,
                        bool(args.no_distance),
                        bool(args.uniform_w),
                        bool(args.no_popularity),
                    ]
                )

    elapsed = time.time() - start
    if not saved:
        save_ckpt(epoch=epoch, elapsed_sec=elapsed, train_triples=len(triples), best_val_ndcg=best_val_ndcg)
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
        )
    )
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
