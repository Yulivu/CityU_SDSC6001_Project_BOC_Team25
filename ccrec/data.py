from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .paths import outputs_dir


@dataclass(frozen=True)
class BusinessFeatures:
    business_ids: List[str]
    business_id_to_idx: Dict[str, int]
    city: np.ndarray
    review_count: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    x: np.ndarray
    bert: np.ndarray


@dataclass(frozen=True)
class UserCitySplit:
    source_city: Dict[str, str]
    source_history_businesses: Dict[str, List[str]]
    target_city_positives: Dict[Tuple[str, str], List[str]]


@dataclass(frozen=True)
class FullUserFeatures:
    users: List[str]
    user_to_idx: Dict[str, int]
    hist_business_idx_by_aspect: List[List[List[int]]]
    w: np.ndarray
    src_centroid_lat: np.ndarray
    src_centroid_lon: np.ndarray


@dataclass(frozen=True)
class DGLGraphData:
    user_ids: List[str]
    business_ids: List[str]
    user_to_idx: Dict[str, int]
    business_to_idx: Dict[str, int]
    ub_src: np.ndarray
    ub_dst: np.ndarray
    ub_weight: np.ndarray
    ub_aspect: np.ndarray
    uu_src: np.ndarray
    uu_dst: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    business: BusinessFeatures
    user_city_split: UserCitySplit
    train_users: List[str]
    val_users: List[str]
    test_users: List[str]


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_business_features(out_dir: Path | None = None) -> BusinessFeatures:
    out = outputs_dir() if out_dir is None else out_dir

    df_business = _read_parquet(
        out / "business_filtered.parquet",
        columns=["business_id", "city", "review_count", "latitude", "longitude"],
    )
    df_geo = _read_parquet(
        out / "business_geo_features.parquet",
        columns=["business_id", "checkin_density"]
        + [f"zone_embedding_{i}" for i in range(32)],
    )
    df_bert = _read_parquet(out / "business_bert_embeddings.parquet")

    df = df_business.merge(df_geo, on="business_id", how="left").merge(df_bert, on="business_id", how="left")
    df = df.drop_duplicates(subset=["business_id"]).reset_index(drop=True)

    df["review_count"] = df["review_count"].astype(np.float32).fillna(0.0)

    for col in ["latitude", "longitude"]:
        df[col] = df[col].astype(np.float32)
        city_mean = df.groupby("city")[col].transform(lambda s: s.fillna(s.mean()))
        df[col] = df[col].fillna(city_mean).fillna(0.0)

    if "checkin_density" in df.columns:
        df["checkin_density"] = df["checkin_density"].astype(np.float32).fillna(0.0)

    log_rc = np.log1p(df["review_count"].to_numpy(dtype=np.float32))
    city_max = df.groupby("city")["review_count"].transform("max").to_numpy(dtype=np.float32)
    log_city_max = np.log1p(city_max)
    pop = (log_rc / (log_city_max + 1e-12)).astype(np.float32)
    pop = np.nan_to_num(pop, nan=0.0, posinf=0.0, neginf=0.0)

    business_ids = df["business_id"].astype(str).tolist()
    business_id_to_idx = {bid: i for i, bid in enumerate(business_ids)}

    bert_cols = [f"dim_{i}" for i in range(768)]
    zone_cols = [f"zone_embedding_{i}" for i in range(32)]

    for c in zone_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32).fillna(0.0)
        else:
            df[c] = 0.0
    for c in bert_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.float32).fillna(0.0)
        else:
            df[c] = 0.0

    bert = df[bert_cols].to_numpy(dtype=np.float32, copy=True)
    zone = df[zone_cols].to_numpy(dtype=np.float32, copy=True)
    density = df[["checkin_density"]].to_numpy(dtype=np.float32, copy=True)
    pop_feat = pop.reshape(-1, 1)
    x = np.concatenate([bert, zone, density, pop_feat], axis=1)

    return BusinessFeatures(
        business_ids=business_ids,
        business_id_to_idx=business_id_to_idx,
        city=df["city"].astype(str).to_numpy(),
        review_count=df["review_count"].to_numpy(dtype=np.float32, copy=True),
        latitude=df["latitude"].astype(np.float32).to_numpy(),
        longitude=df["longitude"].astype(np.float32).to_numpy(),
        x=x,
        bert=bert,
    )


def load_user_splits(out_dir: Path | None = None) -> tuple[list[str], list[str], list[str]]:
    out = outputs_dir() if out_dir is None else out_dir
    train_users = _read_parquet(out / "users_train.parquet")["user_id"].astype(str).tolist()
    val_users = _read_parquet(out / "users_val.parquet")["user_id"].astype(str).tolist()
    test_users = _read_parquet(out / "users_test.parquet")["user_id"].astype(str).tolist()
    return train_users, val_users, test_users


def build_user_city_split(out_dir: Path | None = None) -> UserCitySplit:
    out = outputs_dir() if out_dir is None else out_dir
    df = _read_parquet(
        out / "reviews_with_aspects.parquet",
        columns=["user_id", "business_id", "city", "stars", "aspect"],
    )
    df["user_id"] = df["user_id"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    df["city"] = df["city"].astype(str)

    city_counts = df.groupby(["user_id", "city"], sort=False).size().reset_index(name="n")
    city_counts = city_counts.sort_values(["user_id", "n", "city"], ascending=[True, False, True])
    source_city = city_counts.drop_duplicates(subset=["user_id"], keep="first").set_index("user_id")["city"].to_dict()

    source_history = (
        df[df["city"] == df["user_id"].map(source_city)][["user_id", "business_id"]]
        .groupby("user_id", sort=False)["business_id"]
        .apply(list)
        .to_dict()
    )

    target_pos = (
        df[df["city"] != df["user_id"].map(source_city)][["user_id", "city", "business_id"]]
        .groupby(["user_id", "city"], sort=False)["business_id"]
        .apply(list)
        .to_dict()
    )

    return UserCitySplit(
        source_city=source_city,
        source_history_businesses=source_history,
        target_city_positives=target_pos,
    )


def build_full_user_features(
    prepared: PreparedData,
    aspects: list[str] | None = None,
    max_hist_per_aspect: int = 50,
    out_dir: Path | None = None,
) -> FullUserFeatures:
    if aspects is None:
        aspects = ["food", "service", "atmosphere", "price"]
    collapse_aspects = len(aspects) == 1

    out = outputs_dir() if out_dir is None else out_dir
    business = prepared.business

    df = _read_parquet(
        out / "reviews_with_aspects.parquet",
        columns=["user_id", "business_id", "city", "stars", "date", "aspect"],
    )
    df["user_id"] = df["user_id"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    df["city"] = df["city"].astype(str)
    df["aspect"] = df["aspect"].astype(str)

    df = df.sort_values(["user_id", "business_id", "date"])
    df = df.drop_duplicates(subset=["user_id", "business_id"], keep="last")

    src_city = prepared.user_city_split.source_city
    df_src = df[df["city"] == df["user_id"].map(src_city)]

    users = sorted(set(df_src["user_id"].tolist()))
    user_to_idx = {u: i for i, u in enumerate(users)}

    aspect_to_idx = {a: i for i, a in enumerate(aspects)}

    hist_by_user_aspect: list[list[list[int]]] = [[[] for _ in aspects] for _ in users]
    stars_by_user_aspect: list[list[list[float]]] = [[[] for _ in aspects] for _ in users]

    for row in df_src.itertuples(index=False):
        u = row.user_id
        b = row.business_id
        a = row.aspect
        if u not in user_to_idx:
            continue
        bi = business.business_id_to_idx.get(b)
        if bi is None:
            continue
        ui = user_to_idx[u]
        if collapse_aspects:
            ai = 0
        else:
            if a not in aspect_to_idx:
                continue
            ai = aspect_to_idx[a]
        if len(hist_by_user_aspect[ui][ai]) < max_hist_per_aspect:
            hist_by_user_aspect[ui][ai].append(int(bi))
            stars_by_user_aspect[ui][ai].append(float(row.stars))

    var = np.full((len(users), len(aspects)), 999.0, dtype=np.float32)
    for ui in range(len(users)):
        for ai in range(len(aspects)):
            s = stars_by_user_aspect[ui][ai]
            if len(s) >= 2:
                var[ui, ai] = float(np.var(np.asarray(s, dtype=np.float32)))
            elif len(s) == 1:
                var[ui, ai] = 0.0

    x = -var
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    w = e / (e.sum(axis=1, keepdims=True) + 1e-12)

    src_centroid_lat = np.zeros((len(users),), dtype=np.float32)
    src_centroid_lon = np.zeros((len(users),), dtype=np.float32)
    for ui, u in enumerate(users):
        idxs = [bi for per_a in hist_by_user_aspect[ui] for bi in per_a]
        if not idxs:
            continue
        lat = business.latitude[np.asarray(idxs, dtype=np.int64)]
        lon = business.longitude[np.asarray(idxs, dtype=np.int64)]
        src_centroid_lat[ui] = float(np.nanmean(lat)) if np.isfinite(lat).any() else 0.0
        src_centroid_lon[ui] = float(np.nanmean(lon)) if np.isfinite(lon).any() else 0.0

    return FullUserFeatures(
        users=users,
        user_to_idx=user_to_idx,
        hist_business_idx_by_aspect=hist_by_user_aspect,
        w=w.astype(np.float32),
        src_centroid_lat=src_centroid_lat,
        src_centroid_lon=src_centroid_lon,
    )


def build_dgl_graph_data(
    prepared: PreparedData,
    aspects: list[str] | None = None,
    max_hist_per_aspect: int = 50,
    include_social: bool = True,
    collapse_aspects: bool = False,
    out_dir: Path | None = None,
) -> DGLGraphData:
    if aspects is None:
        aspects = ["food", "service", "atmosphere", "price"]

    out = outputs_dir() if out_dir is None else out_dir

    user_ids = []
    seen = set()
    for u in prepared.train_users + prepared.val_users + prepared.test_users:
        u = str(u)
        if u in seen:
            continue
        seen.add(u)
        user_ids.append(u)
    user_to_idx = {u: i for i, u in enumerate(user_ids)}

    business_ids = prepared.business.business_ids
    business_to_idx = prepared.business.business_id_to_idx

    df = _read_parquet(
        out / "reviews_with_aspects.parquet",
        columns=["user_id", "business_id", "city", "stars", "date", "aspect"],
    )
    df["user_id"] = df["user_id"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    df["city"] = df["city"].astype(str)
    df["aspect"] = df["aspect"].astype(str)

    df = df[df["user_id"].isin(user_to_idx)]
    df = df.sort_values(["user_id", "business_id", "date"])
    df = df.drop_duplicates(subset=["user_id", "business_id"], keep="last")

    src_city = prepared.user_city_split.source_city
    df = df[df["city"] == df["user_id"].map(src_city)]

    aspect_to_idx = {a: i for i, a in enumerate(aspects)}
    if not collapse_aspects:
        df = df[df["aspect"].isin(aspect_to_idx)]

    df["b_idx"] = df["business_id"].map(business_to_idx)
    df = df.dropna(subset=["b_idx"])
    df["b_idx"] = df["b_idx"].astype(np.int64)
    df["u_idx"] = df["user_id"].map(user_to_idx).astype(np.int64)
    if collapse_aspects:
        df["a_idx"] = 0
    else:
        df["a_idx"] = df["aspect"].map(aspect_to_idx).astype(np.int64)
    df["w"] = (df["stars"].astype(np.float32) / 5.0).astype(np.float32)

    df = df.sample(frac=1.0, random_state=42)
    df = df.groupby(["u_idx", "a_idx"], sort=False).head(max_hist_per_aspect)

    ub_src = df["u_idx"].to_numpy(dtype=np.int64, copy=True)
    ub_dst = df["b_idx"].to_numpy(dtype=np.int64, copy=True)
    ub_weight = df["w"].to_numpy(dtype=np.float32, copy=True)
    ub_aspect = df["a_idx"].to_numpy(dtype=np.int64, copy=True)

    if include_social:
        df_social = _read_parquet(out / "graph_user_social_edges.parquet", columns=["user_id_1", "user_id_2"])
        df_social["user_id_1"] = df_social["user_id_1"].astype(str)
        df_social["user_id_2"] = df_social["user_id_2"].astype(str)
        df_social = df_social[df_social["user_id_1"].isin(user_to_idx) & df_social["user_id_2"].isin(user_to_idx)]
        uu_src = df_social["user_id_1"].map(user_to_idx).to_numpy(dtype=np.int64, copy=True)
        uu_dst = df_social["user_id_2"].map(user_to_idx).to_numpy(dtype=np.int64, copy=True)
    else:
        uu_src = np.zeros((0,), dtype=np.int64)
        uu_dst = np.zeros((0,), dtype=np.int64)

    return DGLGraphData(
        user_ids=user_ids,
        business_ids=business_ids,
        user_to_idx=user_to_idx,
        business_to_idx=business_to_idx,
        ub_src=ub_src,
        ub_dst=ub_dst,
        ub_weight=ub_weight,
        ub_aspect=ub_aspect,
        uu_src=uu_src,
        uu_dst=uu_dst,
    )


def load_prepared_data(out_dir: Path | None = None) -> PreparedData:
    out = outputs_dir() if out_dir is None else out_dir
    business = load_business_features(out)
    train_users, val_users, test_users = load_user_splits(out)
    user_city_split = build_user_city_split(out)
    return PreparedData(
        business=business,
        user_city_split=user_city_split,
        train_users=train_users,
        val_users=val_users,
        test_users=test_users,
    )


def city_to_business_indices(business: BusinessFeatures) -> Dict[str, np.ndarray]:
    by_city: Dict[str, List[int]] = {}
    for i, city in enumerate(business.city.tolist()):
        by_city.setdefault(city, []).append(i)
    return {c: np.asarray(v, dtype=np.int64) for c, v in by_city.items()}


def make_train_triples(
    prepared: PreparedData,
    rng: np.random.Generator,
    negatives_per_positive: int = 1,
) -> list[tuple[str, str, str, str]]:
    business_by_city = city_to_business_indices(prepared.business)
    city_business_ids: Dict[str, np.ndarray] = {}
    for city, idxs in business_by_city.items():
        city_business_ids[city] = np.asarray([prepared.business.business_ids[i] for i in idxs.tolist()], dtype=object)

    triples: list[tuple[str, str, str, str]] = []
    train_users_set = set(prepared.train_users)

    for (user_id, target_city), pos_businesses in prepared.user_city_split.target_city_positives.items():
        if user_id not in train_users_set:
            continue
        if target_city not in city_business_ids:
            continue

        src_hist = prepared.user_city_split.source_history_businesses.get(user_id)
        if not src_hist:
            continue

        pos_set = set(pos_businesses)
        city_bids = city_business_ids[target_city]

        for pos_bid in pos_businesses:
            for _ in range(negatives_per_positive):
                for _tries in range(100):
                    neg_bid = str(rng.choice(city_bids))
                    if neg_bid not in pos_set:
                        triples.append((user_id, target_city, str(pos_bid), neg_bid))
                        break

    rng.shuffle(triples)
    return triples
