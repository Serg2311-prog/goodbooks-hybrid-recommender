"""Загрузка, проверка, очистка и разбиение Goodbooks-10k.

Данные читаются напрямую из ``archive.zip`` (или из папки с CSV) без распаковки и копирования.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

RAW_FILES = ("ratings.csv", "books.csv", "tags.csv", "book_tags.csv")


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        return pd.read_csv(path / name)
    with zipfile.ZipFile(path) as zf:
        members = {Path(m).name: m for m in zf.namelist()}
        if name not in members:
            raise FileNotFoundError(f"{name} не найден в {path}")
        with zf.open(members[name]) as f:
            return pd.read_csv(f)


def load_raw(cfg: Config) -> dict[str, pd.DataFrame]:
    if not Path(cfg.data_path).exists():
        raise FileNotFoundError(
            f"Данные не найдены: {cfg.data_path}. Укажите путь через переменную GOODBOOKS_DATA."
        )
    return {name.removesuffix(".csv"): _read_csv(cfg.data_path, name) for name in RAW_FILES}


def validate_raw(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Проверки схем и ссылочной целостности. Падает с AssertionError при нарушении."""
    ratings, books, tags, book_tags = raw["ratings"], raw["books"], raw["tags"], raw["book_tags"]

    assert {"user_id", "book_id", "rating"} <= set(ratings.columns)
    assert {"id", "book_id", "title", "authors"} <= set(books.columns)
    assert {"tag_id", "tag_name"} <= set(tags.columns)
    assert {"goodreads_book_id", "tag_id", "count"} <= set(book_tags.columns)

    assert ratings[["user_id", "book_id", "rating"]].notna().all().all()
    assert ratings["rating"].between(1, 5).all(), "rating вне диапазона [1, 5]"
    assert books["id"].is_unique and books["book_id"].is_unique
    assert np.array_equal(np.sort(books["id"].to_numpy()), np.arange(1, len(books) + 1)), \
        "books.id должен быть непрерывным индексом 1..N"
    assert ratings["book_id"].isin(books["id"]).all(), "ratings.book_id ссылается на books.id"
    assert book_tags["goodreads_book_id"].isin(books["book_id"]).all(), \
        "book_tags.goodreads_book_id ссылается на books.book_id (Goodreads ID)"
    assert book_tags["tag_id"].isin(tags["tag_id"]).all()

    rows = []
    for name, df in raw.items():
        rows.append({
            "dataset": name,
            "rows": len(df),
            "columns": df.shape[1],
            "missing_cells": int(df.isna().sum().sum()),
            "full_duplicates": int(df.duplicated().sum()),
        })
    report = pd.DataFrame(rows)
    report.loc[report["dataset"] == "ratings", "duplicate_user_book_pairs"] = int(
        ratings.duplicated(["user_id", "book_id"]).sum()
    )
    return report


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Повторные пары (user_id, book_id) усредняются, как в HW_1_FINAL."""
    interactions = ratings.groupby(["user_id", "book_id"], as_index=False)["rating"].mean()
    assert not interactions.duplicated(["user_id", "book_id"]).any()
    return interactions


def per_user_holdout(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user random holdout из HW_1_FINAL.

    Для пользователя с n >= 2 взаимодействиями в holdout уходит min(max(1, floor(fraction*n)), n-1),
    пользователь с одним взаимодействием целиком остаётся в keep.
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_rand"] = rng.random(len(work))
    work = work.sort_values(["user_id", "_rand"], kind="mergesort")
    n_by_user = work.groupby("user_id")["book_id"].transform("size").to_numpy()
    n_hold = np.minimum(np.maximum(1, np.floor(fraction * n_by_user).astype(np.int64)), n_by_user - 1)
    rank = work.groupby("user_id").cumcount().to_numpy()
    is_hold = rank >= (n_by_user - n_hold)
    work = work.drop(columns="_rand")
    return work.loc[~is_hold].copy(), work.loc[is_hold].copy()


@dataclass
class Splits:
    train: pd.DataFrame        # train для обучения при подборе на validation
    val: pd.DataFrame          # validation: веса гибрида, гиперпараметры, early stopping
    train_full: pd.DataFrame   # train + val: финальное переобучение (совпадает с train_df HW_1_FINAL)
    test: pd.DataFrame         # только финальная оценка


def make_splits(interactions: pd.DataFrame, cfg: Config) -> Splits:
    train_full, test = per_user_holdout(interactions, cfg.holdout_fraction, cfg.random_state)
    train, val = per_user_holdout(train_full, cfg.holdout_fraction, cfg.val_random_state)
    splits = Splits(train=train, val=val, train_full=train_full, test=test)
    check_no_leakage(splits, n_interactions=len(interactions))
    return splits


def _pairs(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[["user_id", "book_id"]])


def check_no_leakage(s: Splits, n_interactions: int | None = None) -> None:
    tr, va, trf, te = _pairs(s.train), _pairs(s.val), _pairs(s.train_full), _pairs(s.test)
    assert len(tr.intersection(va)) == 0, "train ∩ val"
    assert len(tr.intersection(te)) == 0, "train ∩ test"
    assert len(va.intersection(te)) == 0, "val ∩ test"
    assert len(trf.intersection(te)) == 0, "train_full ∩ test"
    assert len(trf) == len(tr) + len(va), "train_full = train + val"
    assert len(trf.intersection(tr.union(va))) == len(trf), "train_full = train + val"
    assert set(s.train.index).isdisjoint(s.val.index) and set(s.train_full.index).isdisjoint(s.test.index)
    if n_interactions is not None:
        assert len(s.train_full) + len(s.test) == n_interactions
    assert set(s.test["user_id"]) <= set(s.train_full["user_id"]), "у каждого test-пользователя есть train"
    assert set(s.val["user_id"]) <= set(s.train["user_id"]), "у каждого val-пользователя есть train"


def split_summary(s: Splits) -> pd.DataFrame:
    rows = []
    for name in ("train", "val", "train_full", "test"):
        df = getattr(s, name)
        rows.append({
            "part": name,
            "interactions": len(df),
            "users": df["user_id"].nunique(),
            "books": df["book_id"].nunique(),
            "share_relevant(>=4)": float((df["rating"] >= 4).mean()),
            "median_per_user": float(df.groupby("user_id").size().median()),
        })
    return pd.DataFrame(rows)


@dataclass
class Index:
    """Отображение внешних ID в непрерывные индексы для sparse-матриц."""
    user_ids: np.ndarray
    n_items: int

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    def users_to_idx(self, user_ids) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        pos = np.searchsorted(self.user_ids, user_ids)
        pos = np.clip(pos, 0, len(self.user_ids) - 1)
        ok = self.user_ids[pos] == user_ids
        return np.where(ok, pos, -1)

    @staticmethod
    def items_to_idx(book_ids) -> np.ndarray:
        return np.asarray(book_ids, dtype=np.int64) - 1

    @staticmethod
    def idx_to_items(item_idx) -> np.ndarray:
        return np.asarray(item_idx, dtype=np.int64) + 1


def build_index(interactions: pd.DataFrame, books: pd.DataFrame) -> Index:
    return Index(user_ids=np.sort(interactions["user_id"].unique()), n_items=len(books))


def relevant_sets(df: pd.DataFrame, index: Index, threshold: float) -> dict[int, np.ndarray]:
    """user_idx -> массив item_idx с rating >= threshold."""
    rel = df.loc[df["rating"] >= threshold, ["user_id", "book_id"]]
    u = index.users_to_idx(rel["user_id"].to_numpy())
    i = index.items_to_idx(rel["book_id"].to_numpy())
    order = np.argsort(u, kind="mergesort")
    u, i = u[order], i[order]
    bounds = np.flatnonzero(np.diff(u)) + 1
    return {int(uu[0]): ii for uu, ii in zip(np.split(u, bounds), np.split(i, bounds)) if len(uu)}
