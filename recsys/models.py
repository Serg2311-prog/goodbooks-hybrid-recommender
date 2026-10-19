"""Модели HW_1_FINAL, адаптированные к единому интерфейсу.

Каждый scorer после ``fit(ctx)`` умеет ``score(user_idx) -> (len(user_idx), n_items)``: оценку всего
каталога для батча пользователей. Маскирование прочитанного и нормализация — в ``hybrid``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from .config import Config
from .data import Index
from .features import (Catalog, build_item_features, build_user_features, liked_matrix,
                       ratings_matrix)


@dataclass
class FitContext:
    """Всё, что модели могут знать при обучении: только переданная обучающая часть."""
    train: pd.DataFrame
    index: Index
    catalog: Catalog
    cfg: Config
    R: csr_matrix
    item_feats: pd.DataFrame
    user_feats: pd.DataFrame
    topic_profile: np.ndarray
    extra: dict = field(default_factory=dict)

    @property
    def n_users(self) -> int:
        return self.index.n_users

    @property
    def n_items(self) -> int:
        return self.index.n_items

    def seen_rows(self, users: np.ndarray) -> csr_matrix:
        return self.R[users]


def fit_context(train: pd.DataFrame, index: Index, catalog: Catalog, cfg: Config) -> FitContext:
    R = ratings_matrix(train, index)
    item_feats = build_item_features(R, catalog)
    user_feats, topic_profile = build_user_features(R, item_feats, catalog, cfg)
    # Защита от утечки: статистики построены ровно по переданной части.
    assert int(item_feats["item_n_ratings"].sum()) == len(train)
    assert int(user_feats["user_n_ratings"].sum()) == len(train)
    assert abs(item_feats.attrs["global_mean"] - train["rating"].mean()) < 1e-4
    return FitContext(train=train, index=index, catalog=catalog, cfg=cfg, R=R,
                      item_feats=item_feats, user_feats=user_feats, topic_profile=topic_profile)


class Scorer:
    name = "base"

    def fit(self, ctx: FitContext) -> "Scorer":
        self.ctx = ctx
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PopularityScorer(Scorer):
    """Popularity из HW_1_FINAL.

    kind: ``weighted_rating`` — WR = v/(v+m)*R + m/(v+m)*C (как в HW_1_FINAL);
          ``n_ratings`` — число train-оценок; ``share_pos`` — доля оценок >= 4.
    """
    name = "pop"
    KINDS = {"weighted_rating": "item_weighted_rating", "n_ratings": "item_n_ratings",
             "share_pos": "item_share_pos"}

    def __init__(self, kind: str = "weighted_rating"):
        assert kind in self.KINDS
        self.kind = kind

    def fit(self, ctx):
        super().fit(ctx)
        self.item_score = ctx.item_feats[self.KINDS[self.kind]].to_numpy(np.float32)
        return self

    def score(self, users):
        return np.broadcast_to(self.item_score, (len(users), len(self.item_score))).copy()


class ContentScorer(Scorer):
    """Content-Based HW_1_FINAL: профиль = среднее TF-IDF понравившихся (rating >= 4) книг."""
    name = "cb"

    def __init__(self, profile: str = "base"):
        assert profile in ("base", "enhanced")
        self.profile = profile

    def fit(self, ctx):
        super().fit(ctx)
        self.T = ctx.catalog.tfidf_base if self.profile == "base" else ctx.catalog.tfidf_enhanced
        self.L = liked_matrix(ctx.R, ctx.cfg.relevance_threshold)
        return self

    def user_profiles(self, users):
        return normalize(self.L[users] @ self.T)

    def score(self, users):
        return (self.user_profiles(users) @ self.T.T).toarray().astype(np.float32)


class ItemCFScorer(Scorer):
    """Item-Based CF HW_1_FINAL (cosine по векторам оценок) с хранением только top-k соседей.

    mode:
      * ``avg``          — как в HW_1_FINAL: взвешенное среднее оценок пользователя по соседям,
                           при отсутствии соседей — средняя train-оценка книги;
      * ``sum_centered`` — sum_j s_ij (r_uj - mean_u): учитывает и понравившиеся, и не понравившиеся;
      * ``sum_liked``    — sum_j s_ij [r_uj >= 4]: суммарное сходство с понравившимися книгами.
    """
    name = "icf"

    def __init__(self, k: int = 50, mode: str = "sum_liked", max_neighbors: int = 200, batch: int = 1000):
        self.k, self.mode, self.max_neighbors, self.batch = k, mode, max_neighbors, batch

    def fit(self, ctx, neighbors: tuple[np.ndarray, np.ndarray] | None = None):
        super().fit(ctx)
        self.nbr_idx, self.nbr_sim = neighbors if neighbors is not None else self.compute_neighbors(ctx.R)
        self.set_params(self.k, self.mode)
        self.user_mean = ctx.user_feats["user_mean"].to_numpy(np.float32)
        self.item_mean = ctx.item_feats["item_mean"].to_numpy(np.float32)
        self.B = (ctx.R > 0).astype(np.float32).tocsr()
        self.Lk = (ctx.R >= ctx.cfg.relevance_threshold).astype(np.float32).tocsr()
        return self

    def compute_neighbors(self, R: csr_matrix):
        """Top-k соседей каждой книги пакетами: полная матрица 10000x10000 не хранится."""
        Xn = normalize(R.T.tocsr()).astype(np.float32)
        XnT = Xn.T.tocsr()
        n, k = Xn.shape[0], self.max_neighbors
        idx = np.zeros((n, k), dtype=np.int32)
        sim = np.zeros((n, k), dtype=np.float32)
        for a in range(0, n, self.batch):
            b = min(a + self.batch, n)
            S = (Xn[a:b] @ XnT).toarray()
            S[np.arange(b - a), np.arange(a, b)] = 0.0
            top = np.argpartition(-S, k, axis=1)[:, :k]
            vals = np.take_along_axis(S, top, axis=1)
            order = np.argsort(-vals, axis=1, kind="mergesort")
            idx[a:b] = np.take_along_axis(top, order, axis=1)
            sim[a:b] = np.take_along_axis(vals, order, axis=1)
        return idx, sim

    def set_params(self, k: int, mode: str):
        assert k <= self.nbr_idx.shape[1] and mode in ("avg", "sum_centered", "sum_liked")
        self.k, self.mode = k, mode
        n = self.nbr_idx.shape[0]
        rows = np.repeat(np.arange(n), k)
        S = csr_matrix((self.nbr_sim[:, :k].ravel(), (rows, self.nbr_idx[:, :k].ravel())), shape=(n, n))
        S.eliminate_zeros()
        self.ST = S.T.tocsr()
        return self

    def score(self, users):
        if self.mode == "avg":
            num = (self.ctx.R[users] @ self.ST).toarray()
            den = (self.B[users] @ self.ST).toarray()
            out = np.where(den > 0, num / np.maximum(den, 1e-12), self.item_mean[None, :])
            return out.astype(np.float32)
        if self.mode == "sum_centered":
            Ru = self.ctx.R[users].tocsr(copy=True)
            Ru.data -= np.repeat(self.user_mean[users], np.diff(Ru.indptr))
            return (Ru @ self.ST).toarray().astype(np.float32)
        return (self.Lk[users] @ self.ST).toarray().astype(np.float32)


class SVDScorer(Scorer):
    """SVD (surprise, Funk SVD с bias) из HW_1_FINAL. score = mu + b_u + b_i + p_u·q_i."""
    name = "svd"

    def __init__(self, params: dict | None = None, random_state: int = 42):
        self.params = dict(params or {})
        self.random_state = random_state

    def fit(self, ctx):
        from surprise import SVD, Dataset, Reader

        super().fit(ctx)
        data = Dataset.load_from_df(ctx.train[["user_id", "book_id", "rating"]], Reader(rating_scale=(1, 5)))
        ts = data.build_full_trainset()
        assert ts.n_ratings == len(ctx.train)
        self.algo = SVD(**self.params, random_state=self.random_state)
        self.algo.fit(ts)
        f = self.algo.pu.shape[1]
        raw_u = np.array([ts.to_raw_uid(i) for i in range(ts.n_users)])
        raw_i = np.array([ts.to_raw_iid(i) for i in range(ts.n_items)])
        u_idx = ctx.index.users_to_idx(raw_u)
        i_idx = ctx.index.items_to_idx(raw_i)
        self.mu = float(ts.global_mean)
        self.pu = np.zeros((ctx.n_users, f), np.float32)
        self.bu = np.zeros(ctx.n_users, np.float32)
        self.qi = np.zeros((ctx.n_items, f), np.float32)
        self.bi = np.zeros(ctx.n_items, np.float32)
        self.pu[u_idx], self.bu[u_idx] = self.algo.pu, self.algo.bu
        self.qi[i_idx], self.bi[i_idx] = self.algo.qi, self.algo.bi
        return self

    def score(self, users):
        return (self.mu + self.bu[users, None] + self.bi[None, :] + self.pu[users] @ self.qi.T).astype(np.float32)

    def save(self, path):
        np.savez_compressed(path, mu=self.mu, pu=self.pu, bu=self.bu, qi=self.qi, bi=self.bi,
                            params=str(self.params))


class PureSVDScorer(Scorer):
    """Адаптация SVD для top-N (PureSVD, Cremonesi et al., 2010).

    Усечённое SVD бинарной sparse-матрицы взаимодействий train: score = b_u V V^T. В отличие от
    Funk SVD, модель учится на факте «пользователь оценил книгу», а не на величине оценки.
    """
    name = "psvd"

    def __init__(self, k: int = 128, random_state: int = 42):
        self.k, self.random_state = k, random_state

    def fit(self, ctx):
        from scipy.sparse.linalg import svds

        super().fit(ctx)
        self.B = (ctx.R > 0).astype(np.float32).tocsr()
        _, s, vt = svds(self.B, k=self.k, random_state=self.random_state)
        self.V = vt[np.argsort(-s)].T.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray((self.B[users] @ self.V) @ self.V.T, dtype=np.float32)


class AuthorAffinityScorer(Scorer):
    """Признак взаимодействия: доля train-истории пользователя у автора книги."""
    name = "author"

    def fit(self, ctx):
        super().fit(ctx)
        a = ctx.catalog.author_idx
        keep = a > 0
        self.M = csr_matrix((np.ones(keep.sum(), np.float32), (np.flatnonzero(keep), a[keep])),
                            shape=(ctx.n_items, ctx.catalog.n_authors))
        self.L = liked_matrix(ctx.R, ctx.cfg.relevance_threshold)
        self.n = np.asarray(self.L.sum(axis=1)).ravel()
        return self

    def score(self, users):
        A = (self.L[users] @ self.M).toarray() / np.maximum(self.n[users, None], 1)
        A[:, 0] = 0.0
        return A[:, self.ctx.catalog.author_idx].astype(np.float32)


class TopicAffinityScorer(Scorer):
    """Признак взаимодействия: скалярное произведение тематического профиля и тематик книги."""
    name = "topic"

    def score(self, users):
        return (self.ctx.topic_profile[users] @ self.ctx.catalog.topics.T).astype(np.float32)


CLASSIC_MODELS = ("pop", "cb", "icf", "svd", "psvd")
FEATURE_SIGNALS = ("author", "topic")
MODEL_TITLES = {
    "pop": "Popularity", "cb": "Content-Based", "icf": "Item-Based CF", "svd": "SVD (Funk, HW1)",
    "psvd": "SVD (PureSVD, top-N)", "tt": "Two-Tower", "author": "Author affinity",
    "topic": "Topic affinity",
}
