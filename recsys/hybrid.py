"""Гибридная система: генерация кандидатов -> нормализация -> взвешенное смешивание -> MMR.

Нормализация: для каждого пользователя score каждой модели переводится в z-score по всем ещё
не прочитанным книгам каталога. Так нормированный score не зависит от состава пула кандидатов,
и одни и те же веса применимы и к пулу, и к любому внешнему candidate set (протокол HW_1_FINAL).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .features import SEGMENTS
from .metrics import ranking_metrics
from .models import FitContext, Scorer


def zscore_unseen(S: np.ndarray, seen: csr_matrix) -> np.ndarray:
    """Per-user z-score по непрочитанным книгам; прочитанные -> -inf (исключение из выдачи)."""
    S = np.array(S, dtype=np.float32, copy=True)
    r, c = seen.nonzero()
    n_unseen = S.shape[1] - np.diff(seen.indptr)
    S[r, c] = 0.0
    total = S.sum(axis=1, dtype=np.float64)
    sq = np.einsum("ij,ij->i", S, S, dtype=np.float64)
    mean = total / n_unseen
    var = np.maximum(sq / n_unseen - mean ** 2, 0.0)
    Z = (S - mean[:, None].astype(np.float32)) / (np.sqrt(var)[:, None].astype(np.float32) + 1e-6)
    Z[r, c] = -np.inf
    return Z


@dataclass
class Pool:
    users: np.ndarray            # (U,) user_idx
    items: np.ndarray            # (U, P) item_idx, -1 = пустая позиция
    signals: np.ndarray          # (U, P, S) z-scores
    signal_names: tuple
    source: np.ndarray           # (U, P, G) книга пришла от генератора g
    generator_names: tuple
    labels: np.ndarray           # (U, P) релевантна в holdout
    n_rel: np.ndarray            # (U,) всего релевантных в holdout (знаменатель Recall/nDCG)
    segment: np.ndarray          # (U,) сегмент активности
    n_history: np.ndarray        # (U,) длина train-истории

    def signal(self, name: str) -> np.ndarray:
        return self.signals[..., self.signal_names.index(name)]

    def valid(self, generators: tuple | None = None) -> np.ndarray:
        v = self.items >= 0
        if generators is not None:
            g = [self.generator_names.index(x) for x in generators]
            v &= self.source[..., g].any(axis=-1)
        return v

    def subset(self, mask: np.ndarray) -> "Pool":
        return Pool(self.users[mask], self.items[mask], self.signals[mask], self.signal_names,
                    self.source[mask], self.generator_names, self.labels[mask], self.n_rel[mask],
                    self.segment[mask], self.n_history[mask])


def _row_topn(Z: np.ndarray, n: int) -> np.ndarray:
    n = min(n, Z.shape[1])
    if n == Z.shape[1]:
        top = np.broadcast_to(np.arange(n), Z.shape).copy()
    else:
        top = np.argpartition(-Z, n, axis=1)[:, :n]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(Z, top, axis=1), axis=1, kind="mergesort"), axis=1)


def build_pool(ctx: FitContext, scorers: dict[str, Scorer], users: np.ndarray, rel_csr: csr_matrix,
               generators: tuple, n_per_model: int, batch: int = 500) -> Pool:
    """Общий пул: объединение top-N каждого генератора (прочитанное исключено) + score всех сигналов."""
    names = tuple(scorers)
    n_items = ctx.n_items
    P = n_per_model * len(generators)
    U = len(users)
    items = np.full((U, P), -1, np.int32)
    signals = np.zeros((U, P, len(names)), np.float32)
    source = np.zeros((U, P, len(generators)), bool)
    labels = np.zeros((U, P), bool)
    for a in range(0, U, batch):
        ub = users[a:a + batch]
        b = len(ub)
        seen = ctx.R[ub]
        Z = {name: zscore_unseen(scorers[name].score(ub), seen) for name in names}
        tops = [_row_topn(Z[g], n_per_model) for g in generators]
        cat = np.sort(np.concatenate(tops, axis=1), axis=1)
        dup = np.zeros_like(cat, dtype=bool)
        dup[:, 1:] = cat[:, 1:] == cat[:, :-1]
        cat = np.where(dup, n_items, cat)
        cat = np.sort(cat, axis=1)
        valid = cat < n_items
        safe = np.where(valid, cat, 0)
        rows = np.arange(b)[:, None]
        items[a:a + b] = np.where(valid, cat, -1)
        for s, name in enumerate(names):
            signals[a:a + b, :, s] = np.where(valid, Z[name][rows, safe], 0.0)
        for g, top in enumerate(tops):
            member = np.zeros((b, n_items), bool)
            member[rows, top] = True
            source[a:a + b, :, g] = member[rows, safe] & valid
        rel = rel_csr[ub].toarray().astype(bool)
        labels[a:a + b] = rel[rows, safe] & valid
    n_hist = ctx.user_feats["user_n_ratings"].to_numpy()[users].astype(int)
    return Pool(users=users, items=items, signals=signals, signal_names=names, source=source,
                generator_names=tuple(generators), labels=labels,
                n_rel=np.diff(rel_csr[users].indptr), segment=ctx.user_feats["segment"].to_numpy()[users],
                n_history=n_hist)


# ----------------------------------------------------------------------------------------------
# Смешивание и ранжирование внутри пула
# ----------------------------------------------------------------------------------------------
def weight_vector(weights: dict, names: tuple) -> np.ndarray:
    w = np.array([weights.get(n, 0.0) for n in names], dtype=np.float32)
    return w / w.sum() if w.sum() > 0 else w


def blend(pool: Pool, weights, valid: np.ndarray | None = None) -> np.ndarray:
    """weights: dict сигнал->вес (общие) или dict сегмент->dict (по сегментам)."""
    valid = pool.valid() if valid is None else valid
    if weights and isinstance(next(iter(weights.values())), dict):
        W = np.stack([weight_vector(weights.get(seg, {}), pool.signal_names) for seg in pool.segment])
        scores = np.einsum("ups,us->up", pool.signals, W)
    else:
        scores = pool.signals @ weight_vector(weights, pool.signal_names)
    return np.where(valid, scores, -np.inf)


def topk_positions(scores: np.ndarray, k: int) -> np.ndarray:
    return _row_topn(scores, k)


def pool_metrics(pool: Pool, pos: np.ndarray, ks=(5, 10)) -> dict[str, np.ndarray]:
    hits = np.take_along_axis(pool.labels, pos, axis=1)
    hits &= np.take_along_axis(pool.items, pos, axis=1) >= 0
    out = {}
    for k in ks:
        out.update(ranking_metrics(hits, pool.n_rel, k))
    return out


def pool_items(pool: Pool, pos: np.ndarray) -> np.ndarray:
    return np.take_along_axis(pool.items, pos, axis=1)


def mean_ndcg(pool: Pool, scores: np.ndarray, k: int) -> float:
    pos = topk_positions(scores, k)
    return float(pool_metrics(pool, pos, ks=(k,))[f"nDCG@{k}"].mean())


def switch_weights(n_history: np.ndarray, min_history: int) -> list[dict]:
    """Правило HW_1_FINAL: нет истории -> Popularity, история < min_history -> Content-Based, иначе SVD."""
    return [{"pop": 1.0} if n == 0 else {"cb": 1.0} if n < min_history else {"svd": 1.0} for n in n_history]


def blend_per_user(pool: Pool, per_user_weights: list[dict], valid=None) -> np.ndarray:
    W = np.stack([weight_vector(w, pool.signal_names) for w in per_user_weights])
    scores = np.einsum("ups,us->up", pool.signals, W)
    return np.where(pool.valid() if valid is None else valid, scores, -np.inf)


# ----------------------------------------------------------------------------------------------
# Подбор весов на validation
# ----------------------------------------------------------------------------------------------
def tune_weights(pool: Pool, signals: tuple, valid: np.ndarray, k: int, n_random: int, seed: int,
                 refine_rounds: int = 3) -> tuple[dict, pd.DataFrame]:
    """Случайный поиск по симплексу весов (Dirichlet) + локальное уточнение. Цель — mean nDCG@k."""
    rng = np.random.default_rng(seed)
    idx = [pool.signal_names.index(s) for s in signals]
    X = pool.signals[..., idx]
    m = len(signals)
    cands = [np.eye(m)[i] for i in range(m)] + [np.ones(m) / m]
    cands += list(rng.dirichlet(np.ones(m), size=n_random))
    cands += list(rng.dirichlet(np.full(m, 0.3), size=n_random // 2))

    def evaluate(w):
        return mean_ndcg(pool, np.where(valid, X @ w.astype(np.float32), -np.inf), k)

    trials = [(w, evaluate(w)) for w in cands]
    best_w, best_s = max(trials, key=lambda t: t[1])
    step = 0.5
    for _ in range(refine_rounds):
        improved = False
        for j in range(m):
            for factor in (1 - step, 1 + step, 0.0):
                w = best_w.copy()
                w[j] = w[j] * factor if factor else 0.0
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                s = evaluate(w)
                trials.append((w, s))
                if s > best_s + 1e-6:
                    best_w, best_s, improved = w, s, True
        step /= 2
        if not improved and step < 0.1:
            break
    df = pd.DataFrame([dict(zip(signals, w), **{f"nDCG@{k}": s}) for w, s in trials])
    return dict(zip(signals, map(float, best_w))), df


def tune_segment_weights(pool: Pool, signals: tuple, valid: np.ndarray, k: int, n_random: int, seed: int,
                         min_users: int = 300) -> tuple[dict, dict]:
    """Глобальные веса + отдельные веса для каждого сегмента активности (если в нём достаточно пользователей)."""
    global_w, _ = tune_weights(pool, signals, valid, k, n_random, seed)
    result, trials = {}, {}
    for seg in SEGMENTS:
        mask = pool.segment == seg
        if mask.sum() < min_users:
            result[seg] = dict(global_w)
            continue
        w, df = tune_weights(pool.subset(mask), signals, valid[mask], k, n_random, seed)
        result[seg], trials[seg] = w, df
    result["_global"] = global_w
    return result, trials


# ----------------------------------------------------------------------------------------------
# MMR: баланс релевантности и разнообразия
# ----------------------------------------------------------------------------------------------
def mmr_rerank(pool: Pool, scores: np.ndarray, emb: np.ndarray, lam: float, k: int, top_m: int,
               chunk: int = 2000) -> np.ndarray:
    """Maximal Marginal Relevance: argmax lam*rel(i) - (1-lam)*max_{j in S} sim(i, j).

    rel — score гибрида, min-max нормированный внутри top_m кандидатов пользователя;
    sim — косинус контентных эмбеддингов книг. Возвращает позиции в пуле (U, k).
    """
    base = topk_positions(scores, top_m)
    k = min(k, base.shape[1])
    if lam >= 1.0:
        return base[:, :k]
    out = np.zeros((len(scores), k), dtype=base.dtype)
    for a in range(0, len(scores), chunk):
        cand = base[a:a + chunk]
        s = np.take_along_axis(scores[a:a + chunk], cand, axis=1)
        finite = np.isfinite(s)
        s_min = np.where(finite, s, np.inf).min(axis=1, keepdims=True)
        s_max = np.where(finite, s, -np.inf).max(axis=1, keepdims=True)
        rel = np.where(finite, (s - s_min) / np.maximum(s_max - s_min, 1e-9), -np.inf)
        items = np.take_along_axis(pool.items[a:a + chunk], cand, axis=1)
        E = emb[np.clip(items, 0, None)]
        sim = np.clip(np.einsum("umd,und->umn", E, E), 0, None)
        n = len(cand)
        rows = np.arange(n)
        max_sim = np.zeros_like(rel)
        avail = finite.copy()
        for step in range(k):
            val = lam * rel - (1 - lam) * max_sim
            val = np.where(avail, val, -np.inf)
            j = val.argmax(axis=1)
            out[a + rows, step] = cand[rows, j]
            avail[rows, j] = False
            max_sim = np.maximum(max_sim, sim[rows, j, :])
    return out


# ----------------------------------------------------------------------------------------------
# Сервисный класс: рекомендации для конкретного пользователя
# ----------------------------------------------------------------------------------------------
@dataclass
class HybridRecommender:
    ctx: FitContext
    scorers: dict
    weights: dict                 # сегмент -> {сигнал: вес}
    generators: tuple
    mmr_lambda: float
    n_per_model: int
    mmr_top_m: int
    name: str = "hybrid"
    new_user_strategy: str = "pop"   # pop | tt — выбирается на validation (cold-start эксперимент)
    meta: dict = field(default_factory=dict)

    def _user_signals(self, uidx: int) -> dict[str, np.ndarray]:
        users = np.array([uidx])
        seen = self.ctx.R[users]
        return {n: zscore_unseen(s.score(users), seen)[0] for n, s in self.scorers.items()}

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        cat = self.ctx.catalog
        uidx = int(self.ctx.index.users_to_idx(np.array([user_id]))[0])
        n_hist = 0 if uidx < 0 else int(self.ctx.user_feats["user_n_ratings"].iat[uidx])
        if n_hist == 0:
            segment = "new"
            name = "tt" if self.new_user_strategy == "tt" and "tt" in self.scorers else "pop"
            vec = (self.scorers["tt"].score_new_user() if name == "tt"
                   else self.scorers["pop"].item_score).astype(np.float32)
            sig = {name: (vec - vec.mean()) / (vec.std() + 1e-6)}
            w = {name: 1.0}
            gens = (name,)
        else:
            segment = str(self.ctx.user_feats["segment"].iat[uidx])
            sig = self._user_signals(uidx)
            w = self.weights.get(segment, self.weights.get("_global"))
            gens = tuple(g for g in self.generators if g in sig)
        pool_ids = np.unique(np.concatenate([np.argsort(-sig[g], kind="mergesort")[: self.n_per_model] for g in gens]))
        pool_ids = pool_ids[np.isfinite(sig[gens[0]][pool_ids])]
        names = tuple(sig)
        X = np.stack([sig[s][pool_ids] for s in names], axis=1)
        wv = weight_vector(w, names)
        score = X @ wv
        p = Pool(users=np.array([max(uidx, 0)]), items=pool_ids[None, :].astype(np.int32),
                 signals=X[None], signal_names=names, source=np.ones((1, len(pool_ids), 1), bool),
                 generator_names=("all",), labels=np.zeros((1, len(pool_ids)), bool), n_rel=np.zeros(1),
                 segment=np.array([segment]), n_history=np.array([n_hist]))
        pos = mmr_rerank(p, score[None, :], cat.content_emb, self.mmr_lambda, n, self.mmr_top_m)[0]
        chosen = pool_ids[pos]
        contrib = X[pos] * wv[None, :]
        books = cat.books.iloc[chosen]
        out = pd.DataFrame({
            "rank": np.arange(1, len(chosen) + 1),
            "book_id": books["id"].to_numpy(),
            "title": books["title"].to_numpy(),
            "authors": books["main_author"].to_numpy(),
            "topic": [cat.topic_names[t].split(",")[0] for t in books["dominant_topic"]],
            "hybrid_score": score[pos].round(3),
            "main_signal": [names[j] for j in contrib.argmax(axis=1)],
        })
        out.attrs.update(segment=segment, n_history=n_hist, weights=w)
        return out
