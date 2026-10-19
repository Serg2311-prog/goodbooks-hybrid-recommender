"""Протоколы оценки.

1. Full-catalog (основной): каждая модель ранжирует весь каталог без прочитанных книг; гибриды
   ранжируют общий пул кандидатов. Ground truth — книги holdout-части с rating >= 4.
2. Sampled (протокол HW_1_FINAL): relevant + 100 случайных негативов, 1000 пользователей, seed 42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from . import metrics as M
from .data import Index
from .hybrid import Pool, pool_items, pool_metrics, zscore_unseen
from .models import FitContext


def relevance_matrix(df: pd.DataFrame, index: Index, threshold: float) -> csr_matrix:
    rel = df[df["rating"] >= threshold]
    u = index.users_to_idx(rel["user_id"].to_numpy())
    i = index.items_to_idx(rel["book_id"].to_numpy())
    return csr_matrix((np.ones(len(rel), np.float32), (u, i)), shape=(index.n_users, index.n_items))


def sample_eval_users(rel_csr: csr_matrix, ctx: FitContext, n: int, seed: int) -> np.ndarray:
    """Пользователи с >= 1 релевантной holdout-книгой и непустой train-историей."""
    has_rel = np.diff(rel_csr.indptr) > 0
    has_hist = np.diff(ctx.R.indptr) > 0
    eligible = np.flatnonzero(has_rel & has_hist)
    rng = np.random.default_rng(seed)
    if n >= len(eligible):
        return eligible
    return np.sort(rng.choice(eligible, size=n, replace=False))


def item_popularity_prob(ctx: FitContext) -> np.ndarray:
    n_users_active = max(int((np.diff(ctx.R.indptr) > 0).sum()), 1)
    return ctx.item_feats["item_n_ratings"].to_numpy() / n_users_active


def evaluate_lists(pool: Pool, pos: np.ndarray, ctx: FitContext, ks=(5, 10)) -> tuple[dict, pd.DataFrame]:
    """Метрики релевантности (per user) + разнообразие/покрытие для списка top-K."""
    per_user = pool_metrics(pool, pos, ks)
    k = max(ks)
    top = pool_items(pool, pos[:, :k])
    cat = ctx.catalog
    per_user[f"ILD@{k}"] = M.intra_list_diversity(top, cat.content_emb)
    per_user[f"Novelty@{k}"] = M.novelty(top, item_popularity_prob(ctx))
    per_user[f"AuthorDiv@{k}"] = M.author_diversity(top, cat.author_idx)
    pop_pct = ctx.item_feats["item_pop_pct"].to_numpy()
    per_user[f"AvgPopPct@{k}"] = pop_pct[np.clip(top, 0, None)].mean(axis=1)
    summary = {name: float(v.mean()) for name, v in per_user.items()}
    summary[f"Coverage@{k}"] = M.catalog_coverage(top, ctx.n_items)
    summary[f"Gini@{k}"] = M.gini_index(top, ctx.n_items)
    df = pd.DataFrame(per_user)
    df["user_idx"] = pool.users
    df["segment"] = pool.segment
    return summary, df


def candidate_recall(pool: Pool) -> pd.DataFrame:
    """Какая доля релевантных holdout-книг попала в пул, всего и от каждого генератора."""
    total = pool.n_rel.sum()
    rows = [{"generator": "union (pool)", "recall_of_pool": pool.labels.sum() / total,
             "avg_pool_size": float((pool.items >= 0).sum(axis=1).mean())}]
    for g, name in enumerate(pool.generator_names):
        hits = (pool.labels & pool.source[..., g]).sum()
        only = (pool.labels & pool.source[..., g] & (pool.source.sum(axis=-1) == 1)).sum()
        rows.append({"generator": name, "recall_of_pool": hits / total, "unique_hits_share": only / max(pool.labels.sum(), 1)})
    return pd.DataFrame(rows)


def segment_table(per_user: dict[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    rows = {}
    for name, df in per_user.items():
        rows[name] = df.groupby("segment")[metric].mean()
        rows[name]["all"] = df[metric].mean()
    out = pd.DataFrame(rows).T
    order = [c for c in ("short", "medium", "active", "all") if c in out.columns]
    return out[order]


# ----------------------------------------------------------------------------------------------
# Протокол HW_1_FINAL: relevant + 100 случайных негативов на пользователя
# ----------------------------------------------------------------------------------------------
def hw1_candidates(test: pd.DataFrame, train_full: pd.DataFrame, books: pd.DataFrame, cfg):
    """Точное воспроизведение выборки пользователей и негативов HW_1_FINAL."""
    relevant_by_user, seen_by_user = {}, {}
    mask = test["rating"].to_numpy() >= cfg.relevance_threshold
    for u, b in test.loc[mask, ["user_id", "book_id"]].itertuples(index=False):
        relevant_by_user.setdefault(int(u), set()).add(int(b))
    for u, b in train_full[["user_id", "book_id"]].itertuples(index=False):
        seen_by_user.setdefault(int(u), set()).add(int(b))
    catalog = np.sort(books["id"].astype(np.int64).unique())
    eligible = np.array(sorted(relevant_by_user), dtype=np.int64)
    rng_users = np.random.default_rng(cfg.random_state)
    users = np.sort(rng_users.choice(eligible, size=min(cfg.hw1_eval_users, len(eligible)), replace=False))
    rng_neg = np.random.default_rng(cfg.random_state)
    out = []
    for u in users:
        u = int(u)
        rel, seen = relevant_by_user[u], seen_by_user.get(u, set())
        banned = np.fromiter(seen | rel, dtype=np.int64)
        neg_pool = np.setdiff1d(catalog, banned)
        neg = rng_neg.choice(neg_pool, size=min(cfg.hw1_negatives, len(neg_pool)), replace=False)
        cands = sorted(set(rel) | set(int(x) for x in neg))
        assert rel.isdisjoint(seen) and set(int(x) for x in neg).isdisjoint(seen | rel)
        out.append((u, cands, rel))
    return out


def hw1_item_cf_scores(ctx: FitContext, user_idx: int, cand_items: np.ndarray, neighbors: int) -> np.ndarray:
    """Item-CF scorer из HW_1_FINAL без изменений (top-20 похожих среди книг истории)."""
    item_mean = ctx.item_feats["item_mean"].to_numpy()
    scores = item_mean[cand_items].astype(float)
    row = ctx.R[user_idx]
    if row.nnz == 0:
        return scores
    X = ctx.extra.setdefault("item_user", ctx.R.T.tocsr())
    sims = cosine_similarity(X[cand_items], X[row.indices])
    k = min(neighbors, sims.shape[1])
    top = np.argpartition(sims, kth=-k, axis=1)[:, -k:] if k < sims.shape[1] else np.broadcast_to(np.arange(k), (sims.shape[0], k))
    ts = np.take_along_axis(sims, top, axis=1)
    tr = row.data.astype(float)[top]
    w = np.where(ts > 0, ts, 0.0)
    ws = w.sum(axis=1)
    ok = ws > 0
    scores[ok] = np.clip((w[ok] * tr[ok]).sum(axis=1) / ws[ok], 1, 5)
    return scores


def run_hw1_protocol(ctx: FitContext, cands, score_fns: dict, k: int = 5) -> tuple[pd.DataFrame, dict]:
    """score_fns: имя -> f(user_idx, cand_item_idx) -> scores. Один candidate set для всех моделей."""
    per = {name: {f"Precision@{k}": [], f"Recall@{k}": [], f"nDCG@{k}": []} for name in score_fns}
    for user_id, cand_ids, rel in cands:
        uidx = int(ctx.index.users_to_idx(np.array([user_id]))[0])
        cand_items = Index.items_to_idx(cand_ids)
        for name, fn in score_fns.items():
            s = np.asarray(fn(uidx, cand_items), dtype=float)
            order = np.argsort(-s, kind="mergesort")[:k]
            top = [cand_ids[i] for i in order]
            per[name][f"Precision@{k}"].append(M.precision_at_k(top, rel, k))
            per[name][f"Recall@{k}"].append(M.recall_at_k(top, rel, k))
            per[name][f"nDCG@{k}"].append(M.ndcg_at_k(top, rel, k))
    table = pd.DataFrame({name: {m: float(np.mean(v)) for m, v in d.items()} for name, d in per.items()}).T
    return table, per


def z_signals_for_user(ctx: FitContext, scorers: dict, uidx: int) -> dict[str, np.ndarray]:
    users = np.array([uidx])
    seen = ctx.R[users]
    return {n: zscore_unseen(s.score(users), seen)[0] for n, s in scorers.items()}
