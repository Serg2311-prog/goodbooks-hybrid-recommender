"""Метрики ранжирования (определения HW_1_FINAL) и метрики разнообразия/покрытия."""
from __future__ import annotations

import numpy as np


# --- эталонные скалярные версии из HW_1_FINAL --------------------------------------------------
def precision_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    if len(recommended) == 0:
        return 0.0
    return len(set(recommended) & set(relevant)) / k


def recall_at_k(recommended, relevant, k=5):
    if len(relevant) == 0:
        return 0.0
    return len(set(list(recommended)[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k=5):
    recommended = list(recommended)[:k]
    if len(relevant) == 0:
        return 0.0
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# --- векторизованные версии: hits (n_users, K) bool, n_rel (n_users,) --------------------------
_DISCOUNT = 1.0 / np.log2(np.arange(2, 202))


def ranking_metrics(hits: np.ndarray, n_rel: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Per-user Precision@k, Recall@k, nDCG@k; совпадают со скалярными версиями выше."""
    h = hits[:, :k].astype(np.float64)
    n_rel = np.asarray(n_rel, dtype=np.float64)
    n_hits = h.sum(axis=1)
    dcg = h @ _DISCOUNT[: h.shape[1]]
    idcg = np.cumsum(_DISCOUNT[:k])[np.clip(np.minimum(n_rel, k).astype(int) - 1, 0, None)]
    ok = n_rel > 0
    return {
        f"Precision@{k}": n_hits / k,
        f"Recall@{k}": np.where(ok, n_hits / np.maximum(n_rel, 1), 0.0),
        f"nDCG@{k}": np.where(ok, dcg / idcg, 0.0),
    }


def intra_list_diversity(topk: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """ILD: средняя попарная косинусная дистанция (1 - cos) книг в списке, per user."""
    valid = topk >= 0
    E = emb[np.where(valid, topk, 0)] * valid[..., None]
    sims = np.einsum("ukd,ujd->ukj", E, E)
    n = valid.sum(axis=1)
    pair_sum = (sims.sum(axis=(1, 2)) - np.einsum("ukk->u", sims)) / 2
    n_pairs = n * (n - 1) / 2
    return np.where(n_pairs > 0, 1.0 - pair_sum / np.maximum(n_pairs, 1), 0.0)


def novelty(topk: np.ndarray, item_pop_prob: np.ndarray) -> np.ndarray:
    """Средняя самоинформация -log2 p(i), p(i) — доля train-пользователей, оценивших книгу."""
    p = np.clip(item_pop_prob[np.clip(topk, 0, None)], 1e-9, 1.0)
    return (-np.log2(p) * (topk >= 0)).sum(axis=1) / np.maximum((topk >= 0).sum(axis=1), 1)


def catalog_coverage(topk: np.ndarray, n_items: int) -> float:
    return len(np.unique(topk[topk >= 0])) / n_items


def author_diversity(topk: np.ndarray, author_idx: np.ndarray) -> np.ndarray:
    """Доля уникальных авторов в списке (редкие авторы считаются различными)."""
    a = author_idx[np.clip(topk, 0, None)].astype(np.int64)
    a = np.where(a == 0, -1 - np.clip(topk, 0, None), a)
    s = np.sort(a, axis=1)
    uniq = 1 + (np.diff(s, axis=1) != 0).sum(axis=1)
    return uniq / topk.shape[1]


def gini_index(topk: np.ndarray, n_items: int) -> float:
    """Gini частот попадания книг в рекомендации (0 — равномерно, 1 — одна книга у всех)."""
    counts = np.bincount(topk[topk >= 0].ravel(), minlength=n_items).astype(np.float64)
    counts.sort()
    n = len(counts)
    cum = np.cumsum(counts)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n) if cum[-1] > 0 else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Доверительный интервал разности средних mean(a) - mean(b) по пользователям."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_value_two_sided": float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))),
    }
