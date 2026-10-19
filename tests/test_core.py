"""Быстрые тесты без реальных данных: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys import hybrid as H  # noqa: E402
from recsys import metrics as M  # noqa: E402
from recsys.data import per_user_holdout  # noqa: E402


def test_scalar_metrics_match_hw1_examples():
    rec, rel = [10, 20, 30, 40, 50], {20, 40, 60}
    assert M.precision_at_k(rec, rel, 5) == 0.4
    assert abs(M.recall_at_k(rec, rel, 5) - 2 / 3) < 1e-12
    assert abs(M.ndcg_at_k(rec, rel, 5) - 0.49818925746641285) < 1e-12
    assert M.ndcg_at_k([20, 40, 60, 10, 30], rel, 5) == 1.0
    assert M.ndcg_at_k([10, 20, 30], set(), 5) == 0.0


def test_vectorized_metrics_equal_scalar():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_items = 30
        rel = set(rng.choice(n_items, size=rng.integers(1, 8), replace=False).tolist())
        rec = rng.choice(n_items, size=10, replace=False).tolist()
        hits = np.array([[i in rel for i in rec]])
        for k in (5, 10):
            v = M.ranking_metrics(hits, np.array([len(rel)]), k)
            assert abs(v[f"Precision@{k}"][0] - M.precision_at_k(rec, rel, k)) < 1e-12
            assert abs(v[f"Recall@{k}"][0] - M.recall_at_k(rec, rel, k)) < 1e-12
            assert abs(v[f"nDCG@{k}"][0] - M.ndcg_at_k(rec, rel, k)) < 1e-12


def test_per_user_holdout_properties():
    rng = np.random.default_rng(1)
    rows = [(u, b, int(rng.integers(1, 6))) for u in range(1, 60) for b in rng.choice(500, rng.integers(1, 40), replace=False)]
    df = pd.DataFrame(rows, columns=["user_id", "book_id", "rating"])
    keep, hold = per_user_holdout(df, 0.2, 42)
    assert len(keep) + len(hold) == len(df)
    assert set(keep.index).isdisjoint(hold.index)
    assert set(hold["user_id"]) <= set(keep["user_id"])
    n = df.groupby("user_id").size()
    expected = np.minimum(np.maximum(1, np.floor(0.2 * n)), n - 1)
    assert (hold.groupby("user_id").size().reindex(n.index, fill_value=0) == expected).all()
    keep2, hold2 = per_user_holdout(df, 0.2, 42)
    assert keep.equals(keep2) and hold.equals(hold2)


def test_zscore_excludes_seen():
    S = np.arange(12, dtype=np.float32).reshape(2, 6)
    seen = csr_matrix(np.array([[1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]], dtype=np.float32))
    Z = H.zscore_unseen(S, seen)
    assert np.isneginf(Z[0, 0]) and np.isneginf(Z[0, 5]) and np.isneginf(Z[1, 2])
    unseen = Z[0, 1:5]
    assert abs(unseen.mean()) < 1e-5 and abs(unseen.std() - 1) < 1e-3


def test_mmr_lambda_one_equals_topk():
    rng = np.random.default_rng(2)
    U, P = 5, 40
    items = np.tile(np.arange(P, dtype=np.int32), (U, 1))
    pool = H.Pool(np.arange(U), items, np.zeros((U, P, 1), np.float32), ("x",), np.ones((U, P, 1), bool),
                  ("g",), np.zeros((U, P), bool), np.ones(U), np.array(["short"] * U), np.ones(U, int))
    scores = rng.random((U, P))
    emb = rng.normal(size=(P, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert (H.mmr_rerank(pool, scores, emb, 1.0, 10, 30) == H.topk_positions(scores, 10)).all()
    div = H.mmr_rerank(pool, scores, emb, 0.3, 10, 30)
    assert all(len(set(r)) == 10 for r in div)
