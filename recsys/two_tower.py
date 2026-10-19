"""Two-Tower модель на PyTorch.

Башня пользователя: ID-эмбеддинг (индекс 0 — «новый пользователь») + признаки пользователя
+ тематический профиль -> MLP -> L2-нормированный вектор.
Башня книги: ID-эмбеддинг + эмбеддинг автора + признаки книги + тематики NMF -> MLP -> вектор.

Обучение: позитивы — train-пары с rating >= 4; sampled softmax с in-batch negatives,
logQ-коррекцией популярности и маскированием accidental hits. Early stopping по nDCG@10 на validation.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import ranking_metrics
from .models import FitContext, Scorer

USER_NUM = ["user_log_n", "user_mean", "user_std", "user_share_pos", "user_bias_shrunk",
            "user_mainstreamness", "user_taste_diversity"]
ITEM_NUM = ["item_log_n", "item_mean", "item_std", "item_share_pos", "item_weighted_rating",
            "item_pop_pct", "item_year_norm"]


def _standardize(x: np.ndarray, mean=None, std=None):
    mean = x.mean(axis=0) if mean is None else mean
    std = x.std(axis=0) + 1e-6 if std is None else std
    return ((x - mean) / std).astype(np.float32), mean, std


class TwoTowerNet(nn.Module):
    def __init__(self, n_users, n_items, n_authors, user_dim, item_dim, emb_dim=64, hidden=256,
                 out_dim=64, use_features=True, id_dropout=0.1, author_dim=16):
        super().__init__()
        self.use_features = use_features
        self.id_dropout = id_dropout
        self.user_emb = nn.Embedding(n_users + 1, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        u_in, i_in = emb_dim, emb_dim
        if use_features:
            self.author_emb = nn.Embedding(n_authors, author_dim)
            nn.init.normal_(self.author_emb.weight, std=0.05)
            u_in += user_dim
            i_in += item_dim + author_dim
        self.user_mlp = nn.Sequential(nn.Linear(u_in, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
        self.item_mlp = nn.Sequential(nn.Linear(i_in, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def encode_users(self, uid: torch.Tensor, ux: torch.Tensor) -> torch.Tensor:
        """uid: user_idx + 1 (0 — новый/неизвестный пользователь)."""
        if self.training and self.id_dropout > 0:
            drop = torch.rand(uid.shape[0]) < self.id_dropout
            uid = torch.where(drop, torch.zeros_like(uid), uid)
        h = self.user_emb(uid)
        if self.use_features:
            h = torch.cat([h, ux], dim=1)
        return F.normalize(self.user_mlp(h), dim=1)

    def encode_items(self, iid: torch.Tensor, ix: torch.Tensor, aid: torch.Tensor) -> torch.Tensor:
        h = self.item_emb(iid)
        if self.use_features:
            h = torch.cat([h, ix, self.author_emb(aid)], dim=1)
        return F.normalize(self.item_mlp(h), dim=1)


class TwoTowerScorer(Scorer):
    name = "tt"

    def __init__(self, cfg, use_features: bool = True, verbose: bool = True):
        self.cfg = cfg
        self.use_features = use_features
        self.verbose = verbose
        self.history: list[dict] = []

    # --- подготовка признаков ------------------------------------------------------------------
    def _prepare(self, ctx: FitContext):
        self.ctx = ctx
        ux_raw = np.hstack([ctx.user_feats[USER_NUM].to_numpy(np.float64), ctx.topic_profile])
        ix_raw = np.hstack([ctx.item_feats[ITEM_NUM].to_numpy(np.float64), ctx.catalog.topics])
        self.ux, self.u_mean, self.u_std = _standardize(ux_raw)
        self.ix, _, _ = _standardize(ix_raw)
        self.new_user_x = self._new_user_features(ctx)
        self.t_ux = torch.from_numpy(self.ux)
        self.t_ix = torch.from_numpy(self.ix)
        self.t_aid = torch.from_numpy(ctx.catalog.author_idx.astype(np.int64))
        self.t_iid = torch.arange(ctx.n_items)

    def _new_user_features(self, ctx) -> np.ndarray:
        gm = ctx.item_feats.attrs["global_mean"]
        raw = np.zeros(len(USER_NUM) + ctx.topic_profile.shape[1])
        raw[USER_NUM.index("user_mean")] = gm
        raw[USER_NUM.index("user_mainstreamness")] = 0.5
        return ((raw - self.u_mean) / self.u_std).astype(np.float32)

    def _build_net(self, ctx):
        torch.manual_seed(self.cfg.random_state)
        return TwoTowerNet(
            ctx.n_users, ctx.n_items, ctx.catalog.n_authors, self.ux.shape[1], self.ix.shape[1],
            emb_dim=self.cfg.tt_emb_dim, hidden=self.cfg.tt_hidden, out_dim=self.cfg.tt_out_dim,
            use_features=self.use_features, id_dropout=self.cfg.tt_id_dropout,
        )

    # --- обучение ------------------------------------------------------------------------------
    def fit(self, ctx: FitContext, val_rel=None, val_users=None, n_epochs: int | None = None,
            save_path: Path | None = None):
        """Если заданы val_*, обучение идёт с early stopping по nDCG@10 на validation.
        Иначе — фиксированное число эпох ``n_epochs`` (финальное переобучение на train+val)."""
        cfg = self.cfg
        self._prepare(ctx)
        self.net = self._build_net(ctx)
        R = ctx.R
        pos = (R >= cfg.relevance_threshold).tocsr()
        pos.eliminate_zeros()
        coo = pos.tocoo()
        pu, pi = coo.row.astype(np.int64), coo.col.astype(np.int64)
        item_freq = np.bincount(pi, minlength=ctx.n_items) + 1.0
        logq = torch.from_numpy(np.log(item_freq / item_freq.sum()).astype(np.float32))
        self.pos_matrix = pos

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.tt_lr, weight_decay=cfg.tt_weight_decay)
        gen = torch.Generator().manual_seed(cfg.random_state)
        max_epochs = n_epochs if n_epochs is not None else cfg.tt_max_epochs
        best, best_state, bad = -np.inf, None, 0
        self.history = []
        bs = cfg.tt_batch_size
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            self.net.train()
            perm = torch.randperm(len(pu), generator=gen).numpy()
            total, n_batches = 0.0, 0
            for start in range(0, len(perm), bs):
                b = perm[start:start + bs]
                ub, ib = pu[b], pi[b]
                tu, ti = torch.from_numpy(ub), torch.from_numpy(ib)
                U = self.net.encode_users(tu + 1, self.t_ux[tu])
                I = self.net.encode_items(ti, self.t_ix[ti], self.t_aid[ti])
                logits = U @ I.T / cfg.tt_temperature - logq[ti][None, :]
                # accidental hits: книга-позитив этого же пользователя или дубликат книги в батче
                hit = torch.from_numpy(pos[ub][:, ib].toarray() > 0) | (ti[None, :] == ti[:, None])
                hit.fill_diagonal_(False)
                logits = logits.masked_fill(hit, -1e9)
                loss = F.cross_entropy(logits, torch.arange(len(b)))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            rec = {"epoch": epoch, "train_loss": total / n_batches, "sec": time.perf_counter() - t0}
            if val_rel is not None:
                rec.update(self.evaluate(val_users, val_rel))
                score = rec["nDCG@10"]
                if score > best + cfg.tt_min_delta:
                    best, bad = score, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                    rec["best"] = True
                else:
                    bad += 1
            self.history.append(rec)
            if self.verbose:
                extra = f" val nDCG@10={rec['nDCG@10']:.4f} Recall@10={rec['Recall@10']:.4f}" if val_rel is not None else ""
                print(f"epoch {epoch:2d} loss={rec['train_loss']:.4f}{extra} ({rec['sec']:.1f}s)")
            if val_rel is not None and bad >= cfg.tt_patience:
                if self.verbose:
                    print(f"early stopping: нет улучшения {cfg.tt_patience} эпохи")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_epoch = int(max((r for r in self.history if r.get("best")), key=lambda r: r["epoch"])["epoch"])
        else:
            self.best_epoch = max_epochs
        self._cache_embeddings()
        if save_path is not None:
            self.save(save_path)
        return self

    @torch.no_grad()
    def _cache_embeddings(self):
        self.net.eval()
        self.item_vecs = self.net.encode_items(self.t_iid, self.t_ix, self.t_aid).numpy()
        uid = torch.arange(self.ctx.n_users) + 1
        self.user_vecs = self.net.encode_users(uid, self.t_ux).numpy()
        self.new_user_vec = self.net.encode_users(
            torch.zeros(1, dtype=torch.long), torch.from_numpy(self.new_user_x)[None, :]).numpy()[0]

    @torch.no_grad()
    def evaluate(self, users: np.ndarray, rel_csr, k: int = 10) -> dict:
        self._cache_embeddings()
        S = self.user_vecs[users] @ self.item_vecs.T
        seen = self.ctx.R[users]
        S[seen.nonzero()] = -np.inf
        top = np.argpartition(-S, k, axis=1)[:, :k]
        top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, axis=1), axis=1), axis=1)
        rel = rel_csr[users]
        hits = np.take_along_axis(rel.toarray(), top, axis=1) > 0
        n_rel = np.diff(rel.indptr)
        m = ranking_metrics(hits, n_rel, k)
        return {key: float(v.mean()) for key, v in m.items()}

    # --- интерфейс Scorer ----------------------------------------------------------------------
    def score(self, users):
        return (self.user_vecs[users] @ self.item_vecs.T).astype(np.float32)

    def score_new_user(self) -> np.ndarray:
        return self.item_vecs @ self.new_user_vec

    def save(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "use_features": self.use_features,
                    "best_epoch": self.best_epoch, "history": self.history,
                    "user_feature_names": USER_NUM, "item_feature_names": ITEM_NUM}, path)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
