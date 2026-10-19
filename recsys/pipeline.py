"""Сквозной пайплайн: данные -> разбиение -> признаки -> модели -> кандидаты -> ранжирование -> оценка.

Запуск из папки FINAL_PROJECT:  ``python -m recsys.pipeline``  (``--fast`` — быстрый прогон).

Схема использования данных:
  * этап VALIDATION: всё обучается на ``train``; на ``val`` подбираются гиперпараметры моделей,
    веса гибрида, lambda MMR и эпоха early stopping Two-Tower;
  * этап TEST: все модели заново обучаются на ``train_full = train + val`` с зафиксированными
    гиперпараметрами; ``test`` используется один раз — для финальной оценки.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch

from . import evaluation as E
from . import hybrid as H
from .config import HW1_REFERENCE, PROJECT_DIR, Config
from .data import Index, Splits, build_index, clean_ratings, load_raw, make_splits, validate_raw
from .features import Catalog, build_catalog
from .metrics import paired_bootstrap, ranking_metrics
from .models import (CLASSIC_MODELS, FEATURE_SIGNALS, MODEL_TITLES, AuthorAffinityScorer,
                     ContentScorer, FitContext, ItemCFScorer, PopularityScorer, PureSVDScorer,
                     SVDScorer, TopicAffinityScorer, fit_context)
from .two_tower import TwoTowerScorer
from .utils import save_json, set_seed, timer


# ----------------------------------------------------------------------------------------------
# 1. Данные
# ----------------------------------------------------------------------------------------------
@dataclass
class Prepared:
    raw: dict
    quality_report: pd.DataFrame
    interactions: pd.DataFrame
    splits: Splits
    index: Index
    catalog: Catalog


def prepare_data(cfg: Config) -> Prepared:
    raw = load_raw(cfg)
    report = validate_raw(raw)
    interactions = clean_ratings(raw["ratings"])
    splits = make_splits(interactions, cfg)
    index = build_index(interactions, raw["books"])
    catalog = build_catalog(raw, cfg)
    return Prepared(raw, report, interactions, splits, index, catalog)


# ----------------------------------------------------------------------------------------------
# 2. Модели: подбор гиперпараметров на validation и обучение
# ----------------------------------------------------------------------------------------------
@dataclass
class ComponentParams:
    pop_kind: str = "weighted_rating"
    cb_profile: str = "base"
    icf_mode: str = "avg"
    icf_k: int = 20
    svd_params: dict = field(default_factory=dict)
    psvd_k: int = 128
    tt_epochs: int | None = None


def full_catalog_metrics(scorer, ctx: FitContext, users: np.ndarray, rel_csr, k: int = 10,
                         batch: int = 1000) -> dict:
    """nDCG@k / Recall@k одной модели при ранжировании всего каталога без прочитанного."""
    hits_all, nrel_all = [], []
    for a in range(0, len(users), batch):
        ub = users[a:a + batch]
        Z = H.zscore_unseen(scorer.score(ub), ctx.R[ub])
        top = H._row_topn(Z, k)
        rel = rel_csr[ub]
        hits_all.append(np.take_along_axis(rel.toarray(), top, axis=1) > 0)
        nrel_all.append(np.diff(rel.indptr))
    m = ranking_metrics(np.vstack(hits_all), np.concatenate(nrel_all), k)
    return {key: float(v.mean()) for key, v in m.items()}


def svd_rmse(svd: SVDScorer, ctx: FitContext, df: pd.DataFrame) -> float:
    u = ctx.index.users_to_idx(df["user_id"].to_numpy())
    i = ctx.index.items_to_idx(df["book_id"].to_numpy())
    pred = np.clip(svd.mu + svd.bu[u] + svd.bi[i] + np.einsum("ij,ij->i", svd.pu[u], svd.qi[i]), 1, 5)
    return float(np.sqrt(np.mean((pred - df["rating"].to_numpy()) ** 2)))


def tune_components(ctx: FitContext, val: pd.DataFrame, rel_val, users: np.ndarray, cfg: Config,
                    verbose: bool = True):
    """Подбор гиперпараметров отдельных моделей на validation (модели обучены на train)."""
    rows = []

    def log(model, variant, metrics, **extra):
        rows.append({"model": model, "variant": variant, **metrics, **extra})
        if verbose:
            print(f"{model:<14} {variant:<38} nDCG@10={metrics['nDCG@10']:.4f}"
                  + "".join(f" {k}={v:.4f}" for k, v in extra.items()))

    p = ComponentParams()
    best = {}
    for kind in PopularityScorer.KINDS:
        m = full_catalog_metrics(PopularityScorer(kind).fit(ctx), ctx, users, rel_val)
        log("Popularity", kind, m)
        best.setdefault("pop", (kind, m["nDCG@10"]))
        if m["nDCG@10"] > best["pop"][1]:
            best["pop"] = (kind, m["nDCG@10"])
    for prof in ("base", "enhanced"):
        m = full_catalog_metrics(ContentScorer(prof).fit(ctx), ctx, users, rel_val)
        log("Content-Based", prof, m)
        if m["nDCG@10"] > best.get("cb", (None, -1))[1]:
            best["cb"] = (prof, m["nDCG@10"])
    icf = ItemCFScorer(k=cfg.icf_neighbors_grid[0], mode="avg", max_neighbors=cfg.icf_max_neighbors).fit(ctx)
    for mode in cfg.icf_modes:
        for k in cfg.icf_neighbors_grid:
            icf.set_params(k, mode)
            m = full_catalog_metrics(icf, ctx, users, rel_val)
            log("Item-Based CF", f"mode={mode}, k={k}", m)
            if m["nDCG@10"] > best.get("icf", (None, -1))[1]:
                best["icf"] = ((mode, k), m["nDCG@10"])
    for params in cfg.svd_grid:
        svd = SVDScorer(params, cfg.random_state).fit(ctx)
        m = full_catalog_metrics(svd, ctx, users, rel_val)
        rmse = svd_rmse(svd, ctx, val)
        log("SVD (Funk)", str(params), m, val_RMSE=rmse)
        # Funk SVD — модель предсказания оценки: выбираем по RMSE на validation.
        if rmse < best.get("svd", (None, np.inf))[1]:
            best["svd"] = (dict(params), rmse)
    for k in cfg.psvd_grid:
        m = full_catalog_metrics(PureSVDScorer(k, cfg.random_state).fit(ctx), ctx, users, rel_val)
        log("PureSVD", f"k={k}", m)
        if m["nDCG@10"] > best.get("psvd", (None, -1))[1]:
            best["psvd"] = (k, m["nDCG@10"])
    p.pop_kind = best["pop"][0]
    p.cb_profile = best["cb"][0]
    p.icf_mode, p.icf_k = best["icf"][0]
    p.svd_params = best["svd"][0]
    p.psvd_k = best["psvd"][0]
    return p, pd.DataFrame(rows), icf


def fit_classic_scorers(ctx: FitContext, p: ComponentParams, cfg: Config, verbose: bool = True) -> dict:
    sc = {}
    with timer("Popularity", verbose):
        sc["pop"] = PopularityScorer(p.pop_kind).fit(ctx)
    with timer("Content-Based", verbose):
        sc["cb"] = ContentScorer(p.cb_profile).fit(ctx)
    with timer("Item-Based CF (top-k соседей, батчами)", verbose):
        sc["icf"] = ItemCFScorer(k=p.icf_k, mode=p.icf_mode, max_neighbors=max(p.icf_k, 20)).fit(ctx)
    with timer("SVD (Funk, surprise)", verbose):
        sc["svd"] = SVDScorer(p.svd_params, cfg.random_state).fit(ctx)
    with timer("PureSVD", verbose):
        sc["psvd"] = PureSVDScorer(p.psvd_k, cfg.random_state).fit(ctx)
    sc["author"] = AuthorAffinityScorer().fit(ctx)
    sc["topic"] = TopicAffinityScorer().fit(ctx)
    return sc


# ----------------------------------------------------------------------------------------------
# 3. Варианты гибрида
# ----------------------------------------------------------------------------------------------
ALL_GENERATORS = CLASSIC_MODELS + ("tt",)


@dataclass
class Variant:
    name: str
    kind: str                      # single | switch | weighted
    generators: tuple
    weights: dict = field(default_factory=dict)   # {сигнал: w} или {сегмент: {сигнал: w}}
    mmr_lambda: float = 1.0
    signals: tuple = ()
    per_segment: bool = False
    group: str = "hybrid"

    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "generators": list(self.generators),
                "weights": self.weights, "mmr_lambda": self.mmr_lambda, "signals": list(self.signals),
                "per_segment": self.per_segment}


def rank(pool: H.Pool, v: Variant, catalog: Catalog, cfg: Config, k: int = 10) -> np.ndarray:
    valid = pool.valid(v.generators)
    if v.kind == "switch":
        scores = H.blend_per_user(pool, H.switch_weights(pool.n_history, cfg.switch_min_history), valid)
    else:
        scores = H.blend(pool, v.weights, valid)
    return H.mmr_rerank(pool, scores, catalog.content_emb, v.mmr_lambda, k, cfg.mmr_top_m)


def single_variants(names: tuple) -> list[Variant]:
    return [Variant(MODEL_TITLES[n], "single", (n,), {n: 1.0}, signals=(n,), group="model") for n in names]


SWITCH_VARIANT = Variant("Hybrid-Switch (HW1)", "switch", ("pop", "cb", "svd"), signals=("pop", "cb", "svd"))
HYBRID_ORDER = ("Hybrid-Weighted", "Hybrid-Weighted+F", "Hybrid-Segment", "Hybrid-Improved",
                "Hybrid-Weighted+TT", "Hybrid-Segment+TT", "Hybrid-Improved+TT")


def comparison_variants(variants: dict, with_tt: bool = True) -> list[Variant]:
    """Единый список для сравнения: отдельные модели, гибрид HW1 и все варианты нового гибрида."""
    models = CLASSIC_MODELS + (("tt",) if with_tt else ())
    return single_variants(models) + [SWITCH_VARIANT] + [variants[n] for n in HYBRID_ORDER if n in variants]


def tune_hybrids(pool: H.Pool, ctx: FitContext, cfg: Config, with_tt: bool = True, verbose: bool = True):
    """Подбор весов и lambda MMR на validation-пуле. Возвращает варианты и таблицы подбора."""
    k = cfg.tune_k
    classic = CLASSIC_MODELS
    out, logs = {}, {}

    def global_w(signals, gens, name):
        w, trials = H.tune_weights(pool, signals, pool.valid(gens), k, cfg.n_random_weights, cfg.random_state)
        logs[name] = trials
        return w

    def seg_w(signals, gens, name):
        w, trials = H.tune_segment_weights(pool, signals, pool.valid(gens), k, cfg.n_random_weights, cfg.random_state)
        logs[name] = trials
        return w

    with timer("Hybrid-Weighted: глобальные веса", verbose):
        out["Hybrid-Weighted"] = Variant("Hybrid-Weighted", "weighted", classic,
                                         global_w(classic, classic, "weighted"), signals=classic)
    sig_f = classic + FEATURE_SIGNALS
    with timer("Hybrid-Weighted+F: + признаки взаимодействий", verbose):
        out["Hybrid-Weighted+F"] = Variant("Hybrid-Weighted+F", "weighted", classic,
                                           global_w(sig_f, classic, "weighted_f"), signals=sig_f)
    with timer("Hybrid-Segment: веса по сегментам + признаки взаимодействий", verbose):
        out["Hybrid-Segment"] = Variant("Hybrid-Segment", "weighted", classic,
                                        seg_w(sig_f, classic, "segment"), signals=sig_f, per_segment=True)
    if with_tt:
        gens_tt = classic + ("tt",)
        with timer("Hybrid-Weighted+TT", verbose):
            out["Hybrid-Weighted+TT"] = Variant("Hybrid-Weighted+TT", "weighted", gens_tt,
                                                global_w(classic + ("tt",), gens_tt, "weighted_tt"),
                                                signals=classic + ("tt",))
        sig_tt = sig_f + ("tt",)
        with timer("Hybrid-Segment+TT", verbose):
            out["Hybrid-Segment+TT"] = Variant("Hybrid-Segment+TT", "weighted", gens_tt,
                                               seg_w(sig_tt, gens_tt, "segment_tt"), signals=sig_tt,
                                               per_segment=True)
    mmr_tables = {}
    for base in [n for n in ("Hybrid-Segment", "Hybrid-Segment+TT") if n in out]:
        table, lam = tune_mmr(pool, out[base], ctx, cfg)
        mmr_tables[base] = table
        name = "Hybrid-Improved" + ("+TT" if base.endswith("+TT") else "")
        out[name] = replace(out[base], name=name, mmr_lambda=lam)
    return out, logs, mmr_tables


def tune_mmr(pool: H.Pool, v: Variant, ctx: FitContext, cfg: Config):
    """Кривая релевантность/разнообразие по lambda; выбирается минимальная lambda,
    при которой nDCG@10 на validation падает не более чем на cfg.mmr_max_rel_drop (относительно)."""
    rows = []
    base_top = None
    for lam in cfg.mmr_lambdas:
        pos = rank(pool, replace(v, mmr_lambda=lam), ctx.catalog, cfg)
        top = H.pool_items(pool, pos)
        base_top = top if base_top is None else base_top
        overlap = np.mean([len(np.intersect1d(a, b)) / len(a) for a, b in zip(top, base_top)])
        summ, _ = E.evaluate_lists(pool, pos, ctx)
        rows.append({"lambda": lam, **{m: summ[m] for m in ("nDCG@10", "Recall@10", "ILD@10", "AuthorDiv@10",
                                                             "Coverage@10", "Novelty@10")},
                     "overlap_with_lambda1": overlap})
    df = pd.DataFrame(rows)
    ref = df.loc[df["lambda"] == 1.0, "nDCG@10"].iat[0]
    ok = df[df["nDCG@10"] >= ref * (1 - cfg.mmr_max_rel_drop)]
    lam = float(ok["lambda"].min())
    df["chosen"] = df["lambda"] == lam
    return df, lam


def popularity_tradeoff(pool: H.Pool, v: Variant, ctx: FitContext, cfg: Config,
                        alphas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)) -> pd.DataFrame:
    """score = (1 - a) * персональный score гибрида + a * z(Popularity)."""
    valid = pool.valid(v.generators)
    if v.per_segment:
        W = {s: {n: w for n, w in ws.items() if n != "pop"} for s, ws in v.weights.items()}
    else:
        W = {n: w for n, w in v.weights.items() if n != "pop"}
    personal = np.where(valid, H.blend(pool, W, valid), 0.0)
    pop = pool.signal("pop")
    rows = []
    for a in alphas:
        scores = np.where(valid, (1 - a) * personal + a * pop, -np.inf)
        pos = H.topk_positions(scores, 10)
        summ, _ = E.evaluate_lists(pool, pos, ctx)
        rows.append({"alpha_pop": a, **{m: summ[m] for m in ("nDCG@10", "Recall@10", "Novelty@10",
                                                             "AvgPopPct@10", "Coverage@10", "Gini@10")}})
    return pd.DataFrame(rows)


def evaluate_variants(pool: H.Pool, variants: list[Variant], ctx: FitContext, cfg: Config):
    summary, per_user, tops = [], {}, {}
    for v in variants:
        pos = rank(pool, v, ctx.catalog, cfg, k=max(cfg.k_values))
        s, df = E.evaluate_lists(pool, pos, ctx, cfg.k_values)
        summary.append({"model": v.name, "group": v.group, **s})
        per_user[v.name] = df
        tops[v.name] = H.pool_items(pool, pos)
    return pd.DataFrame(summary).set_index("model"), per_user, tops


# ----------------------------------------------------------------------------------------------
# 4. Протокол HW_1_FINAL на test и cold start
# ----------------------------------------------------------------------------------------------
def sampled_protocol(prep: Prepared, ctx: FitContext, scorers: dict, variants: list[Variant], cfg: Config):
    cands = E.hw1_candidates(prep.splits.test, prep.splits.train_full, prep.raw["books"], cfg)
    users = ctx.index.users_to_idx(np.array([c[0] for c in cands]))
    Z = {}
    for a in range(0, len(users), 500):
        ub = users[a:a + 500]
        seen = ctx.R[ub]
        for n, s in scorers.items():
            Z.setdefault(n, []).append(H.zscore_unseen(s.score(ub), seen))
    Z = {n: np.vstack(v) for n, v in Z.items()}
    row_of = {int(u): r for r, u in enumerate(users)}
    names = tuple(scorers)
    seg = ctx.user_feats["segment"].to_numpy()
    n_hist = ctx.user_feats["user_n_ratings"].to_numpy()
    svd_hw1 = scorers["svd"]
    if svd_hw1.params != cfg.svd_hw1_params:
        svd_hw1 = SVDScorer(cfg.svd_hw1_params, cfg.random_state).fit(ctx)
    svd_algo = svd_hw1.algo
    wr = ctx.item_feats["item_weighted_rating"].to_numpy()

    def hybrid_fn(v: Variant):
        def fn(uidx, items):
            r = row_of[uidx]
            X = np.stack([Z[n][r, items] for n in names], axis=1)
            if v.kind == "switch":
                w = H.switch_weights(np.array([n_hist[uidx]]), cfg.switch_min_history)[0]
            else:
                w = v.weights.get(seg[uidx], v.weights.get("_global")) if v.per_segment else v.weights
            s = X @ H.weight_vector(w, names)
            if v.mmr_lambda >= 1.0:
                return s
            p = H.Pool(np.array([uidx]), items[None, :].astype(np.int32), X[None], names,
                       np.ones((1, len(items), 1), bool), ("all",), np.zeros((1, len(items)), bool),
                       np.zeros(1), np.array([seg[uidx]]), np.array([n_hist[uidx]]))
            pos = H.mmr_rerank(p, s[None, :], ctx.catalog.content_emb, v.mmr_lambda, cfg.hw1_k, cfg.mmr_top_m)[0]
            out = np.full(len(items), -np.inf)
            out[pos] = np.arange(len(pos), 0, -1)
            return out
        return fn

    fns = {
        "Popularity (код HW1)": lambda u, it: wr[it],
        "Item-Based CF (код HW1)": lambda u, it: E.hw1_item_cf_scores(ctx, u, it, cfg.hw1_icf_neighbors),
        "SVD (код HW1)": lambda u, it: np.array([svd_algo.predict(int(ctx.index.user_ids[u]), int(i) + 1).est for i in it]),
    }
    for v in variants:
        if v.kind == "single":
            n = v.generators[0]
            fns[v.name] = (lambda name: (lambda u, it: Z[name][row_of[u], it]))(n)
        else:
            fns[v.name] = hybrid_fn(v)
    table, per = E.run_hw1_protocol(ctx, cands, fns, cfg.hw1_k)
    ref = pd.DataFrame(HW1_REFERENCE).T.add_suffix(" (HW1)")
    ref.index = [f"{i} (код HW1)" for i in ref.index]
    return table.join(ref), per, len(cands)


def cold_start_experiment(ctx: FitContext, scorers: dict, rel_test, users: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Новый пользователь: история скрыта, доступны только неперсональные сигналы.
    Прочитанные в train книги исключаются из выдачи, чтобы не штрафовать модели за них."""
    rng = np.random.default_rng(cfg.random_state)
    seen = ctx.R[users]
    rel = rel_test[users]
    n_rel = np.diff(rel.indptr)
    reld = rel.toarray()
    cand = {f"Popularity ({kind})": ctx.item_feats[col].to_numpy(np.float32)
            for kind, col in PopularityScorer.KINDS.items()}
    if "tt" in scorers:
        cand["Two-Tower (вектор нового пользователя)"] = scorers["tt"].score_new_user().astype(np.float32)
    rows = []
    for name, vec in cand.items():
        Z = H.zscore_unseen(np.broadcast_to(vec, (len(users), len(vec))), seen)
        top = H._row_topn(Z, 10)
        m = ranking_metrics(np.take_along_axis(reld, top, axis=1) > 0, n_rel, 10)
        rows.append({"model": name, **{k: float(v.mean()) for k, v in m.items()},
                     "Coverage@10": len(np.unique(top)) / ctx.n_items})
    rnd = rng.random((len(users), ctx.n_items)).astype(np.float32)
    Z = H.zscore_unseen(rnd, seen)
    top = H._row_topn(Z, 10)
    m = ranking_metrics(np.take_along_axis(reld, top, axis=1) > 0, n_rel, 10)
    rows.append({"model": "Random", **{k: float(v.mean()) for k, v in m.items()},
                 "Coverage@10": len(np.unique(top)) / ctx.n_items})
    return pd.DataFrame(rows).set_index("model")


def bootstrap_table(per_user: dict, pairs: list[tuple[str, str]], metric: str, cfg: Config) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        r = paired_bootstrap(per_user[a][metric].to_numpy(), per_user[b][metric].to_numpy(),
                             cfg.n_bootstrap, cfg.random_state)
        rows.append({"A": a, "B": b, "metric": metric, **r,
                     "significant_95": r["ci_low"] > 0 or r["ci_high"] < 0})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 5. Полный прогон (CLI)
# ----------------------------------------------------------------------------------------------
def fast_config(cfg: Config) -> Config:
    """Уменьшенные выборки и число эпох для быстрой проверки работоспособности.

    Артефакты пишутся в fast_check/, чтобы не затирать результаты полного прогона."""
    out = PROJECT_DIR / "fast_check"
    return replace(cfg, n_val_users=2000, n_tune_users=1000, n_test_users=2000, tt_max_epochs=3,
                   n_random_weights=100, hw1_eval_users=200, svd_grid=cfg.svd_grid[:1], psvd_grid=(128,),
                   models_dir=out / "models", results_dir=out / "results", figures_dir=out / "figures")


def run_pipeline(cfg: Config | None = None, fast: bool = False, verbose: bool = True) -> dict:
    cfg = cfg or Config()
    if fast:
        cfg = fast_config(cfg)
    cfg.ensure_dirs()
    set_seed(cfg.random_state)
    torch.set_num_threads(max(1, __import__("os").cpu_count() or 1))
    t0 = time.perf_counter()
    res = {}

    with timer("1. Загрузка, проверка, очистка, разбиение, каталог", verbose):
        prep = prepare_data(cfg)
    s = prep.splits

    # --- этап VALIDATION ------------------------------------------------------------------------
    ctx_v = fit_context(s.train, prep.index, prep.catalog, cfg)
    rel_val = E.relevance_matrix(s.val, prep.index, cfg.relevance_threshold)
    tune_users = E.sample_eval_users(rel_val, ctx_v, cfg.n_tune_users, cfg.random_state + 1)
    with timer("2. Подбор гиперпараметров моделей на validation", verbose):
        params, comp_table, _ = tune_components(ctx_v, s.val, rel_val, tune_users, cfg, verbose)
    sc_v = fit_classic_scorers(ctx_v, params, cfg, verbose)
    tt_users = E.sample_eval_users(rel_val, ctx_v, cfg.tt_val_users, cfg.random_state + 2)
    with timer("3. Two-Tower: обучение на train, early stopping по validation", verbose):
        tt_v = TwoTowerScorer(cfg, use_features=True, verbose=verbose).fit(
            ctx_v, val_rel=rel_val, val_users=tt_users, save_path=cfg.models_dir / "two_tower_val_best.pt")
    params.tt_epochs = tt_v.best_epoch
    sc_v["tt"] = tt_v
    val_users = E.sample_eval_users(rel_val, ctx_v, cfg.n_val_users, cfg.random_state + 3)
    with timer("4. Пул кандидатов validation", verbose):
        pool_v = H.build_pool(ctx_v, sc_v, val_users, rel_val, ALL_GENERATORS, cfg.candidates_per_model)
    with timer("5. Подбор весов гибрида и lambda MMR на validation", verbose):
        variants, _, mmr_tables = tune_hybrids(pool_v, ctx_v, cfg, with_tt=True, verbose=verbose)

    # --- этап TEST ------------------------------------------------------------------------------
    ctx_t = fit_context(s.train_full, prep.index, prep.catalog, cfg)
    rel_test = E.relevance_matrix(s.test, prep.index, cfg.relevance_threshold)
    with timer("6. Переобучение моделей на train+val", verbose):
        sc_t = fit_classic_scorers(ctx_t, params, cfg, verbose)
        sc_t["tt"] = TwoTowerScorer(cfg, use_features=True, verbose=verbose).fit(
            ctx_t, n_epochs=params.tt_epochs, save_path=cfg.models_dir / "two_tower_final.pt")
    test_users = E.sample_eval_users(rel_test, ctx_t, cfg.n_test_users, cfg.random_state + 4)
    with timer("7. Пул кандидатов test и финальная оценка", verbose):
        pool_t = H.build_pool(ctx_t, sc_t, test_users, rel_test, ALL_GENERATORS, cfg.candidates_per_model)
        all_variants = comparison_variants(variants)
        summary, per_user, _ = evaluate_variants(pool_t, all_variants, ctx_t, cfg)
    with timer("8. Протокол HW_1_FINAL (relevant + 100 негативов)", verbose):
        sampled, _, _ = sampled_protocol(prep, ctx_t, sc_t, all_variants, cfg)

    save_artifacts(cfg, params, variants, sc_t)
    summary.to_csv(cfg.results_dir / "test_full_catalog.csv")
    sampled.to_csv(cfg.results_dir / "test_sampled_hw1_protocol.csv")
    comp_table.to_csv(cfg.results_dir / "val_component_tuning.csv", index=False)
    for name, t in mmr_tables.items():
        t.to_csv(cfg.results_dir / f"val_mmr_{name}.csv", index=False)
    res.update(summary=summary, sampled=sampled, params=params, variants=variants,
               seconds=time.perf_counter() - t0)
    if verbose:
        print(summary[["nDCG@5", "Recall@5", "nDCG@10", "Recall@10", "ILD@10", "Coverage@10"]].round(4))
        print(f"Всего: {res['seconds'] / 60:.1f} мин")
    return res


def save_artifacts(cfg: Config, params: ComponentParams, variants: dict, scorers: dict) -> None:
    cfg.ensure_dirs()
    scorers["svd"].save(cfg.models_dir / "svd_funk_final.npz")
    np.savez_compressed(cfg.models_dir / "puresvd_final.npz", V=scorers["psvd"].V)
    icf = scorers["icf"]
    np.savez_compressed(cfg.models_dir / "item_cf_neighbors_final.npz",
                        idx=icf.nbr_idx[:, :icf.k], sim=icf.nbr_sim[:, :icf.k])
    save_json({"component_params": params.__dict__,
               "hybrid_variants": {n: v.to_dict() for n, v in variants.items()},
               "config": cfg.to_dict()}, cfg.models_dir / "hybrid_config.json")


def main():
    ap = argparse.ArgumentParser(description="Гибридная рекомендательная система Goodbooks-10k")
    ap.add_argument("--fast", action="store_true", help="быстрый прогон на подвыборке")
    args = ap.parse_args()
    run_pipeline(fast=args.fast)


if __name__ == "__main__":
    main()
