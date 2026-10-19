"""Графики проекта. Каждая функция сохраняет PNG в figures/ и возвращает Figure."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {"model": "#7f8c8d", "hybrid": "#2e86c1"}


def _save(fig, path: Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def metric_bars(summary: pd.DataFrame, metrics: list[str], title: str, path=None):
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 0.42 * len(summary) + 1.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, metrics):
        vals = summary[m]
        colors = [PALETTE.get(g, "#7f8c8d") for g in summary.get("group", pd.Series("model", index=summary.index))]
        ax.barh(summary.index, vals, color=colors)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)
        ax.set_title(m)
        ax.invert_yaxis()
        ax.set_xlim(0, vals.max() * 1.25)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, path)


def tt_history(hist: pd.DataFrame, path=None, title="Two-Tower: обучение"):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(hist["epoch"], hist["train_loss"], "o-", color="#c0392b", label="train loss")
    ax1.set_xlabel("эпоха")
    ax1.set_ylabel("train loss (sampled softmax)", color="#c0392b")
    if "nDCG@10" in hist:
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["nDCG@10"], "s-", color="#2e86c1", label="val nDCG@10")
        ax2.plot(hist["epoch"], hist["Recall@10"], "^--", color="#27ae60", label="val Recall@10")
        flagged = hist[hist["best"].eq(True)] if "best" in hist else hist.iloc[:0]
        best = flagged.iloc[-1] if len(flagged) else hist.loc[hist["nDCG@10"].idxmax()]
        ax2.axvline(best["epoch"], color="gray", ls=":", label=f"лучшая эпоха = {int(best['epoch'])}")
        ax2.set_ylabel("validation")
        ax2.legend(loc="center right")
    ax1.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def tradeoff(df: pd.DataFrame, x: str, y_left: str, y_right: list[str], title: str, path=None, chosen=None):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df[x], df[y_left], "o-", color="#2e86c1", label=y_left)
    ax1.set_xlabel(x)
    ax1.set_ylabel(y_left, color="#2e86c1")
    ax2 = ax1.twinx()
    for col, style in zip(y_right, ["s--", "^--", "d--"]):
        ax2.plot(df[x], df[col], style, label=col)
    ax2.set_ylabel(", ".join(y_right))
    if chosen is not None:
        ax1.axvline(chosen, color="gray", ls=":", label=f"выбрано: {chosen}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def segment_heatmap(table: pd.DataFrame, title: str, path=None):
    fig, ax = plt.subplots(figsize=(1.6 * table.shape[1] + 3, 0.45 * table.shape[0] + 1.5))
    im = ax.imshow(table.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iat[i, j]:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def weights_bars(weights: dict, title: str, path=None):
    df = pd.DataFrame({seg: w for seg, w in weights.items() if not seg.startswith("_") and seg != "new"}).T.fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    df.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_ylabel("вес в смеси (сумма = 1)")
    ax.set_xlabel("сегмент пользователей")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)
    fig.tight_layout()
    return _save(fig, path)


def protocol_scatter(full: pd.Series, sampled: pd.Series, title: str, path=None):
    common = full.index.intersection(sampled.index)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sampled[common], full[common], color="#2e86c1")
    for n in common:
        ax.annotate(n, (sampled[n], full[n]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("nDCG@5, протокол HW1 (relevant + 100 случайных негативов)")
    ax.set_ylabel("nDCG@5, full-catalog")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)
