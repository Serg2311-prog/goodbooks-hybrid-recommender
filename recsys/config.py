"""Единая конфигурация проекта: пути, seeds и гиперпараметры."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_DIR.parent / "archive.zip"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

# Служебные теги, найденные в EDA HW_1_FINAL: описывают статус чтения, а не содержание книги.
NOISE_TAGS = frozenset({
    "to-read", "currently-reading", "favorites", "favourites", "owned",
    "books-i-own", "owned-books", "library", "kindle", "to-buy", "audiobook",
    "ebook", "default", "my-books", "re-read",
})

# Метрики ранжирования HW_1_FINAL (fair evaluation, 1000 пользователей, 100 негативов).
HW1_REFERENCE = {
    "Popularity": {"Precision@5": 0.0324, "Recall@5": 0.068690, "nDCG@5": 0.052966},
    "Item-Based CF": {"Precision@5": 0.0206, "Recall@5": 0.047056, "nDCG@5": 0.034864},
    "SVD": {"Precision@5": 0.0318, "Recall@5": 0.066516, "nDCG@5": 0.050817},
}


@dataclass
class Config:
    data_path: Path = field(
        default_factory=lambda: Path(os.environ.get("GOODBOOKS_DATA", DEFAULT_DATA_PATH))
    )
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    figures_dir: Path = FIGURES_DIR

    # --- воспроизводимость и разбиение ---
    random_state: int = 42          # тот же seed и та же процедура test-split, что в HW_1_FINAL
    val_random_state: int = 43      # отдельный seed для выделения validation из train
    holdout_fraction: float = 0.2
    relevance_threshold: float = 4.0

    # --- сегменты пользователей по длине train-истории ---
    # 0 -> new, 1..5 -> short, 6..20 -> medium, >20 -> active
    segment_edges: tuple = (5, 20)

    # --- оценка ---
    k_values: tuple = (5, 10)
    tune_k: int = 10                # подбор по nDCG@10: при K=10 метрика менее шумная, чем при K=5
    n_val_users: int = 10_000
    n_tune_users: int = 3_000       # подмножество validation для подбора гиперпараметров отдельных моделей
    n_test_users: int = 10_000
    n_bootstrap: int = 1_000

    # --- протокол HW_1_FINAL (relevant + 100 случайных негативов) ---
    hw1_eval_users: int = 1_000
    hw1_negatives: int = 100
    hw1_k: int = 5
    hw1_icf_neighbors: int = 20

    # --- контентные признаки ---
    top_n_tags: int = 25
    n_topics: int = 20
    content_emb_dim: int = 64
    min_author_books: int = 2

    # --- Item-Based CF ---
    icf_max_neighbors: int = 200
    icf_neighbors_grid: tuple = (20, 50, 100, 200)
    icf_modes: tuple = ("avg", "sum_centered", "sum_liked")

    # --- SVD (surprise) ---
    svd_hw1_params: dict = field(default_factory=lambda: dict(
        n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02,
    ))
    svd_grid: tuple = (
        dict(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02),
        dict(n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.05),
        dict(n_factors=150, n_epochs=30, lr_all=0.005, reg_all=0.08),
    )

    # --- PureSVD (адаптация SVD для top-N) ---
    psvd_grid: tuple = (64, 128, 256, 384)

    # --- Two-Tower ---
    tt_emb_dim: int = 64
    tt_hidden: int = 256
    tt_out_dim: int = 64
    tt_batch_size: int = 2048
    tt_lr: float = 2e-3
    tt_weight_decay: float = 1e-6
    tt_temperature: float = 0.07
    tt_max_epochs: int = 25
    tt_patience: int = 3
    tt_min_delta: float = 1e-3      # улучшение val nDCG@10 меньше этого порога не считается
    tt_id_dropout: float = 0.15
    tt_val_users: int = 4_000

    # --- гибрид ---
    candidates_per_model: int = 50
    n_random_weights: int = 300
    mmr_top_m: int = 50
    mmr_lambdas: tuple = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
    mmr_max_rel_drop: float = 0.03   # допустимое относительное падение nDCG@10 на validation
    switch_min_history: int = 5      # порог правила-переключателя из HW_1_FINAL

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.results_dir, self.figures_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
