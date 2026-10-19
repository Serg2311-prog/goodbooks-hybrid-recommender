"""Вспомогательные функции: seeds, таймер, сохранение артефактов."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


@contextmanager
def timer(label: str, verbose: bool = True):
    start = time.perf_counter()
    yield
    if verbose:
        print(f"[{label}] {time.perf_counter() - start:.1f} c")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def batched(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)
