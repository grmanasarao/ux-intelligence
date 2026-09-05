from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ontology(path: str | Path | None = None) -> dict:
    ontology_path = Path(path) if path else CONFIG_DIR / "ux_ontology.json"
    with ontology_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ontology_labels(ontology: dict) -> list[str]:
    labels: list[str] = []
    for namespace, values in ontology["namespaces"].items():
        labels.extend(f"{namespace}:{value}" for value in values)
    return labels


def ensure_directories() -> None:
    for path in (DATA_DIR, MODELS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def normalize_labels(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        raw = str(value).split(";")
    cleaned = []
    for item in raw:
        label = str(item).strip().lower()
        if label and label not in cleaned:
            cleaned.append(label)
    return cleaned
