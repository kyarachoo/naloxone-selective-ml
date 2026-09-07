from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(path: Path | None = None) -> dict[str, Any]:
    path = path or project_root() / "config" / "settings.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_feature_registry(path: Path | None = None) -> pd.DataFrame:
    path = path or project_root() / "config" / "feature_registry.csv"
    registry = pd.read_csv(path)
    for col in ["available_by_discharge", "include_in_model"]:
        registry[col] = registry[col].astype(str).str.lower().eq("true")
    return registry
