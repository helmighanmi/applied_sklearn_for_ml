# src/utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
import sklearn


def load_config(path: str | Path) -> Dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_meta(meta: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def ensure_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in DataFrame: {missing}")


def annotate_with_iforest(
    df: pd.DataFrame,
    pipe,
    features: List[str],
    score_col: str = "anomaly_score",
    pred_col: str = "anomaly_pred",
    flag_col: str = "anomaly_flag",
) -> pd.DataFrame:
    X = df[features]
    scores = pipe.decision_function(X)
    preds = pipe.predict(X)

    out = df.copy()
    out[score_col] = scores
    out[pred_col] = preds
    out[flag_col] = out[pred_col] == -1
    return out


def default_meta(cfg: Dict, model_path: str | Path) -> Dict:
    return {
        "features": cfg.get("features"),
        "contamination": cfg.get("contamination"),
        "random_state": cfg.get("random_state"),
        "imputer_strategy": cfg.get("imputer_strategy"),
        "scaler": cfg.get("scaler"),
        "sklearn_version": sklearn.__version__,
        "model_path": str(model_path),
    }
