# src/pipelines.py
from __future__ import annotations
from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest


_SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def make_preprocessor(
    features: List[str],
    imputer_strategy: str = "median",
    scaler: str = "standard",
) -> ColumnTransformer:
    scaler_cls = _SCALERS.get(scaler.lower(), StandardScaler)
    numeric_prep = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", scaler_cls()),
        ]
    )
    prep = ColumnTransformer([("num", numeric_prep, features)])
    return prep


def build_pipeline(
    features: List[str],
    contamination: float = 0.1,
    random_state: int = 42,
    imputer_strategy: str = "median",
    scaler: str = "standard",
) -> Pipeline:
    """Create a preprocessing + IsolationForest pipeline."""
    prep = make_preprocessor(
        features, imputer_strategy=imputer_strategy, scaler=scaler
    )
    pipe = Pipeline(
        [
            ("prep", prep),
            (
                "iforest",
                IsolationForest(contamination=contamination, random_state=random_state),
            ),
        ]
    )
    # Keep pandas through the pipeline when available (sklearn >= 1.2)
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass
    return pipe
