# src/pipelines.py
from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA

_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

def make_preprocessor(features: List[str], imputer_strategy: str = "median", scaler: str = "standard") -> ColumnTransformer:
    scaler_cls = _SCALERS.get(scaler.lower(), StandardScaler)
    numeric_prep = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_strategy)),
        ("scaler", scaler_cls()),
    ])
    return ColumnTransformer([("num", numeric_prep, features)])

# Optional: PCA-based outlier estimator
class PCAOutlier(BaseEstimator):
    def __init__(self, n_components=None, contamination: float = 0.1, random_state: int | None = None):
        self.n_components = n_components
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components, random_state=self.random_state).fit(X)
        scores = self.decision_function(X)  # higher = more normal (negative error)
        self.threshold_ = np.quantile(scores, self.contamination)  # cut-off learned on train
        return self

    def decision_function(self, X):
        Z = self.pca_.transform(X)
        Xr = self.pca_.inverse_transform(Z)
        err = ((X - Xr) ** 2).sum(axis=1)
        return -np.asarray(err)  # higher = more normal

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s < self.threshold_, -1, 1)

def make_detector(algo: str, algo_params: Dict[str, Any], random_state: int) -> BaseEstimator:
    a = algo.lower()
    p = dict(algo_params or {})
    if a == "iforest":
        return IsolationForest(random_state=random_state, **p)
    if a == "ocsvm":
        # e.g. nu, kernel='rbf', gamma='scale'
        return OneClassSVM(**p)
    if a == "lof":
        # IMPORTANT for inference on new data:
        p.setdefault("novelty", True)
        return LocalOutlierFactor(**p)
    if a == "elliptic":
        return EllipticEnvelope(**p)
    if a == "pca":
        return PCAOutlier(random_state=random_state, **p)
    raise ValueError(f"Unknown algo: {algo}")

def build_pipeline(
    features: List[str],
    algo: str = "iforest",
    algo_params: Dict[str, Any] | None = None,
    random_state: int = 42,
    imputer_strategy: str = "median",
    scaler: str = "standard",
) -> Pipeline:
    prep = make_preprocessor(features, imputer_strategy=imputer_strategy, scaler=scaler)
    detector = make_detector(algo, algo_params or {}, random_state=random_state)
    pipe = Pipeline([("prep", prep), ("detector", detector)])
    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass
    return pipe
