"""
Baseline classifiers used for comparison in the paper (Figure 5/6): KNN, RF, NN,
each trained on price-only features and on price+sentiment ("combined") features.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ModelSpec:
    name: str
    build_fn: callable
    param_grid: dict = field(default_factory=dict)


def build_knn(k: int = 15):
    return KNeighborsClassifier(n_neighbors=k, weights="distance")


def build_rf(n_estimators: int = 200, max_depth: int = 6, max_features="sqrt"):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
        random_state=42, n_jobs=-1,
    )


def build_nn(hidden_layer_sizes=(64, 32), alpha: float = 1e-3, learning_rate_init: float = 1e-3):
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
        learning_rate_init=learning_rate_init, max_iter=200,
        early_stopping=True, n_iter_no_change=5, random_state=42,
    )


MODEL_REGISTRY = {
    "knn": build_knn,
    "rf": build_rf,
    "nn": build_nn,
}


def scale_to_unit_range(X: np.ndarray, scaler: MinMaxScaler | None = None):
    """Standardize features to [-1, 1] as done in the paper before ML training."""
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
