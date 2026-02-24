"""
Gaussian Process Baseline
=========================

GP classifier for comparison with quantum models.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics import accuracy_score, f1_score


class GPBaseline:
    """
    Gaussian Process classifier baseline.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel: 'rbf' or 'matern'.
    seed : int, default=42
        Random seed.
    """

    def __init__(self, kernel: str = "rbf", seed: int = 42):
        kern = RBF() if kernel == "rbf" else Matern()
        self.model = GaussianProcessClassifier(
            kernel=kern, random_state=seed, max_iter_predict=100,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
        }
