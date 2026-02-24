"""
SVM Baselines
=============

Support Vector Machine classifiers for comparison with quantum kernels.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


class SVMBaseline:
    """
    SVM classifier baseline.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'linear', 'rbf', 'poly', 'precomputed'.
    C : float, default=1.0
        Regularization parameter.
    seed : int, default=42
        Random seed.
    **kwargs
        Additional SVC parameters.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        seed: int = 42,
        **kwargs,
    ):
        self.model = SVC(
            kernel=kernel,
            C=C,
            random_state=seed,
            probability=True,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMBaseline":
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

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(X, y)
