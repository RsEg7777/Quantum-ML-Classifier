"""
Ensemble Baselines
==================

Classical ensemble methods for comparison with quantum models.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


class ClassicalEnsembleBaseline:
    """
    Wrapper for classical ensemble classifiers.

    Parameters
    ----------
    method : str, default='random_forest'
        Ensemble method: 'random_forest', 'xgboost', 'lightgbm'.
    n_estimators : int, default=100
        Number of estimators.
    seed : int, default=42
        Random seed.
    **kwargs
        Additional parameters for the underlying model.

    Examples
    --------
    >>> baseline = ClassicalEnsembleBaseline(method='random_forest')
    >>> baseline.fit(X_train, y_train)
    >>> accuracy = baseline.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        method: str = "random_forest",
        n_estimators: int = 100,
        seed: int = 42,
        **kwargs,
    ):
        self.method = method
        self.n_estimators = n_estimators
        self.seed = seed
        self.extra_params = kwargs
        self.model = self._create_model()

    def _create_model(self):
        """Create the underlying sklearn/xgboost model."""
        if self.method == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.seed,
                **self.extra_params,
            )
        elif self.method == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.seed,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    **self.extra_params,
                )
            except ImportError:
                raise ImportError("xgboost not installed. pip install xgboost")
        elif self.method == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=self.n_estimators,
                    random_state=self.seed,
                    verbose=-1,
                    **self.extra_params,
                )
            except ImportError:
                raise ImportError("lightgbm not installed. pip install lightgbm")
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ClassicalEnsembleBaseline":
        """Fit the model."""
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns
        -------
        dict
            Dictionary with accuracy and F1 score.
        """
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_weighted": f1_score(y, y_pred, average="weighted"),
        }
