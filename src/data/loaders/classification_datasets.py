"""
Classification Dataset Loaders
==============================

Load and preprocess standard classification datasets for quantum ML experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.datasets import load_iris as sklearn_load_iris
from sklearn.datasets import load_wine as sklearn_load_wine
from sklearn.datasets import load_breast_cancer as sklearn_load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class DatasetInfo:
    """
    Container for dataset metadata and splits.

    Attributes
    ----------
    name : str
        Dataset name.
    n_features : int
        Number of features.
    n_classes : int
        Number of target classes.
    class_names : list of str
        Names of target classes.
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    X_val : np.ndarray, optional
        Validation features.
    y_val : np.ndarray, optional
        Validation labels.
    """

    name: str
    n_features: int
    n_classes: int
    class_names: List[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name='{self.name}', "
            f"n_features={self.n_features}, "
            f"n_classes={self.n_classes}, "
            f"train_size={len(self.X_train)}, "
            f"test_size={len(self.X_test)})"
        )


def _preprocess(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.0,
    normalize: Union[str, bool] = "minmax",
    seed: int = 42,
    n_features: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Preprocess data: split, normalize, and optionally reduce features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction for validation set (from training data).
    normalize : str or bool
        Normalization method: 'minmax', 'standard', or False.
    seed : int
        Random seed for reproducibility.
    n_features : int, optional
        Number of features to keep (uses PCA if less than original).

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) or
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Validation split from training data
    X_val, y_val = None, None
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
        )

    # Normalize
    if normalize:
        if normalize == "minmax" or normalize is True:
            scaler = MinMaxScaler(feature_range=(0, np.pi))
        elif normalize == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization: {normalize}")

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if X_val is not None:
            X_val = scaler.transform(X_val)

    # Reduce features if requested
    if n_features is not None and n_features < X_train.shape[1]:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_features)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        if X_val is not None:
            X_val = pca.transform(X_val)

    if X_val is not None:
        return X_train, X_val, X_test, y_train, y_val, y_test
    return X_train, X_test, y_train, y_test


def load_iris(
    test_size: float = 0.2,
    val_size: float = 0.0,
    normalize: Union[str, bool] = "minmax",
    seed: int = 42,
    n_features: Optional[int] = None,
    binary: bool = False,
) -> DatasetInfo:
    """
    Load the Iris dataset.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of data for testing.
    val_size : float, default=0.0
        Fraction of training data for validation.
    normalize : str or bool, default='minmax'
        Normalization: 'minmax' (scales to [0, π]), 'standard', or False.
    seed : int, default=42
        Random seed.
    n_features : int, optional
        Number of features (uses PCA if less than 4).
    binary : bool, default=False
        If True, use only classes 0 and 1 for binary classification.

    Returns
    -------
    DatasetInfo
        Dataset with train/test splits.

    Examples
    --------
    >>> data = load_iris()
    >>> print(data)
    DatasetInfo(name='iris', n_features=4, n_classes=3, ...)

    >>> data = load_iris(binary=True, n_features=2)
    >>> print(data.n_features, data.n_classes)
    2 2
    """
    iris = sklearn_load_iris()
    X, y = iris.data, iris.target
    class_names = list(iris.target_names)

    # Binary classification: keep only classes 0 and 1
    if binary:
        mask = y < 2
        X, y = X[mask], y[mask]
        class_names = class_names[:2]

    # Preprocess
    result = _preprocess(
        X, y, test_size, val_size, normalize, seed, n_features
    )

    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = result
    else:
        X_train, X_test, y_train, y_test = result
        X_val, y_val = None, None

    return DatasetInfo(
        name="iris" + ("_binary" if binary else ""),
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y)),
        class_names=class_names,
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        X_val=X_val.astype(np.float32) if X_val is not None else None,
        y_val=y_val.astype(np.int32) if y_val is not None else None,
    )


def load_wine(
    test_size: float = 0.2,
    val_size: float = 0.0,
    normalize: Union[str, bool] = "minmax",
    seed: int = 42,
    n_features: Optional[int] = None,
) -> DatasetInfo:
    """
    Load the Wine dataset.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of data for testing.
    val_size : float, default=0.0
        Fraction of training data for validation.
    normalize : str or bool, default='minmax'
        Normalization method.
    seed : int, default=42
        Random seed.
    n_features : int, optional
        Number of features to keep.

    Returns
    -------
    DatasetInfo
        Dataset with train/test splits.
    """
    wine = sklearn_load_wine()
    X, y = wine.data, wine.target
    class_names = list(wine.target_names)

    result = _preprocess(X, y, test_size, val_size, normalize, seed, n_features)

    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = result
    else:
        X_train, X_test, y_train, y_test = result
        X_val, y_val = None, None

    return DatasetInfo(
        name="wine",
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y)),
        class_names=class_names,
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        X_val=X_val.astype(np.float32) if X_val is not None else None,
        y_val=y_val.astype(np.int32) if y_val is not None else None,
    )


def load_breast_cancer(
    test_size: float = 0.2,
    val_size: float = 0.0,
    normalize: Union[str, bool] = "minmax",
    seed: int = 42,
    n_features: Optional[int] = None,
) -> DatasetInfo:
    """
    Load the Breast Cancer Wisconsin dataset.

    Parameters
    ----------
    test_size : float, default=0.2
        Fraction of data for testing.
    val_size : float, default=0.0
        Fraction of training data for validation.
    normalize : str or bool, default='minmax'
        Normalization method.
    seed : int, default=42
        Random seed.
    n_features : int, optional
        Number of features to keep.

    Returns
    -------
    DatasetInfo
        Dataset with train/test splits.
    """
    cancer = sklearn_load_breast_cancer()
    X, y = cancer.data, cancer.target
    class_names = list(cancer.target_names)

    result = _preprocess(X, y, test_size, val_size, normalize, seed, n_features)

    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = result
    else:
        X_train, X_test, y_train, y_test = result
        X_val, y_val = None, None

    return DatasetInfo(
        name="breast_cancer",
        n_features=X_train.shape[1],
        n_classes=len(np.unique(y)),
        class_names=class_names,
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        X_val=X_val.astype(np.float32) if X_val is not None else None,
        y_val=y_val.astype(np.int32) if y_val is not None else None,
    )


def load_mnist_binary(
    digits: Tuple[int, int] = (0, 1),
    test_size: float = 0.2,
    val_size: float = 0.0,
    normalize: Union[str, bool] = "minmax",
    seed: int = 42,
    n_features: int = 16,
    max_samples: Optional[int] = 1000,
) -> DatasetInfo:
    """
    Load binary MNIST with PCA dimensionality reduction.

    Parameters
    ----------
    digits : tuple of int, default=(0, 1)
        Two digits to use for binary classification.
    test_size : float, default=0.2
        Fraction of data for testing.
    val_size : float, default=0.0
        Fraction of training data for validation.
    normalize : str or bool, default='minmax'
        Normalization method.
    seed : int, default=42
        Random seed.
    n_features : int, default=16
        Number of PCA components.
    max_samples : int, optional
        Maximum samples per class (for faster experiments).

    Returns
    -------
    DatasetInfo
        Dataset with train/test splits.

    Examples
    --------
    >>> data = load_mnist_binary(digits=(3, 8), n_features=8)
    >>> print(data.n_features, data.n_classes)
    8 2
    """
    try:
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data, mnist.target.astype(int)
    except Exception:
        # Fallback: generate synthetic data if MNIST unavailable
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=2000,
            n_features=64,
            n_informative=10,
            n_classes=2,
            random_state=seed,
        )
        digits = (0, 1)

    # Filter to selected digits
    d0, d1 = digits
    mask = (y == d0) | (y == d1)
    X, y = X[mask], y[mask]

    # Relabel to 0 and 1
    y = (y == d1).astype(int)

    # Limit samples
    if max_samples is not None:
        rng = np.random.RandomState(seed)
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        idx = np.concatenate([idx0[:max_samples], idx1[:max_samples]])
        X, y = X[idx], y[idx]

    result = _preprocess(X, y, test_size, val_size, normalize, seed, n_features)

    if val_size > 0:
        X_train, X_val, X_test, y_train, y_val, y_test = result
    else:
        X_train, X_test, y_train, y_test = result
        X_val, y_val = None, None

    return DatasetInfo(
        name=f"mnist_{d0}v{d1}",
        n_features=X_train.shape[1],
        n_classes=2,
        class_names=[str(d0), str(d1)],
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        X_val=X_val.astype(np.float32) if X_val is not None else None,
        y_val=y_val.astype(np.int32) if y_val is not None else None,
    )
