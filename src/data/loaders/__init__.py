"""
Data Loaders
============

Functions to load and preprocess datasets for quantum ML.
"""

from src.data.loaders.classification_datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_mnist_binary,
    DatasetInfo,
)

__all__ = [
    "load_iris",
    "load_wine",
    "load_breast_cancer",
    "load_mnist_binary",
    "DatasetInfo",
]
