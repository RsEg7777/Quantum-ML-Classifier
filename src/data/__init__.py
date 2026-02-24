"""
Data Package
============

Dataset loaders, preprocessing, and augmentation utilities.
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
