"""
Quantum-Classical Hybrid ML System
===================================

A state-of-the-art hybrid quantum-classical machine learning framework
built on TensorFlow Quantum.

Subpackages
-----------
quantum
    Quantum circuits, encodings, kernels, gates, and measurements.
classical
    Classical neural networks, baselines, and preprocessing.
hybrid
    Hybrid quantum-classical models, training loops, and optimization.
noise
    Noise models and error mitigation techniques.
analysis
    Quantum state analysis, circuit analysis, and performance metrics.
visualization
    Plotting utilities and interactive dashboards.
data
    Dataset loaders, preprocessing, and augmentation.
utils
    Configuration, logging, checkpointing, and utilities.
"""

__version__ = "0.1.0"
__author__ = "Quantum ML Research Team"

from src.utils.config import get_config, load_config
from src.utils.logging import get_logger, setup_logging
from src.utils.reproducibility import set_seed

__all__ = [
    "__version__",
    "get_config",
    "load_config",
    "get_logger",
    "setup_logging",
    "set_seed",
]
