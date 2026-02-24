"""
Reproducibility Utilities
=========================

Seed management for reproducible experiments across TensorFlow, NumPy, and Cirq.

Usage
-----
>>> from src.utils.reproducibility import set_seed, get_reproducibility_info
>>> set_seed(42)
>>> info = get_reproducibility_info()
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, TensorFlow, and configures
    deterministic operations where possible.

    Parameters
    ----------
    seed : int, default=42
        Random seed value.
    deterministic : bool, default=True
        Whether to enable deterministic operations (may reduce performance).

    Notes
    -----
    - TensorFlow determinism may require CUDA_VISIBLE_DEVICES to be set
    - Some TF operations are inherently non-deterministic on GPU
    - Cirq uses NumPy's random state

    Examples
    --------
    >>> set_seed(42)
    >>> set_seed(123, deterministic=False)  # For faster training
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Set environment variables before importing TF
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # TensorFlow (lazy import to avoid forcing TF load)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)

        if deterministic:
            # Enable deterministic ops (TF 2.8+)
            try:
                tf.config.experimental.enable_op_determinism()
            except AttributeError:
                pass  # Older TF version

    except ImportError:
        pass

    # PyTorch (if available)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except ImportError:
        pass


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Get a NumPy RandomState instance.

    Parameters
    ----------
    seed : int, optional
        Seed for the random state. If None, uses a random seed.

    Returns
    -------
    np.random.RandomState
        Seeded random state.
    """
    return np.random.RandomState(seed)


def generate_seed_from_string(s: str) -> int:
    """
    Generate a deterministic seed from a string.

    Useful for generating seeds from experiment names.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    int
        Deterministic seed derived from the string.

    Examples
    --------
    >>> generate_seed_from_string("exp_1_depth_scaling")
    1234567890  # consistent across runs
    """
    hash_bytes = hashlib.md5(s.encode()).digest()
    return int.from_bytes(hash_bytes[:4], byteorder="big")


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Get environment information for reproducibility logging.

    Returns
    -------
    dict
        Dictionary containing version and environment information.

    Examples
    --------
    >>> info = get_reproducibility_info()
    >>> print(info["python_version"])
    '3.10.12'
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    # NumPy
    info["numpy_version"] = np.__version__

    # TensorFlow
    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
        info["gpu_available"] = len(tf.config.list_physical_devices("GPU")) > 0
        info["gpu_devices"] = [
            d.name for d in tf.config.list_physical_devices("GPU")
        ]
    except ImportError:
        info["tensorflow_version"] = "not installed"

    # TensorFlow Quantum
    try:
        import tensorflow_quantum as tfq

        info["tfq_version"] = tfq.__version__
    except ImportError:
        info["tfq_version"] = "not installed"

    # Cirq
    try:
        import cirq

        info["cirq_version"] = cirq.__version__
    except ImportError:
        info["cirq_version"] = "not installed"

    # SymPy
    try:
        import sympy

        info["sympy_version"] = sympy.__version__
    except ImportError:
        info["sympy_version"] = "not installed"

    return info


def save_reproducibility_info(
    path: Path,
    seed: int,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save reproducibility information to a JSON file.

    Parameters
    ----------
    path : Path
        Output file path.
    seed : int
        Random seed used.
    config : dict, optional
        Additional configuration to include.
    """
    info = get_reproducibility_info()
    info["seed"] = seed

    if config:
        info["config"] = config

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)


class ReproducibilityContext:
    """
    Context manager for reproducible code blocks.

    Saves and restores random states.

    Parameters
    ----------
    seed : int
        Seed for this context.

    Examples
    --------
    >>> with ReproducibilityContext(42) as ctx:
    ...     # All random operations here use seed 42
    ...     x = np.random.rand(10)
    >>> # Random states restored after context
    """

    def __init__(self, seed: int):
        self.seed = seed
        self._numpy_state = None
        self._python_state = None
        self._tf_seed = None

    def __enter__(self):
        # Save current states
        self._numpy_state = np.random.get_state()
        self._python_state = random.getstate()

        try:
            import tensorflow as tf

            # TF doesn't have get_seed, but we can track it
            self._tf_available = True
        except ImportError:
            self._tf_available = False

        # Set new seed
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous states
        np.random.set_state(self._numpy_state)
        random.setstate(self._python_state)
        return False


def hash_array(arr: np.ndarray) -> str:
    """
    Compute a hash of a NumPy array for verification.

    Parameters
    ----------
    arr : np.ndarray
        Array to hash.

    Returns
    -------
    str
        MD5 hash of the array.
    """
    return hashlib.md5(arr.tobytes()).hexdigest()
