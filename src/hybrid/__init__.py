"""
Hybrid Package
==============

Hybrid quantum-classical models, training loops, and optimization.

Submodules
----------
models
    Hybrid model architectures combining quantum circuits with classical networks.
training
    Training loops, validation, and distributed training.
optimization
    Quantum-aware optimizers (QNG, Rotosolve, SPSA).
nas
    Neural Architecture Search for hybrid models.
"""

__all__ = [
    "models",
    "training",
    "optimization",
    "nas",
]
