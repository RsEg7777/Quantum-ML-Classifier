"""
Entanglement Module
===================

Entanglement strategies and measures for quantum circuits.
"""

from src.quantum.entanglement.entanglement_strategies import (
    EntanglementStrategy,
    get_entanglement_pairs,
    linear_entanglement,
    full_entanglement,
    circular_entanglement,
    star_entanglement,
)

__all__ = [
    "EntanglementStrategy",
    "get_entanglement_pairs",
    "linear_entanglement",
    "full_entanglement",
    "circular_entanglement",
    "star_entanglement",
]
