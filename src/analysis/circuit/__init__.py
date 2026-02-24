"""Circuit analysis: expressibility and resource estimation."""

from src.analysis.circuit.expressibility import (
    entangling_capability,
    expressibility,
)
from src.analysis.circuit.resource_estimation import (
    ResourceProfile,
    compare_circuits,
    estimate_execution_time,
    estimate_resources,
)

__all__ = [
    "entangling_capability",
    "expressibility",
    "ResourceProfile",
    "compare_circuits",
    "estimate_execution_time",
    "estimate_resources",
]
