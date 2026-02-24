"""Optimization analysis: gradient diagnostics and loss landscape."""

from src.analysis.optimization.gradient_analysis import (
    detect_barren_plateau,
    estimate_gradient_variance,
    parameter_sensitivity,
)
from src.analysis.optimization.loss_landscape import (
    LandscapeSlice,
    estimate_curvature,
    scan_1d,
    scan_2d,
)

__all__ = [
    "detect_barren_plateau",
    "estimate_gradient_variance",
    "parameter_sensitivity",
    "LandscapeSlice",
    "estimate_curvature",
    "scan_1d",
    "scan_2d",
]
