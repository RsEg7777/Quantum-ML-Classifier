"""Classical ML baselines for comparison."""

from src.classical.baselines.ensemble_methods import ClassicalEnsembleBaseline
from src.classical.baselines.gaussian_processes import GPBaseline
from src.classical.baselines.svm import SVMBaseline

__all__ = ["ClassicalEnsembleBaseline", "SVMBaseline", "GPBaseline"]
