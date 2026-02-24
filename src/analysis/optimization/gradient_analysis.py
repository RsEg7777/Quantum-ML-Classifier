"""
Gradient Analysis
=================

Tools for diagnosing barren plateaus, gradient variance, and
parameter sensitivity in variational quantum circuits.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cirq
import numpy as np


def estimate_gradient_variance(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_samples: int = 200,
    epsilon: float = np.pi / 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Estimate the variance of the cost function gradient for each parameter.

    Large variance indicates the landscape is not flat (no barren plateau
    for that parameter). Variance decaying exponentially with qubit count
    is a signature of barren plateaus.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    n_params : int
        Number of parameters.
    n_samples : int, default=200
        Number of random parameter samples.
    epsilon : float, default=pi/2
        Finite difference step for parameter-shift.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Gradient variance for each parameter, shape (n_params,).
    """
    rng = np.random.RandomState(seed)
    gradients = np.zeros((n_samples, n_params))

    for s in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        for i in range(n_params):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon
            gradients[s, i] = (cost_fn(theta_plus) - cost_fn(theta_minus)) / (
                2 * np.sin(epsilon)
            )

    return np.var(gradients, axis=0)


def detect_barren_plateau(
    cost_fn: Callable[[np.ndarray], float],
    n_params: int,
    n_qubits_list: Optional[List[int]] = None,
    n_samples: int = 200,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Heuristic barren-plateau detection for a fixed circuit.

    Returns gradient statistics; exponentially small variance
    indicates a barren plateau.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    n_params : int
        Number of parameters.
    n_qubits_list : list of int, optional
        Not used for single-circuit detection; reserved for scaling studies.
    n_samples : int, default=200
        Number of random samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Keys: 'mean_grad_variance', 'max_grad_variance', 'min_grad_variance',
        'suspected_barren_plateau'.
    """
    variances = estimate_gradient_variance(
        cost_fn, n_params, n_samples, seed=seed
    )
    mean_var = float(np.mean(variances))

    return {
        "mean_grad_variance": mean_var,
        "max_grad_variance": float(np.max(variances)),
        "min_grad_variance": float(np.min(variances)),
        "per_param_variance": variances.tolist(),
        "suspected_barren_plateau": mean_var < 1e-5,
    }


def parameter_sensitivity(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    perturbation_scale: float = 0.1,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Measure cost sensitivity to perturbations of each parameter.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    theta : np.ndarray
        Base parameter values.
    perturbation_scale : float, default=0.1
        Standard deviation of Gaussian perturbation.
    n_samples : int, default=100
        Number of perturbation samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Sensitivity per parameter (std of cost change), shape (n_params,).
    """
    rng = np.random.RandomState(seed)
    base_cost = cost_fn(theta)
    n_params = len(theta)
    deltas = np.zeros((n_samples, n_params))

    for s in range(n_samples):
        for i in range(n_params):
            perturbed = theta.copy()
            perturbed[i] += rng.normal(0, perturbation_scale)
            deltas[s, i] = cost_fn(perturbed) - base_cost

    return np.std(deltas, axis=0)
