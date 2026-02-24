"""
Loss Landscape Analysis
========================

Visualisation-ready loss landscape slicing and curvature estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class LandscapeSlice:
    """Result of a 1D or 2D loss landscape scan."""

    alphas: np.ndarray
    betas: Optional[np.ndarray]
    losses: np.ndarray
    direction1: np.ndarray
    direction2: Optional[np.ndarray]
    center: np.ndarray


def random_direction(n_params: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate a normalised random direction vector."""
    rng = np.random.RandomState(seed)
    d = rng.randn(n_params)
    return d / np.linalg.norm(d)


def scan_1d(
    cost_fn: Callable[[np.ndarray], float],
    center: np.ndarray,
    direction: Optional[np.ndarray] = None,
    n_points: int = 51,
    range_limit: float = np.pi,
    seed: Optional[int] = None,
) -> LandscapeSlice:
    """
    1D slice of the loss landscape along a direction.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    center : np.ndarray
        Center point in parameter space.
    direction : np.ndarray, optional
        Direction vector. Random if None.
    n_points : int, default=51
        Number of evaluation points.
    range_limit : float, default=pi
        Half-width of the scan range.
    seed : int, optional
        Random seed (used if direction is None).

    Returns
    -------
    LandscapeSlice
    """
    if direction is None:
        direction = random_direction(len(center), seed)

    alphas = np.linspace(-range_limit, range_limit, n_points)
    losses = np.array([cost_fn(center + a * direction) for a in alphas])

    return LandscapeSlice(
        alphas=alphas,
        betas=None,
        losses=losses,
        direction1=direction,
        direction2=None,
        center=center,
    )


def scan_2d(
    cost_fn: Callable[[np.ndarray], float],
    center: np.ndarray,
    direction1: Optional[np.ndarray] = None,
    direction2: Optional[np.ndarray] = None,
    n_points: int = 25,
    range_limit: float = np.pi,
    seed: Optional[int] = None,
) -> LandscapeSlice:
    """
    2D slice of the loss landscape along two directions.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    center : np.ndarray
        Center point.
    direction1, direction2 : np.ndarray, optional
        Orthogonal direction vectors. Random if None.
    n_points : int, default=25
        Grid points per axis.
    range_limit : float, default=pi
        Half-width of the scan range.
    seed : int, optional
        Random seed.

    Returns
    -------
    LandscapeSlice
        losses has shape (n_points, n_points).
    """
    rng = np.random.RandomState(seed)
    n_params = len(center)

    if direction1 is None:
        direction1 = random_direction(n_params, seed=rng.randint(0, 2**31))
    if direction2 is None:
        d2 = random_direction(n_params, seed=rng.randint(0, 2**31))
        # Gram-Schmidt orthogonalisation
        d2 = d2 - np.dot(d2, direction1) * direction1
        d2 = d2 / np.linalg.norm(d2)
        direction2 = d2

    alphas = np.linspace(-range_limit, range_limit, n_points)
    betas = np.linspace(-range_limit, range_limit, n_points)
    losses = np.zeros((n_points, n_points))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            losses[i, j] = cost_fn(center + a * direction1 + b * direction2)

    return LandscapeSlice(
        alphas=alphas,
        betas=betas,
        losses=losses,
        direction1=direction1,
        direction2=direction2,
        center=center,
    )


def estimate_curvature(
    cost_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Estimate the diagonal of the Hessian via finite differences.

    Parameters
    ----------
    cost_fn : callable
        Maps parameter vector -> scalar cost.
    theta : np.ndarray
        Point at which to estimate curvature.
    epsilon : float, default=0.01
        Finite difference step.

    Returns
    -------
    np.ndarray
        Approximate diagonal Hessian entries, shape (n_params,).
    """
    n = len(theta)
    f0 = cost_fn(theta)
    hess_diag = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = epsilon
        hess_diag[i] = (
            cost_fn(theta + e_i) - 2 * f0 + cost_fn(theta - e_i)
        ) / epsilon**2

    return hess_diag
