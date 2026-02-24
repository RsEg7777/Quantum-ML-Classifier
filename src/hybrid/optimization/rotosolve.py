"""
Rotosolve Optimizer
===================

Analytical optimization for rotation gates.
For a cost function C(θ) = A sin(θ - B) + C, the optimal θ
can be found analytically from two evaluations.

References
----------
Ostaszewski et al., "Structure optimization for parameterized
quantum circuits", Quantum 5, 391 (2021).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class RotosolveOptimizer:
    """
    Rotosolve: analytical per-parameter optimization.

    Parameters
    ----------
    maxiter : int, default=50
        Maximum number of full sweeps.
    tol : float, default=1e-6
        Convergence tolerance on cost change.

    Examples
    --------
    >>> optimizer = RotosolveOptimizer(maxiter=100)
    >>> result = optimizer.minimize(cost_fn, initial_params)
    """

    def __init__(self, maxiter: int = 50, tol: float = 1e-6):
        self.maxiter = maxiter
        self.tol = tol

    def _optimize_single(
        self,
        cost_fn: Callable,
        params: np.ndarray,
        idx: int,
    ) -> float:
        """
        Analytically optimize a single parameter.

        Uses the sinusoidal property of quantum cost landscapes:
        C(θ_i) = A sin(θ_i - B) + C

        Three evaluations at θ, θ+π/2, θ-π/2 determine A, B, C.
        """
        original = params[idx]

        # Evaluate at three points
        params[idx] = 0.0
        c0 = cost_fn(params)

        params[idx] = np.pi / 2
        c_plus = cost_fn(params)

        params[idx] = -np.pi / 2
        c_minus = cost_fn(params)

        # Reconstruct sinusoidal: C(θ) = a*sin(θ) + b*cos(θ) + c
        # From: c0 = b + c, c+ = a + c, c- = -a + c
        a = (c_plus - c_minus) / 2
        c = (c_plus + c_minus) / 2
        b = c0 - c

        # Optimal angle: θ* = -arctan2(a, b)
        theta_opt = -np.arctan2(a, b)

        params[idx] = theta_opt
        return theta_opt

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Minimize via coordinate-wise analytical optimization.

        Parameters
        ----------
        cost_fn : callable
            Cost function f(params) -> scalar.
        x0 : np.ndarray
            Initial parameters.
        callback : callable, optional
            Called with (iteration, params, cost) per sweep.

        Returns
        -------
        dict
            Optimization result.
        """
        params = x0.copy()
        n = len(params)
        history = []
        nfev = 0
        prev_cost = cost_fn(params)
        nfev += 1

        for iteration in range(self.maxiter):
            # Sweep through all parameters
            for i in range(n):
                self._optimize_single(cost_fn, params, i)
                nfev += 3  # three evaluations per parameter

            cost = cost_fn(params)
            nfev += 1
            history.append(cost)

            if callback:
                callback(iteration, params, cost)

            # Check convergence
            if abs(prev_cost - cost) < self.tol:
                break
            prev_cost = cost

        return {
            "x": params,
            "fun": cost,
            "nfev": nfev,
            "nit": iteration + 1,
            "history": history,
        }
