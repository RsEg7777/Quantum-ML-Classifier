"""
SPSA Optimizer
==============

Simultaneous Perturbation Stochastic Approximation.
Gradient-free optimizer well-suited for noisy quantum circuits.

References
----------
Spall, "Multivariate stochastic approximation using a simultaneous
perturbation gradient approximation", IEEE TAC 37, 332 (1992).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class SPSAOptimizer:
    """
    SPSA optimizer for variational quantum circuits.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum iterations.
    a : float, default=0.1
        Step size scaling.
    c : float, default=0.1
        Perturbation size.
    A : float, default=10.0
        Stability constant.
    alpha : float, default=0.602
        Step size decay exponent.
    gamma : float, default=0.101
        Perturbation decay exponent.
    seed : int, optional
        Random seed.

    Examples
    --------
    >>> optimizer = SPSAOptimizer(maxiter=200)
    >>> result = optimizer.minimize(cost_fn, initial_params)
    """

    def __init__(
        self,
        maxiter: int = 100,
        a: float = 0.1,
        c: float = 0.1,
        A: float = 10.0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        seed: Optional[int] = None,
    ):
        self.maxiter = maxiter
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        callback: Optional[Callable] = None,
    ) -> dict:
        """
        Minimize the cost function.

        Parameters
        ----------
        cost_fn : callable
            Objective function f(params) -> scalar.
        x0 : np.ndarray
            Initial parameters.
        callback : callable, optional
            Called with (iteration, params, cost) each step.

        Returns
        -------
        dict
            {'x': optimal_params, 'fun': final_cost, 'nfev': n_evaluations, 'history': costs}
        """
        x = x0.copy()
        n = len(x)
        history = []
        nfev = 0

        for k in range(1, self.maxiter + 1):
            # Decaying step sizes
            ak = self.a / (k + self.A) ** self.alpha
            ck = self.c / k ** self.gamma

            # Random perturbation direction (Bernoulli ±1)
            delta = self.rng.choice([-1, 1], size=n).astype(float)

            # Evaluate perturbed points
            f_plus = cost_fn(x + ck * delta)
            f_minus = cost_fn(x - ck * delta)
            nfev += 2

            # Gradient estimate
            g_hat = (f_plus - f_minus) / (2 * ck * delta)

            # Update
            x = x - ak * g_hat

            cost = (f_plus + f_minus) / 2
            history.append(cost)

            if callback:
                callback(k, x, cost)

        final_cost = cost_fn(x)
        nfev += 1

        return {
            "x": x,
            "fun": final_cost,
            "nfev": nfev,
            "history": history,
        }
