"""
Zero-Noise Extrapolation (ZNE)
==============================

Mitigates noise by running circuits at multiple noise levels
and extrapolating to the zero-noise limit.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cirq
import numpy as np


def fold_gates_at_random(
    circuit: cirq.Circuit,
    scale_factor: float,
    seed: Optional[int] = None,
) -> cirq.Circuit:
    """
    Scale circuit noise by unitary folding: G -> G G† G.

    Parameters
    ----------
    circuit : cirq.Circuit
        Original circuit.
    scale_factor : float
        Noise scale factor >= 1. Must satisfy (scale_factor - 1) % 2 ≈ 0
        for full folding, otherwise partial folding is applied.
    seed : int, optional
        Random seed for selecting gates in partial folding.

    Returns
    -------
    cirq.Circuit
        Folded circuit with effectively higher noise.
    """
    if scale_factor < 1.0:
        raise ValueError("scale_factor must be >= 1.0")
    if np.isclose(scale_factor, 1.0):
        return circuit.copy()

    rng = np.random.RandomState(seed)
    all_ops = list(circuit.all_operations())
    n_ops = len(all_ops)

    # Number of full folds (each fold triples the gate count contribution)
    n_full_folds = int((scale_factor - 1) // 2)
    remainder = scale_factor - 1 - 2 * n_full_folds

    # Full circuit folds
    folded = cirq.Circuit()
    for _ in range(n_full_folds):
        folded = circuit + cirq.inverse(circuit) + circuit
        circuit = folded

    if n_full_folds == 0:
        folded = circuit.copy()

    # Partial fold: randomly fold a subset of gates
    if remainder > 0:
        n_gates_to_fold = int(round(remainder / 2 * n_ops))
        n_gates_to_fold = min(n_gates_to_fold, n_ops)
        indices_to_fold = set(rng.choice(n_ops, size=n_gates_to_fold, replace=False))

        partial = cirq.Circuit()
        for i, op in enumerate(folded.all_operations()):
            partial.append(op)
            if i in indices_to_fold:
                partial.append(cirq.inverse(cirq.Circuit([op])))
                partial.append(op)
        folded = partial

    return folded


def richardson_extrapolate(
    scale_factors: Sequence[float],
    expectation_values: Sequence[float],
) -> float:
    """
    Richardson extrapolation to zero-noise limit.

    Fits a polynomial of degree len(scale_factors)-1 and evaluates at 0.

    Parameters
    ----------
    scale_factors : sequence of float
        Noise scale factors used.
    expectation_values : sequence of float
        Corresponding expectation values.

    Returns
    -------
    float
        Extrapolated zero-noise expectation value.
    """
    scale_factors = np.asarray(scale_factors, dtype=float)
    expectation_values = np.asarray(expectation_values, dtype=float)
    degree = len(scale_factors) - 1
    coeffs = np.polyfit(scale_factors, expectation_values, degree)
    return float(np.polyval(coeffs, 0.0))


def linear_extrapolate(
    scale_factors: Sequence[float],
    expectation_values: Sequence[float],
) -> float:
    """Linear (first-order) extrapolation to zero noise."""
    coeffs = np.polyfit(scale_factors, expectation_values, 1)
    return float(np.polyval(coeffs, 0.0))


def exponential_extrapolate(
    scale_factors: Sequence[float],
    expectation_values: Sequence[float],
) -> float:
    """
    Exponential extrapolation: E(λ) = a * exp(b * λ) + c.

    Falls back to Richardson if the fit fails.
    """
    from scipy.optimize import curve_fit

    def model(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        popt, _ = curve_fit(
            model,
            np.array(scale_factors),
            np.array(expectation_values),
            p0=[1.0, -0.5, 0.0],
            maxfev=5000,
        )
        return float(model(0.0, *popt))
    except RuntimeError:
        return richardson_extrapolate(scale_factors, expectation_values)


class ZeroNoiseExtrapolator:
    """
    Full ZNE pipeline.

    Parameters
    ----------
    scale_factors : list of float, default=[1.0, 2.0, 3.0]
        Noise amplification factors.
    extrapolation : str, default='richardson'
        Extrapolation method: 'linear', 'richardson', or 'exponential'.
    fold_method : str, default='random'
        Gate folding method (currently only 'random' supported).
    seed : int, optional
        Random seed.
    """

    EXTRAPOLATORS = {
        "linear": linear_extrapolate,
        "richardson": richardson_extrapolate,
        "exponential": exponential_extrapolate,
    }

    def __init__(
        self,
        scale_factors: Optional[List[float]] = None,
        extrapolation: str = "richardson",
        fold_method: str = "random",
        seed: Optional[int] = None,
    ):
        self.scale_factors = scale_factors or [1.0, 2.0, 3.0]
        self.extrapolation = extrapolation
        self.fold_method = fold_method
        self.seed = seed

        if extrapolation not in self.EXTRAPOLATORS:
            raise ValueError(
                f"Unknown extrapolation: {extrapolation}. "
                f"Available: {list(self.EXTRAPOLATORS.keys())}"
            )

    def generate_folded_circuits(
        self, circuit: cirq.Circuit
    ) -> List[cirq.Circuit]:
        """Generate noise-scaled circuits for each scale factor."""
        return [
            fold_gates_at_random(circuit, sf, seed=self.seed)
            for sf in self.scale_factors
        ]

    def extrapolate(self, expectation_values: Sequence[float]) -> float:
        """Extrapolate to the zero-noise limit."""
        extrap_fn = self.EXTRAPOLATORS[self.extrapolation]
        return extrap_fn(self.scale_factors, expectation_values)

    def mitigate(
        self,
        circuit: cirq.Circuit,
        executor: Callable[[cirq.Circuit], float],
    ) -> float:
        """
        End-to-end ZNE mitigation.

        Parameters
        ----------
        circuit : cirq.Circuit
            Original circuit.
        executor : callable
            Function that takes a circuit and returns an expectation value.

        Returns
        -------
        float
            Mitigated expectation value.
        """
        folded_circuits = self.generate_folded_circuits(circuit)
        expectations = [executor(c) for c in folded_circuits]
        return self.extrapolate(expectations)
