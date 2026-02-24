"""
Circuit Expressibility
======================

Measures how uniformly a parameterised circuit can explore the Hilbert space,
following the framework of Sim et al. (2019).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import cirq
import numpy as np


def haar_probability(fidelity: float, n_qubits: int) -> float:
    """
    PDF of fidelity for Haar-random states.

    P(F) = (2^n - 1) * (1 - F)^(2^n - 2)

    Parameters
    ----------
    fidelity : float
        State fidelity in [0, 1].
    n_qubits : int
        Number of qubits.

    Returns
    -------
    float
        Probability density.
    """
    N = 2 ** n_qubits
    return (N - 1) * (1 - fidelity) ** (N - 2)


def compute_fidelities(
    circuit_fn: Callable[[np.ndarray], cirq.Circuit],
    n_qubits: int,
    n_params: int,
    n_samples: int = 5000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Sample pairwise state fidelities from a parameterised circuit.

    Parameters
    ----------
    circuit_fn : callable
        Function mapping parameter vector -> cirq.Circuit.
    n_qubits : int
        Number of qubits.
    n_params : int
        Dimensionality of the parameter vector.
    n_samples : int, default=5000
        Number of parameter pairs to sample.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Array of fidelity values, shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    simulator = cirq.Simulator()
    fidelities = np.zeros(n_samples)

    for i in range(n_samples):
        theta1 = rng.uniform(0, 2 * np.pi, size=n_params)
        theta2 = rng.uniform(0, 2 * np.pi, size=n_params)

        circ1 = circuit_fn(theta1)
        circ2 = circuit_fn(theta2)

        sv1 = simulator.simulate(circ1).final_state_vector
        sv2 = simulator.simulate(circ2).final_state_vector

        fidelities[i] = np.abs(np.vdot(sv1, sv2)) ** 2

    return fidelities


def expressibility(
    circuit_fn: Callable[[np.ndarray], cirq.Circuit],
    n_qubits: int,
    n_params: int,
    n_samples: int = 5000,
    n_bins: int = 75,
    seed: Optional[int] = None,
) -> float:
    """
    Compute circuit expressibility as KL divergence from the Haar distribution.

    Lower values indicate higher expressibility (closer to Haar-random).

    Parameters
    ----------
    circuit_fn : callable
        Maps parameter vector -> cirq.Circuit.
    n_qubits : int
        Number of qubits.
    n_params : int
        Number of circuit parameters.
    n_samples : int, default=5000
        Number of fidelity samples.
    n_bins : int, default=75
        Number of histogram bins.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        KL divergence D_KL(P_circuit || P_Haar). Lower = more expressive.
    """
    fidelities = compute_fidelities(
        circuit_fn, n_qubits, n_params, n_samples, seed
    )

    # Empirical fidelity distribution
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)

    # Haar distribution
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    haar_dist = np.array([haar_probability(f, n_qubits) for f in bin_centres])

    # Avoid log(0) with small epsilon
    eps = 1e-10
    hist = hist + eps
    haar_dist = haar_dist + eps

    # Normalise to proper distributions
    hist = hist / hist.sum()
    haar_dist = haar_dist / haar_dist.sum()

    # KL divergence
    kl = np.sum(hist * np.log(hist / haar_dist))
    return float(kl)


def entangling_capability(
    circuit_fn: Callable[[np.ndarray], cirq.Circuit],
    n_qubits: int,
    n_params: int,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> float:
    """
    Estimate the entangling capability via Meyer-Wallach measure.

    Parameters
    ----------
    circuit_fn : callable
        Maps parameter vector -> cirq.Circuit.
    n_qubits : int
        Number of qubits.
    n_params : int
        Number of parameters.
    n_samples : int, default=1000
        Number of random parameter samples.
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Average Meyer-Wallach entanglement Q in [0, 1].
    """
    rng = np.random.RandomState(seed)
    simulator = cirq.Simulator()
    q_values = []

    for _ in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi, size=n_params)
        circ = circuit_fn(theta)
        sv = simulator.simulate(circ).final_state_vector.reshape(
            *([2] * n_qubits)
        )
        q_values.append(_meyer_wallach(sv, n_qubits))

    return float(np.mean(q_values))


def _meyer_wallach(state_tensor: np.ndarray, n_qubits: int) -> float:
    """Compute the Meyer-Wallach entanglement measure for a pure state."""
    q = 0.0
    for k in range(n_qubits):
        # Partial trace over qubit k -> reduced density matrix
        rho_k = _partial_trace_single(state_tensor, k, n_qubits)
        purity = np.real(np.trace(rho_k @ rho_k))
        q += 1 - purity
    return 2 * q / n_qubits


def _partial_trace_single(
    state_tensor: np.ndarray, qubit_idx: int, n_qubits: int
) -> np.ndarray:
    """Compute reduced density matrix for a single qubit."""
    # Reshape state to (2, 2^(n-1)) with qubit_idx as the first axis
    axes = list(range(n_qubits))
    axes.remove(qubit_idx)
    axes = [qubit_idx] + axes
    psi = np.transpose(state_tensor, axes).reshape(2, -1)
    rho = psi @ psi.conj().T
    return rho
