"""
Entanglement Strategies
=======================

Different topologies for entangling qubits in variational circuits.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import cirq


class EntanglementStrategy(Enum):
    """Enumeration of entanglement topologies."""

    LINEAR = "linear"
    FULL = "full"
    CIRCULAR = "circular"
    STAR = "star"
    REVERSE_LINEAR = "reverse_linear"
    PAIRWISE = "pairwise"


def linear_entanglement(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Linear (chain) entanglement: qubit i connects to i+1.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.

    Examples
    --------
    >>> linear_entanglement(4)
    [(0, 1), (1, 2), (2, 3)]
    """
    return [(i, i + 1) for i in range(n_qubits - 1)]


def full_entanglement(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Full (all-to-all) entanglement: every qubit connects to every other.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.

    Examples
    --------
    >>> full_entanglement(3)
    [(0, 1), (0, 2), (1, 2)]
    """
    pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            pairs.append((i, j))
    return pairs


def circular_entanglement(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Circular (ring) entanglement: linear plus last-to-first.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.

    Examples
    --------
    >>> circular_entanglement(4)
    [(0, 1), (1, 2), (2, 3), (3, 0)]
    """
    pairs = linear_entanglement(n_qubits)
    if n_qubits > 2:
        pairs.append((n_qubits - 1, 0))
    return pairs


def star_entanglement(n_qubits: int, center: int = 0) -> List[Tuple[int, int]]:
    """
    Star (hub-and-spoke) entanglement: all qubits connect to center.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    center : int, default=0
        Index of the center qubit.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.

    Examples
    --------
    >>> star_entanglement(4)
    [(0, 1), (0, 2), (0, 3)]
    """
    return [(center, i) for i in range(n_qubits) if i != center]


def reverse_linear_entanglement(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Reverse linear entanglement: qubit i connects to i-1.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.
    """
    return [(i, i - 1) for i in range(n_qubits - 1, 0, -1)]


def pairwise_entanglement(n_qubits: int) -> List[Tuple[int, int]]:
    """
    Pairwise entanglement: pairs of adjacent qubits (0-1, 2-3, ...).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.
    """
    return [(i, i + 1) for i in range(0, n_qubits - 1, 2)]


def get_entanglement_pairs(
    strategy: EntanglementStrategy,
    n_qubits: int,
    **kwargs,
) -> List[Tuple[int, int]]:
    """
    Get entanglement pairs for a given strategy.

    Parameters
    ----------
    strategy : EntanglementStrategy
        The entanglement topology.
    n_qubits : int
        Number of qubits.
    **kwargs
        Additional arguments for specific strategies.

    Returns
    -------
    list of tuple
        Pairs of (control, target) qubit indices.
    """
    if strategy == EntanglementStrategy.LINEAR:
        return linear_entanglement(n_qubits)
    elif strategy == EntanglementStrategy.FULL:
        return full_entanglement(n_qubits)
    elif strategy == EntanglementStrategy.CIRCULAR:
        return circular_entanglement(n_qubits)
    elif strategy == EntanglementStrategy.STAR:
        return star_entanglement(n_qubits, **kwargs)
    elif strategy == EntanglementStrategy.REVERSE_LINEAR:
        return reverse_linear_entanglement(n_qubits)
    elif strategy == EntanglementStrategy.PAIRWISE:
        return pairwise_entanglement(n_qubits)
    else:
        raise ValueError(f"Unknown entanglement strategy: {strategy}")


def build_entanglement_layer(
    qubits: List[cirq.GridQubit],
    strategy: EntanglementStrategy,
    gate: cirq.Gate = cirq.CNOT,
) -> cirq.Circuit:
    """
    Build an entanglement layer circuit.

    Parameters
    ----------
    qubits : list of cirq.GridQubit
        Qubit objects.
    strategy : EntanglementStrategy
        Entanglement topology.
    gate : cirq.Gate, default=CNOT
        Two-qubit entangling gate.

    Returns
    -------
    cirq.Circuit
        Entanglement layer circuit.
    """
    n_qubits = len(qubits)
    pairs = get_entanglement_pairs(strategy, n_qubits)

    circuit = cirq.Circuit()
    for i, j in pairs:
        circuit.append(gate(qubits[i], qubits[j]))

    return circuit
