"""
Pauli Measurements
==================

Measurement operators in the Pauli basis (X, Y, Z).
Provides utilities for constructing expectation value operators
compatible with TensorFlow Quantum.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import cirq
import numpy as np


def z_measurement(
    qubits: Sequence[cirq.GridQubit],
    qubit_indices: Optional[Sequence[int]] = None,
) -> List[cirq.PauliString]:
    """
    Z-basis measurement operators.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Available qubits.
    qubit_indices : sequence of int, optional
        Which qubits to measure. If None, measures all.

    Returns
    -------
    list of PauliString
        Z operators on specified qubits.
    """
    if qubit_indices is not None:
        return [cirq.Z(qubits[i]) for i in qubit_indices]
    return [cirq.Z(q) for q in qubits]


def x_measurement(
    qubits: Sequence[cirq.GridQubit],
    qubit_indices: Optional[Sequence[int]] = None,
) -> List[cirq.PauliString]:
    """
    X-basis measurement operators.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Available qubits.
    qubit_indices : sequence of int, optional
        Which qubits to measure.

    Returns
    -------
    list of PauliString
        X operators on specified qubits.
    """
    if qubit_indices is not None:
        return [cirq.X(qubits[i]) for i in qubit_indices]
    return [cirq.X(q) for q in qubits]


def y_measurement(
    qubits: Sequence[cirq.GridQubit],
    qubit_indices: Optional[Sequence[int]] = None,
) -> List[cirq.PauliString]:
    """
    Y-basis measurement operators.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Available qubits.
    qubit_indices : sequence of int, optional
        Which qubits to measure.

    Returns
    -------
    list of PauliString
        Y operators on specified qubits.
    """
    if qubit_indices is not None:
        return [cirq.Y(qubits[i]) for i in qubit_indices]
    return [cirq.Y(q) for q in qubits]


def zz_correlator(
    qubits: Sequence[cirq.GridQubit],
    pairs: Optional[List[tuple]] = None,
) -> List[cirq.PauliString]:
    """
    ZZ correlator measurements: ⟨Z_i Z_j⟩.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Available qubits.
    pairs : list of tuple, optional
        Qubit pairs (i, j). If None, uses nearest-neighbor.

    Returns
    -------
    list of PauliString
        ZZ correlator operators.
    """
    if pairs is None:
        pairs = [(i, i + 1) for i in range(len(qubits) - 1)]

    operators = []
    for i, j in pairs:
        operators.append(cirq.Z(qubits[i]) * cirq.Z(qubits[j]))
    return operators


def multi_basis_measurement(
    qubits: Sequence[cirq.GridQubit],
    bases: str = "Z",
) -> List[cirq.PauliString]:
    """
    Measurement in specified Pauli bases.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Target qubits.
    bases : str, default='Z'
        Measurement bases: 'Z', 'X', 'Y', 'XY', 'XYZ', etc.

    Returns
    -------
    list of PauliString
        Measurement operators in all specified bases.
    """
    gate_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    operators = []

    for basis in bases.upper():
        if basis in gate_map:
            for qubit in qubits:
                operators.append(gate_map[basis](qubit))

    return operators


def weighted_pauli_sum(
    qubits: Sequence[cirq.GridQubit],
    weights: Optional[Sequence[float]] = None,
    basis: str = "Z",
) -> cirq.PauliSum:
    """
    Weighted sum of Pauli operators: H = Σ_i w_i P_i.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Target qubits.
    weights : sequence of float, optional
        Weights for each qubit. Defaults to uniform.
    basis : str, default='Z'
        Pauli basis ('X', 'Y', 'Z').

    Returns
    -------
    cirq.PauliSum
        Weighted Pauli sum operator.
    """
    gate_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    pauli = gate_map[basis.upper()]

    if weights is None:
        weights = [1.0 / len(qubits)] * len(qubits)

    terms = []
    for qubit, weight in zip(qubits, weights):
        terms.append(weight * pauli(qubit))

    return sum(terms)


def get_measurement_operators(
    qubits: Sequence[cirq.GridQubit],
    measurement_type: str = "z",
    **kwargs,
) -> List:
    """
    Factory function for measurement operators.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Available qubits.
    measurement_type : str, default='z'
        Type: 'z', 'x', 'y', 'zz', 'xyz', 'weighted_z'.

    Returns
    -------
    list
        Measurement operators.
    """
    if measurement_type == "z":
        return z_measurement(qubits, **kwargs)
    elif measurement_type == "x":
        return x_measurement(qubits, **kwargs)
    elif measurement_type == "y":
        return y_measurement(qubits, **kwargs)
    elif measurement_type == "zz":
        return zz_correlator(qubits, **kwargs)
    elif measurement_type in ("xyz", "xy", "xz", "yz"):
        return multi_basis_measurement(qubits, bases=measurement_type)
    elif measurement_type == "weighted_z":
        return [weighted_pauli_sum(qubits, basis="Z", **kwargs)]
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")


def measurement_basis_change(
    qubit: cirq.GridQubit,
    basis: str,
) -> cirq.Circuit:
    """
    Get basis-change circuit for measuring in a non-Z basis.

    Parameters
    ----------
    qubit : GridQubit
        Target qubit.
    basis : str
        Target basis: 'X', 'Y', or 'Z'.

    Returns
    -------
    cirq.Circuit
        Basis-change circuit (apply before Z measurement).
    """
    circuit = cirq.Circuit()

    if basis.upper() == "X":
        circuit.append(cirq.H(qubit))
    elif basis.upper() == "Y":
        circuit.append(cirq.rx(-np.pi / 2).on(qubit))
    # Z basis requires no change

    return circuit
