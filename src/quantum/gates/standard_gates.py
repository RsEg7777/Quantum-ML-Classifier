"""
Standard Quantum Gates
======================

Unified interface for standard quantum gates built on Cirq.
Provides convenient access to common single-qubit and multi-qubit gates.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import cirq
import numpy as np
import sympy


# =============================================================================
# Single-Qubit Gates
# =============================================================================

def hadamard(qubit: cirq.GridQubit) -> cirq.Operation:
    """Hadamard gate: creates equal superposition."""
    return cirq.H(qubit)


def pauli_x(qubit: cirq.GridQubit) -> cirq.Operation:
    """Pauli-X (NOT) gate: bit flip."""
    return cirq.X(qubit)


def pauli_y(qubit: cirq.GridQubit) -> cirq.Operation:
    """Pauli-Y gate."""
    return cirq.Y(qubit)


def pauli_z(qubit: cirq.GridQubit) -> cirq.Operation:
    """Pauli-Z gate: phase flip."""
    return cirq.Z(qubit)


def s_gate(qubit: cirq.GridQubit) -> cirq.Operation:
    """S gate (√Z): π/2 phase rotation."""
    return cirq.S(qubit)


def t_gate(qubit: cirq.GridQubit) -> cirq.Operation:
    """T gate (√S): π/4 phase rotation."""
    return cirq.T(qubit)


def rx(angle: Union[float, sympy.Symbol], qubit: cirq.GridQubit) -> cirq.Operation:
    """X-axis rotation gate."""
    return cirq.rx(angle).on(qubit)


def ry(angle: Union[float, sympy.Symbol], qubit: cirq.GridQubit) -> cirq.Operation:
    """Y-axis rotation gate."""
    return cirq.ry(angle).on(qubit)


def rz(angle: Union[float, sympy.Symbol], qubit: cirq.GridQubit) -> cirq.Operation:
    """Z-axis rotation gate."""
    return cirq.rz(angle).on(qubit)


def phase_gate(
    angle: Union[float, sympy.Symbol],
    qubit: cirq.GridQubit,
) -> cirq.Operation:
    """General phase gate: R_z(θ) up to global phase."""
    return cirq.rz(angle).on(qubit)


# =============================================================================
# Two-Qubit Gates
# =============================================================================

def cnot(control: cirq.GridQubit, target: cirq.GridQubit) -> cirq.Operation:
    """Controlled-NOT (CX) gate."""
    return cirq.CNOT(control, target)


def cz(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> cirq.Operation:
    """Controlled-Z gate."""
    return cirq.CZ(qubit1, qubit2)


def swap(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> cirq.Operation:
    """SWAP gate."""
    return cirq.SWAP(qubit1, qubit2)


def iswap(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> cirq.Operation:
    """iSWAP gate."""
    return cirq.ISWAP(qubit1, qubit2)


def sqrt_iswap(qubit1: cirq.GridQubit, qubit2: cirq.GridQubit) -> cirq.Operation:
    """√iSWAP gate (native on some hardware)."""
    return cirq.SQRT_ISWAP(qubit1, qubit2)


# =============================================================================
# Three-Qubit Gates
# =============================================================================

def toffoli(
    control1: cirq.GridQubit,
    control2: cirq.GridQubit,
    target: cirq.GridQubit,
) -> cirq.Operation:
    """Toffoli (CCX) gate: doubly-controlled NOT."""
    return cirq.TOFFOLI(control1, control2, target)


def fredkin(
    control: cirq.GridQubit,
    target1: cirq.GridQubit,
    target2: cirq.GridQubit,
) -> cirq.Operation:
    """Fredkin (CSWAP) gate: controlled SWAP."""
    return cirq.FREDKIN(control, target1, target2)


# =============================================================================
# Gate Collections
# =============================================================================

def rotation_layer(
    qubits: Sequence[cirq.GridQubit],
    angles: Sequence[Union[float, sympy.Symbol]],
    axis: str = "Y",
) -> List[cirq.Operation]:
    """
    Apply rotation gates to all qubits.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Qubits to rotate.
    angles : sequence of float or Symbol
        Rotation angles (one per qubit).
    axis : str, default='Y'
        Rotation axis: 'X', 'Y', or 'Z'.

    Returns
    -------
    list of Operation
        Rotation operations.
    """
    gate_fn = {"X": cirq.rx, "Y": cirq.ry, "Z": cirq.rz}[axis.upper()]
    ops = []
    for qubit, angle in zip(qubits, angles):
        ops.append(gate_fn(angle).on(qubit))
    return ops


def entangling_layer(
    qubits: Sequence[cirq.GridQubit],
    gate: str = "CNOT",
    topology: str = "linear",
) -> List[cirq.Operation]:
    """
    Apply entangling gates between qubit pairs.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Qubits to entangle.
    gate : str, default='CNOT'
        Entangling gate: 'CNOT', 'CZ', 'SWAP'.
    topology : str, default='linear'
        Connectivity: 'linear', 'circular', 'full'.

    Returns
    -------
    list of Operation
        Entangling operations.
    """
    gate_fn = {
        "CNOT": cirq.CNOT,
        "CZ": cirq.CZ,
        "SWAP": cirq.SWAP,
    }[gate.upper()]

    n = len(qubits)
    pairs = []

    if topology == "linear":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif topology == "circular":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    elif topology == "full":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    ops = []
    for i, j in pairs:
        ops.append(gate_fn(qubits[i], qubits[j]))
    return ops


def hadamard_layer(qubits: Sequence[cirq.GridQubit]) -> List[cirq.Operation]:
    """Apply Hadamard to all qubits."""
    return [cirq.H(q) for q in qubits]


# =============================================================================
# Gate Registry
# =============================================================================

SINGLE_QUBIT_GATES = {
    "H": cirq.H,
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
    "S": cirq.S,
    "T": cirq.T,
}

ROTATION_GATES = {
    "RX": cirq.rx,
    "RY": cirq.ry,
    "RZ": cirq.rz,
}

TWO_QUBIT_GATES = {
    "CNOT": cirq.CNOT,
    "CX": cirq.CNOT,
    "CZ": cirq.CZ,
    "SWAP": cirq.SWAP,
    "ISWAP": cirq.ISWAP,
}

THREE_QUBIT_GATES = {
    "TOFFOLI": cirq.TOFFOLI,
    "CCX": cirq.TOFFOLI,
    "FREDKIN": cirq.FREDKIN,
    "CSWAP": cirq.FREDKIN,
}


def get_gate(name: str):
    """
    Get a gate by name.

    Parameters
    ----------
    name : str
        Gate name (case-insensitive).

    Returns
    -------
    Gate or gate function.

    Raises
    ------
    ValueError
        If gate name is not recognized.
    """
    name = name.upper()
    all_gates = {**SINGLE_QUBIT_GATES, **ROTATION_GATES, **TWO_QUBIT_GATES, **THREE_QUBIT_GATES}

    if name in all_gates:
        return all_gates[name]
    raise ValueError(f"Unknown gate: {name}. Available: {list(all_gates.keys())}")
