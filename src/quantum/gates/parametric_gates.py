"""
Parametric Quantum Gates
========================

Custom parametric gates including controlled rotations,
multi-qubit rotation gates, and the general U3 gate.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import cirq
import numpy as np
import sympy


# =============================================================================
# Controlled Rotation Gates
# =============================================================================

def crx(
    angle: Union[float, sympy.Symbol],
    control: cirq.GridQubit,
    target: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    Controlled-RX gate.

    Decomposition: CNOT structure with half-angle rotations.
    """
    half = angle / 2
    return [
        cirq.rz(np.pi / 2).on(target),
        cirq.CNOT(control, target),
        cirq.ry(-half).on(target),
        cirq.CNOT(control, target),
        cirq.ry(half).on(target),
        cirq.rz(-np.pi / 2).on(target),
    ]


def cry(
    angle: Union[float, sympy.Symbol],
    control: cirq.GridQubit,
    target: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    Controlled-RY gate.

    Decomposition: half-angle RY with CNOT.
    """
    half = angle / 2
    return [
        cirq.ry(half).on(target),
        cirq.CNOT(control, target),
        cirq.ry(-half).on(target),
        cirq.CNOT(control, target),
    ]


def crz(
    angle: Union[float, sympy.Symbol],
    control: cirq.GridQubit,
    target: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    Controlled-RZ gate.

    Decomposition: half-angle RZ with CNOT.
    """
    half = angle / 2
    return [
        cirq.rz(half).on(target),
        cirq.CNOT(control, target),
        cirq.rz(-half).on(target),
        cirq.CNOT(control, target),
    ]


# =============================================================================
# Two-Qubit Rotation Gates (Pauli-Pauli)
# =============================================================================

def rxx(
    angle: Union[float, sympy.Symbol],
    qubit1: cirq.GridQubit,
    qubit2: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    XX rotation: exp(-i θ/2 X⊗X).

    Decomposition: H - CNOT - RZ - CNOT - H.
    """
    return [
        cirq.H(qubit1),
        cirq.H(qubit2),
        cirq.CNOT(qubit1, qubit2),
        cirq.rz(angle).on(qubit2),
        cirq.CNOT(qubit1, qubit2),
        cirq.H(qubit1),
        cirq.H(qubit2),
    ]


def ryy(
    angle: Union[float, sympy.Symbol],
    qubit1: cirq.GridQubit,
    qubit2: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    YY rotation: exp(-i θ/2 Y⊗Y).

    Decomposition: S†H - CNOT - RZ - CNOT - HS.
    """
    return [
        cirq.rx(np.pi / 2).on(qubit1),
        cirq.rx(np.pi / 2).on(qubit2),
        cirq.CNOT(qubit1, qubit2),
        cirq.rz(angle).on(qubit2),
        cirq.CNOT(qubit1, qubit2),
        cirq.rx(-np.pi / 2).on(qubit1),
        cirq.rx(-np.pi / 2).on(qubit2),
    ]


def rzz(
    angle: Union[float, sympy.Symbol],
    qubit1: cirq.GridQubit,
    qubit2: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    ZZ rotation: exp(-i θ/2 Z⊗Z).

    Decomposition: CNOT - RZ - CNOT.
    """
    return [
        cirq.CNOT(qubit1, qubit2),
        cirq.rz(angle).on(qubit2),
        cirq.CNOT(qubit1, qubit2),
    ]


# =============================================================================
# General Single-Qubit Gate
# =============================================================================

def u3(
    theta: Union[float, sympy.Symbol],
    phi: Union[float, sympy.Symbol],
    lam: Union[float, sympy.Symbol],
    qubit: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    General single-qubit unitary U3(θ, φ, λ).

    U3 = RZ(φ) RY(θ) RZ(λ)

    Any single-qubit unitary can be expressed as U3.

    Parameters
    ----------
    theta : float or Symbol
        Polar angle.
    phi : float or Symbol
        First azimuthal angle.
    lam : float or Symbol
        Second azimuthal angle.
    qubit : GridQubit
        Target qubit.

    Returns
    -------
    list of Operation
        Gate decomposition.
    """
    return [
        cirq.rz(lam).on(qubit),
        cirq.ry(theta).on(qubit),
        cirq.rz(phi).on(qubit),
    ]


def u2(
    phi: Union[float, sympy.Symbol],
    lam: Union[float, sympy.Symbol],
    qubit: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    U2(φ, λ) = U3(π/2, φ, λ).

    Creates arbitrary single-qubit rotation on the Bloch equator.
    """
    return u3(np.pi / 2, phi, lam, qubit)


def u1(
    lam: Union[float, sympy.Symbol],
    qubit: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    U1(λ) = U3(0, 0, λ) = RZ(λ).

    Phase gate.
    """
    return [cirq.rz(lam).on(qubit)]


# =============================================================================
# Controlled-U Gate
# =============================================================================

def controlled_u3(
    theta: Union[float, sympy.Symbol],
    phi: Union[float, sympy.Symbol],
    lam: Union[float, sympy.Symbol],
    control: cirq.GridQubit,
    target: cirq.GridQubit,
) -> List[cirq.Operation]:
    """
    Controlled-U3 gate.

    Decomposition based on ABC decomposition:
    CU = (A ⊗ I) CNOT (B ⊗ I) CNOT (C ⊗ I) Phase
    """
    alpha = (phi + lam) / 2
    beta = (phi - lam) / 2

    ops = [
        cirq.rz(alpha).on(target),
        cirq.CNOT(control, target),
        cirq.rz(-(theta / 2 + beta)).on(target),
        cirq.ry(-theta / 2).on(target),
        cirq.CNOT(control, target),
        cirq.ry(theta / 2).on(target),
        cirq.rz(beta).on(target),
    ]
    return ops


# =============================================================================
# Circuit Builders
# =============================================================================

def parametric_rotation_layer(
    qubits: Sequence[cirq.GridQubit],
    symbols: Sequence[sympy.Symbol],
    gates: str = "RY_RZ",
) -> List[cirq.Operation]:
    """
    Build a parametric rotation layer.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Target qubits.
    symbols : sequence of Symbol
        Parameters (must have enough for all rotations).
    gates : str, default='RY_RZ'
        Gate combination: 'RY', 'RZ', 'RY_RZ', 'RX_RY_RZ'.

    Returns
    -------
    list of Operation
        Rotation operations.
    """
    ops = []
    param_idx = 0
    gate_list = gates.upper().split("_")

    gate_map = {"RX": cirq.rx, "RY": cirq.ry, "RZ": cirq.rz}

    for qubit in qubits:
        for gate_name in gate_list:
            if param_idx < len(symbols):
                ops.append(gate_map[gate_name](symbols[param_idx]).on(qubit))
                param_idx += 1

    return ops
