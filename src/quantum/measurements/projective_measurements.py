"""
Projective Measurements
=======================

Custom projective measurement operators and POVM-like measurements
for quantum state analysis and classification.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import cirq
import numpy as np


class ProjectiveMeasurement:
    """
    Projective measurement onto specified subspaces.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Qubits to measure.
    n_outcomes : int, default=2
        Number of measurement outcomes.

    Examples
    --------
    >>> qubits = cirq.GridQubit.rect(1, 4)
    >>> pm = ProjectiveMeasurement(qubits)
    >>> operators = pm.computational_basis()
    """

    def __init__(
        self,
        qubits: Sequence[cirq.GridQubit],
        n_outcomes: int = 2,
    ):
        self.qubits = list(qubits)
        self.n_qubits = len(qubits)
        self.n_outcomes = n_outcomes

    def computational_basis(self) -> List[cirq.PauliString]:
        """
        Z-basis projective measurement on all qubits.

        Returns
        -------
        list of PauliString
            One Z operator per qubit.
        """
        return [cirq.Z(q) for q in self.qubits]

    def parity_measurement(
        self,
        qubit_groups: Optional[List[List[int]]] = None,
    ) -> List[cirq.PauliString]:
        """
        Parity measurement: product of Z operators on groups of qubits.

        Parameters
        ----------
        qubit_groups : list of list of int, optional
            Groups of qubit indices. Default: even/odd partition.

        Returns
        -------
        list of PauliString
            Parity operators.
        """
        if qubit_groups is None:
            # Default: measure parity of all qubits
            qubit_groups = [list(range(self.n_qubits))]

        operators = []
        for group in qubit_groups:
            if len(group) == 0:
                continue
            op = cirq.Z(self.qubits[group[0]])
            for idx in group[1:]:
                op = op * cirq.Z(self.qubits[idx])
            operators.append(op)

        return operators

    def binary_classification_operator(
        self,
        qubit_index: int = 0,
    ) -> List[cirq.PauliString]:
        """
        Single-qubit measurement for binary classification.

        Maps ⟨Z⟩ ∈ [-1, 1] to class probabilities.

        Parameters
        ----------
        qubit_index : int, default=0
            Which qubit to measure.

        Returns
        -------
        list of PauliString
            Single Z operator.
        """
        return [cirq.Z(self.qubits[qubit_index])]

    def multiclass_operators(
        self,
        n_classes: int,
    ) -> List[cirq.PauliString]:
        """
        Measurement operators for multi-class classification.

        Uses one qubit per class (up to n_qubits).

        Parameters
        ----------
        n_classes : int
            Number of classes.

        Returns
        -------
        list of PauliString
            One Z operator per class.
        """
        n_measured = min(n_classes, self.n_qubits)
        return [cirq.Z(self.qubits[i]) for i in range(n_measured)]


class POVMMeasurement:
    """
    Positive Operator-Valued Measure (POVM) approximation.

    Implements informationally complete measurements using
    rotated Pauli measurements.

    Parameters
    ----------
    qubits : sequence of GridQubit
        Qubits to measure.
    """

    def __init__(self, qubits: Sequence[cirq.GridQubit]):
        self.qubits = list(qubits)

    def symmetric_ic_povm(self) -> Dict[str, List]:
        """
        Symmetric informationally complete POVM (SIC-POVM) approximation.

        For single-qubit SIC-POVM, uses 4 measurement directions
        corresponding to tetrahedron vertices on the Bloch sphere.

        Returns
        -------
        dict
            'operators': measurement operators per qubit,
            'basis_changes': circuits for basis rotation.
        """
        # Tetrahedron directions on Bloch sphere
        # Each direction defines a projector |ψ_i⟩⟨ψ_i|/2
        angles = [
            (0.0, 0.0),              # +Z
            (2 * np.arccos(1 / np.sqrt(3)), 0.0),              # rotated
            (2 * np.arccos(1 / np.sqrt(3)), 2 * np.pi / 3),    # rotated
            (2 * np.arccos(1 / np.sqrt(3)), 4 * np.pi / 3),    # rotated
        ]

        basis_changes = []
        for theta, phi in angles:
            circuit = cirq.Circuit()
            for qubit in self.qubits:
                circuit.append(cirq.ry(theta).on(qubit))
                circuit.append(cirq.rz(phi).on(qubit))
            basis_changes.append(circuit)

        operators = [cirq.Z(q) for q in self.qubits]

        return {
            "operators": operators,
            "basis_changes": basis_changes,
        }

    def pauli_6_povm(self) -> Dict[str, List]:
        """
        6-element POVM using ±X, ±Y, ±Z measurements.

        Returns
        -------
        dict
            'operators': Z operators,
            'basis_changes': circuits for X, Y, Z bases.
        """
        basis_changes = []

        # Z basis (no change needed)
        basis_changes.append(cirq.Circuit())

        # X basis
        x_circuit = cirq.Circuit()
        for qubit in self.qubits:
            x_circuit.append(cirq.H(qubit))
        basis_changes.append(x_circuit)

        # Y basis
        y_circuit = cirq.Circuit()
        for qubit in self.qubits:
            y_circuit.append(cirq.rx(-np.pi / 2).on(qubit))
        basis_changes.append(y_circuit)

        operators = [cirq.Z(q) for q in self.qubits]

        return {
            "operators": operators,
            "basis_changes": basis_changes,
            "labels": ["Z", "X", "Y"],
        }
