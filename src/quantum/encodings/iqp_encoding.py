"""
IQP Encoding
============

Instantaneous Quantum Polynomial (IQP) circuits for data encoding.

IQP circuits consist of:
1. Layer of Hadamard gates
2. Diagonal gates (Z-rotations, ZZ-interactions)
3. Layer of Hadamard gates

These circuits are believed to be classically hard to simulate
and provide high expressivity for quantum ML.

References
----------
Havlicek et al., "Supervised learning with quantum-enhanced feature spaces",
Nature 567, 209-212 (2019).
"""

from __future__ import annotations

from typing import Optional, Sequence, List

import cirq
import numpy as np
import sympy

from src.quantum.encodings.base import BaseEncoding


class IQPEncoding(BaseEncoding):
    """
    IQP (Instantaneous Quantum Polynomial) encoding.

    Structure: H^⊗n → Diagonal(x) → H^⊗n

    The diagonal part contains both single-qubit Z rotations and
    two-qubit ZZ interactions, creating entanglement based on data.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int, optional
        Number of input features. If None, equals n_qubits.
    n_repeats : int, default=1
        Number of IQP layer repetitions.
    include_zz : bool, default=True
        Whether to include ZZ interactions.
    zz_connectivity : str, default='linear'
        Connectivity for ZZ gates: 'linear', 'full', 'circular'.

    Examples
    --------
    >>> encoding = IQPEncoding(n_qubits=4, n_repeats=2)
    >>> circuit = encoding.encode()
    >>> print(circuit)
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: Optional[int] = None,
        n_repeats: int = 1,
        include_zz: bool = True,
        zz_connectivity: str = "linear",
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)
        self.n_repeats = n_repeats
        self.include_zz = include_zz
        self.zz_connectivity = zz_connectivity

        # Calculate total number of parameters
        n_z = self.n_features or self.n_qubits
        n_zz = len(self._get_zz_pairs()) if include_zz else 0
        self._n_params = (n_z + n_zz) * n_repeats

    def _get_zz_pairs(self) -> List[tuple]:
        """Get qubit pairs for ZZ interactions based on connectivity."""
        pairs = []
        if self.zz_connectivity == "linear":
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))
        elif self.zz_connectivity == "circular":
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))
            if self.n_qubits > 2:
                pairs.append((self.n_qubits - 1, 0))
        elif self.zz_connectivity == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))
        return pairs

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Create IQP encoding circuit.

        Parameters
        ----------
        data : np.ndarray, optional
            Input data. If None, creates parametric circuit.
        symbols : sequence of sympy.Symbol, optional
            Custom symbols for parametric encoding.

        Returns
        -------
        cirq.Circuit
            IQP encoding circuit.
        """
        circuit = cirq.Circuit()

        # Determine parameters
        n_features = self.n_features or self.n_qubits
        if data is not None:
            params = list(data[:n_features])
            # Pad with zeros if needed
            while len(params) < n_features:
                params.append(0.0)
        else:
            params = list(symbols or self.symbols[:n_features])

        zz_pairs = self._get_zz_pairs() if self.include_zz else []

        for rep in range(self.n_repeats):
            # Initial Hadamard layer
            circuit.append(cirq.H.on_each(*self.qubits))

            # Diagonal layer: Z rotations
            for i, param in enumerate(params):
                if i < self.n_qubits:
                    circuit.append(cirq.rz(param).on(self.qubits[i]))

            # Diagonal layer: ZZ interactions
            for idx, (i, j) in enumerate(zz_pairs):
                # ZZ interaction: exp(-i * θ/2 * Z_i Z_j)
                # Implemented as CNOT - RZ - CNOT
                param_idx = idx % len(params)
                zz_param = params[param_idx] * params[(param_idx + 1) % len(params)]

                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                circuit.append(cirq.rz(zz_param).on(self.qubits[j]))
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))

            # Final Hadamard layer
            circuit.append(cirq.H.on_each(*self.qubits))

        return circuit


class ZFeatureMapEncoding(BaseEncoding):
    """
    Z-Feature Map encoding (variant of IQP).

    Uses second-order expansion with ZZ interactions:
    φ(x) = U_φ(x) H^⊗n where U_φ(x) contains Z(x_i) and ZZ(x_i x_j)

    This is similar to the ZZFeatureMap in Qiskit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int, optional
        Number of input features.
    feature_dimension : int, default=2
        Order of feature interactions (1 or 2).
    reps : int, default=2
        Number of circuit repetitions.
    entanglement : str, default='full'
        Entanglement pattern: 'linear', 'full', 'circular'.
    insert_barriers : bool, default=False
        Whether to insert visual barriers (for debugging).
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: Optional[int] = None,
        feature_dimension: int = 2,
        reps: int = 2,
        entanglement: str = "full",
        insert_barriers: bool = False,
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        self.insert_barriers = insert_barriers

    def _phi_z(self, x, i: int) -> float:
        """First-order feature map: just the feature value."""
        return x

    def _phi_zz(self, x_i, x_j) -> float:
        """Second-order feature map: (π - x_i)(π - x_j)."""
        return (np.pi - x_i) * (np.pi - x_j)

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create Z-feature map encoding circuit."""
        circuit = cirq.Circuit()

        n_features = self.n_features or self.n_qubits
        if data is not None:
            params = list(data[:n_features])
            while len(params) < n_features:
                params.append(0.0)
        else:
            params = list(symbols or self.symbols[:n_features])

        for rep in range(self.reps):
            # Hadamard layer
            circuit.append(cirq.H.on_each(*self.qubits))

            # First-order Z rotations
            for i in range(min(n_features, self.n_qubits)):
                if isinstance(params[i], (int, float)):
                    angle = 2 * params[i]
                else:
                    angle = 2 * params[i]
                circuit.append(cirq.rz(angle).on(self.qubits[i]))

            # Second-order ZZ interactions
            if self.feature_dimension >= 2:
                pairs = self._get_pairs()
                for i, j in pairs:
                    if i < len(params) and j < len(params):
                        if isinstance(params[i], (int, float)):
                            zz_angle = 2 * self._phi_zz(params[i], params[j])
                        else:
                            zz_angle = 2 * (np.pi - params[i]) * (np.pi - params[j])

                        # ZZ gate implementation
                        circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                        circuit.append(cirq.rz(zz_angle).on(self.qubits[j]))
                        circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))

        return circuit

    def _get_pairs(self) -> List[tuple]:
        """Get qubit pairs for entanglement."""
        pairs = []
        if self.entanglement == "linear":
            pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif self.entanglement == "circular":
            pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        elif self.entanglement == "full":
            pairs = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
        return pairs


class PauliFeatureMapEncoding(BaseEncoding):
    """
    Pauli Feature Map encoding.

    Generalizes the Z-feature map to use arbitrary Pauli operators
    for encoding, providing more expressive feature maps.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    paulis : list of str, default=['Z', 'ZZ']
        Pauli strings to use. Options: 'Z', 'ZZ', 'Y', 'YY', 'ZY', etc.
    reps : int, default=2
        Number of repetitions.
    entanglement : str, default='full'
        Entanglement pattern.
    """

    def __init__(
        self,
        n_qubits: int,
        paulis: Optional[List[str]] = None,
        reps: int = 2,
        entanglement: str = "full",
    ):
        super().__init__(n_qubits=n_qubits)
        self.paulis = paulis or ["Z", "ZZ"]
        self.reps = reps
        self.entanglement = entanglement

    def _apply_pauli_rotation(
        self,
        circuit: cirq.Circuit,
        pauli: str,
        qubits: List[cirq.GridQubit],
        angle: float,
    ) -> None:
        """Apply a Pauli rotation gate."""
        if pauli == "Z":
            circuit.append(cirq.rz(angle).on(qubits[0]))
        elif pauli == "Y":
            circuit.append(cirq.ry(angle).on(qubits[0]))
        elif pauli == "X":
            circuit.append(cirq.rx(angle).on(qubits[0]))
        elif pauli == "ZZ":
            # ZZ rotation
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.rz(angle).on(qubits[1]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        elif pauli == "YY":
            # YY rotation via basis change
            circuit.append(cirq.rx(np.pi / 2).on_each(*qubits))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.rz(angle).on(qubits[1]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.rx(-np.pi / 2).on_each(*qubits))
        elif pauli == "XX":
            # XX rotation via basis change
            circuit.append(cirq.H.on_each(*qubits))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.rz(angle).on(qubits[1]))
            circuit.append(cirq.CNOT(qubits[0], qubits[1]))
            circuit.append(cirq.H.on_each(*qubits))

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create Pauli feature map encoding circuit."""
        circuit = cirq.Circuit()

        n_features = self.n_features or self.n_qubits
        if data is not None:
            params = list(data[:n_features])
            while len(params) < n_features:
                params.append(0.0)
        else:
            params = list(symbols or self.symbols[:n_features])

        for rep in range(self.reps):
            # Hadamard layer
            circuit.append(cirq.H.on_each(*self.qubits))

            # Apply each Pauli in the list
            for pauli in self.paulis:
                if len(pauli) == 1:
                    # Single-qubit Pauli
                    for i in range(min(n_features, self.n_qubits)):
                        angle = 2 * params[i] if isinstance(params[i], (int, float)) else 2 * params[i]
                        self._apply_pauli_rotation(
                            circuit, pauli, [self.qubits[i]], angle
                        )
                elif len(pauli) == 2:
                    # Two-qubit Pauli
                    pairs = self._get_pairs()
                    for i, j in pairs:
                        if i < len(params) and j < len(params):
                            if isinstance(params[i], (int, float)):
                                angle = 2 * (np.pi - params[i]) * (np.pi - params[j])
                            else:
                                angle = 2 * (np.pi - params[i]) * (np.pi - params[j])
                            self._apply_pauli_rotation(
                                circuit, pauli, [self.qubits[i], self.qubits[j]], angle
                            )

        return circuit

    def _get_pairs(self) -> List[tuple]:
        """Get qubit pairs."""
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif self.entanglement == "full":
            return [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
        return []
