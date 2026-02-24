"""
Hamiltonian Encoding
====================

Encodes classical data via time-evolution under a data-dependent Hamiltonian.

|ψ(x)⟩ = exp(-i H(x) t) |0⟩^⊗n

where H(x) is constructed from the input features.

References
----------
Schuld et al., "Circuit-centric quantum classifiers",
Phys. Rev. A 101, 032308 (2020).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple

import cirq
import numpy as np
import sympy

from src.quantum.encodings.base import BaseEncoding


class HamiltonianEncoding(BaseEncoding):
    """
    Hamiltonian time-evolution encoding.

    Encodes data as parameters of a Hamiltonian and applies
    time evolution as a unitary transformation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int, optional
        Number of features. Defaults to n_qubits.
    evolution_time : float, default=1.0
        Total evolution time parameter.
    hamiltonian_type : str, default='ising'
        Type of Hamiltonian:
        - 'ising': Transverse-field Ising model
        - 'heisenberg': Heisenberg XXX model
        - 'xy': XY model
    n_trotter_steps : int, default=1
        Number of Trotter-Suzuki decomposition steps.

    Examples
    --------
    >>> encoding = HamiltonianEncoding(n_qubits=4, hamiltonian_type='ising')
    >>> circuit = encoding.encode(data=np.array([0.5, 1.0, 0.3, 0.8]))
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: Optional[int] = None,
        evolution_time: float = 1.0,
        hamiltonian_type: Literal["ising", "heisenberg", "xy"] = "ising",
        n_trotter_steps: int = 1,
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)
        self.evolution_time = evolution_time
        self.hamiltonian_type = hamiltonian_type
        self.n_trotter_steps = n_trotter_steps

    def _ising_layer(
        self,
        params: Sequence,
        dt: float,
    ) -> cirq.Circuit:
        """
        Build one Trotter step for the transverse-field Ising model.

        H = -Σ_i J_i Z_i Z_{i+1} - Σ_i h_i X_i
        where J_i and h_i depend on input data.
        """
        circuit = cirq.Circuit()
        n_features = len(params)

        # ZZ interaction terms
        for i in range(self.n_qubits - 1):
            param_idx = i % n_features
            angle = 2 * dt * params[param_idx]

            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))

        # Transverse field (X rotation) terms
        for i in range(self.n_qubits):
            param_idx = i % n_features
            angle = 2 * dt * params[param_idx]
            circuit.append(cirq.rx(angle).on(self.qubits[i]))

        return circuit

    def _heisenberg_layer(
        self,
        params: Sequence,
        dt: float,
    ) -> cirq.Circuit:
        """
        Build one Trotter step for the Heisenberg XXX model.

        H = Σ_i J_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
        """
        circuit = cirq.Circuit()
        n_features = len(params)

        for i in range(self.n_qubits - 1):
            param_idx = i % n_features
            angle = dt * params[param_idx]

            # XX interaction
            circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(2 * angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[i + 1]))

            # YY interaction
            circuit.append(cirq.rx(np.pi / 2).on_each(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(2 * angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rx(-np.pi / 2).on_each(self.qubits[i], self.qubits[i + 1]))

            # ZZ interaction
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(2 * angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))

        return circuit

    def _xy_layer(
        self,
        params: Sequence,
        dt: float,
    ) -> cirq.Circuit:
        """
        Build one Trotter step for the XY model.

        H = Σ_i J_i (X_i X_{i+1} + Y_i Y_{i+1})
        """
        circuit = cirq.Circuit()
        n_features = len(params)

        for i in range(self.n_qubits - 1):
            param_idx = i % n_features
            angle = dt * params[param_idx]

            # XX + YY via iSWAP-like decomposition
            circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(2 * angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[i + 1]))

            circuit.append(cirq.rx(np.pi / 2).on_each(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rz(2 * angle).on(self.qubits[i + 1]))
            circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            circuit.append(cirq.rx(-np.pi / 2).on_each(self.qubits[i], self.qubits[i + 1]))

        return circuit

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Create Hamiltonian encoding circuit.

        Parameters
        ----------
        data : np.ndarray, optional
            Input features. Used as Hamiltonian coefficients.
        symbols : sequence of sympy.Symbol, optional
            Parametric symbols (if data not provided).

        Returns
        -------
        cirq.Circuit
            Time-evolution encoding circuit.
        """
        circuit = cirq.Circuit()

        # Initial superposition
        circuit.append(cirq.H.on_each(*self.qubits))

        # Determine parameters
        n_features = self.n_features or self.n_qubits
        if data is not None:
            params = list(data[:n_features])
            while len(params) < n_features:
                params.append(0.0)
        else:
            params = list(symbols or self.symbols[:n_features])

        # Time step per Trotter step
        dt = self.evolution_time / self.n_trotter_steps

        # Build Trotter steps
        layer_fn = {
            "ising": self._ising_layer,
            "heisenberg": self._heisenberg_layer,
            "xy": self._xy_layer,
        }[self.hamiltonian_type]

        for _ in range(self.n_trotter_steps):
            circuit += layer_fn(params, dt)

        return circuit

    def __repr__(self) -> str:
        return (
            f"HamiltonianEncoding(n_qubits={self.n_qubits}, "
            f"type='{self.hamiltonian_type}', "
            f"trotter_steps={self.n_trotter_steps})"
        )
