"""
QAOA-Inspired Circuits
======================

Quantum Approximate Optimization Algorithm (QAOA) inspired circuits
adapted for machine learning tasks.

QAOA structure:
1. Initial superposition: H^⊗n
2. Alternating layers of:
   - Cost layer: exp(-i γ H_C) where H_C encodes the problem
   - Mixer layer: exp(-i β H_M) typically H_M = Σ X_i

For ML, we adapt this by making the cost Hamiltonian depend on input data.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Sequence, Tuple, Union

import cirq
import numpy as np
import sympy


class QAOALayer:
    """
    Single QAOA layer with cost and mixer components.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    cost_type : str, default='zz'
        Type of cost Hamiltonian:
        - 'zz': ZZ interactions (MaxCut-like)
        - 'z': Single-qubit Z (simple cost)
        - 'zzz': Three-body interactions
    mixer_type : str, default='x'
        Type of mixer Hamiltonian:
        - 'x': Transverse field (standard)
        - 'xy': XY mixer (preserves Hamming weight)
        - 'grover': Grover-like mixer
    connectivity : str, default='full'
        Connectivity for ZZ terms: 'linear', 'full', 'circular'.
    """

    def __init__(
        self,
        n_qubits: int,
        cost_type: Literal["zz", "z", "zzz"] = "zz",
        mixer_type: Literal["x", "xy", "grover"] = "x",
        connectivity: Literal["linear", "full", "circular"] = "full",
    ):
        self.n_qubits = n_qubits
        self.cost_type = cost_type
        self.mixer_type = mixer_type
        self.connectivity = connectivity

        self.qubits = cirq.GridQubit.rect(1, n_qubits)

    def _get_zz_pairs(self) -> List[Tuple[int, int]]:
        """Get qubit pairs for ZZ interactions."""
        pairs = []
        if self.connectivity == "linear":
            pairs = [(i, i + 1) for i in range(self.n_qubits - 1)]
        elif self.connectivity == "circular":
            pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]
        elif self.connectivity == "full":
            pairs = [
                (i, j)
                for i in range(self.n_qubits)
                for j in range(i + 1, self.n_qubits)
            ]
        return pairs

    def cost_layer(
        self,
        gamma: Union[float, sympy.Symbol],
        data_params: Optional[Sequence] = None,
    ) -> cirq.Circuit:
        """
        Build cost layer exp(-i γ H_C).

        Parameters
        ----------
        gamma : float or Symbol
            Cost layer parameter.
        data_params : sequence, optional
            Data-dependent parameters for the cost function.

        Returns
        -------
        cirq.Circuit
            Cost layer circuit.
        """
        circuit = cirq.Circuit()

        if self.cost_type == "z":
            # Single-qubit Z rotations
            for i, qubit in enumerate(self.qubits):
                if data_params is not None and i < len(data_params):
                    angle = gamma * data_params[i]
                else:
                    angle = gamma
                circuit.append(cirq.rz(2 * angle).on(qubit))

        elif self.cost_type == "zz":
            # ZZ interactions
            pairs = self._get_zz_pairs()
            for i, j in pairs:
                if data_params is not None:
                    # Data-dependent coupling
                    idx = min(i, len(data_params) - 1) if data_params else 0
                    coupling = data_params[idx] if data_params else 1.0
                    angle = gamma * coupling
                else:
                    angle = gamma

                # ZZ gate: CNOT - RZ - CNOT
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                circuit.append(cirq.rz(2 * angle).on(self.qubits[j]))
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))

        return circuit

    def mixer_layer(
        self,
        beta: Union[float, sympy.Symbol],
    ) -> cirq.Circuit:
        """
        Build mixer layer exp(-i β H_M).

        Parameters
        ----------
        beta : float or Symbol
            Mixer layer parameter.

        Returns
        -------
        cirq.Circuit
            Mixer layer circuit.
        """
        circuit = cirq.Circuit()

        if self.mixer_type == "x":
            # Standard transverse field mixer: RX on each qubit
            for qubit in self.qubits:
                circuit.append(cirq.rx(2 * beta).on(qubit))

        elif self.mixer_type == "xy":
            # XY mixer (preserves Hamming weight)
            pairs = self._get_zz_pairs()
            for i, j in pairs:
                # XX + YY interaction
                # Implemented as: H-CNOT-RY-CNOT-H
                circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[j]))
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                circuit.append(cirq.ry(beta).on(self.qubits[j]))
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))
                circuit.append(cirq.H.on_each(self.qubits[i], self.qubits[j]))

        elif self.mixer_type == "grover":
            # Grover-like mixer: diffusion operator
            circuit.append(cirq.H.on_each(*self.qubits))
            circuit.append(cirq.X.on_each(*self.qubits))

            # Multi-controlled Z
            if self.n_qubits >= 2:
                # Simplified: CZ on all pairs
                for i in range(self.n_qubits - 1):
                    circuit.append(cirq.CZ(self.qubits[i], self.qubits[i + 1]))

            circuit.append(cirq.X.on_each(*self.qubits))
            circuit.append(cirq.H.on_each(*self.qubits))

        return circuit


class QAOA:
    """
    QAOA-inspired variational circuit for quantum ML.

    Structure:
    1. Initial state preparation (Hadamard layer)
    2. p layers of (cost_layer, mixer_layer)
    3. Measurements

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    p : int, default=2
        Number of QAOA layers (depth).
    cost_type : str, default='zz'
        Cost Hamiltonian type.
    mixer_type : str, default='x'
        Mixer Hamiltonian type.
    connectivity : str, default='full'
        Qubit connectivity.
    data_reuploading : bool, default=True
        Whether to re-encode data in each layer.

    Examples
    --------
    >>> qaoa = QAOA(n_qubits=4, p=3)
    >>> circuit = qaoa.build()
    >>> print(f"Parameters: {qaoa.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        p: int = 2,
        cost_type: Literal["zz", "z", "zzz"] = "zz",
        mixer_type: Literal["x", "xy", "grover"] = "x",
        connectivity: Literal["linear", "full", "circular"] = "full",
        data_reuploading: bool = True,
    ):
        self.n_qubits = n_qubits
        self.p = p
        self.cost_type = cost_type
        self.mixer_type = mixer_type
        self.connectivity = connectivity
        self.data_reuploading = data_reuploading

        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self.layer = QAOALayer(
            n_qubits, cost_type, mixer_type, connectivity
        )

        # Number of parameters: gamma and beta per layer
        self._n_params = 2 * p
        self._symbols = [sympy.Symbol(f"qaoa_{i}") for i in range(self._n_params)]

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self._n_params

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Trainable parameters."""
        return self._symbols

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
        data_symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Build the QAOA circuit.

        Parameters
        ----------
        symbols : sequence of sympy.Symbol, optional
            Custom symbols for gamma/beta parameters.
        data_symbols : sequence of sympy.Symbol, optional
            Symbols for data encoding in cost layer.

        Returns
        -------
        cirq.Circuit
            The QAOA circuit.
        """
        circuit = cirq.Circuit()
        params = symbols or self._symbols

        # Initial superposition
        circuit.append(cirq.H.on_each(*self.qubits))

        # QAOA layers
        for layer_idx in range(self.p):
            gamma = params[2 * layer_idx]
            beta = params[2 * layer_idx + 1]

            # Cost layer (with optional data encoding)
            data_params = data_symbols if self.data_reuploading else None
            circuit += self.layer.cost_layer(gamma, data_params)

            # Mixer layer
            circuit += self.layer.mixer_layer(beta)

        return circuit

    def get_initial_params(
        self,
        strategy: str = "random",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Get initial parameter values."""
        rng = np.random.RandomState(seed)

        if strategy == "zeros":
            return np.zeros(self._n_params)
        elif strategy == "linear":
            # Linear ramp (good for QAOA)
            gammas = np.linspace(0.1, 1.0, self.p)
            betas = np.linspace(1.0, 0.1, self.p)
            params = np.zeros(self._n_params)
            params[0::2] = gammas
            params[1::2] = betas
            return params
        else:  # random
            return rng.uniform(0, np.pi, self._n_params)

    def __repr__(self) -> str:
        return (
            f"QAOA(n_qubits={self.n_qubits}, p={self.p}, "
            f"cost='{self.cost_type}', mixer='{self.mixer_type}')"
        )


class DataReuploadingQAOA(QAOA):
    """
    QAOA with explicit data re-uploading at each layer.

    This variant encodes input data in every layer, which can increase
    expressivity but also increases the number of input parameters.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int
        Number of input features.
    p : int, default=2
        Number of layers.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        p: int = 2,
        cost_type: str = "zz",
        mixer_type: str = "x",
    ):
        super().__init__(
            n_qubits=n_qubits,
            p=p,
            cost_type=cost_type,
            mixer_type=mixer_type,
            data_reuploading=True,
        )
        self.n_features = n_features

        # Create data symbols
        self._data_symbols = [
            sympy.Symbol(f"x_{i}") for i in range(n_features)
        ]

    @property
    def data_symbols(self) -> List[sympy.Symbol]:
        """Data input symbols."""
        return self._data_symbols

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
        data_symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Build circuit with data re-uploading."""
        data_syms = data_symbols or self._data_symbols
        return super().build(symbols=symbols, data_symbols=data_syms)
