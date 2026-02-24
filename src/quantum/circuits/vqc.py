"""
Variational Quantum Classifier (VQC)
====================================

Parameterized quantum circuits for classification tasks.

The VQC consists of:
1. Data encoding layer (handled separately)
2. Variational ansatz (trainable parameters)
3. Measurement layer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import cirq
import numpy as np
import sympy


class BaseAnsatz(ABC):
    """
    Abstract base class for variational ansatze.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.

    Attributes
    ----------
    qubits : list of cirq.GridQubit
        Qubit objects.
    symbols : list of sympy.Symbol
        Trainable parameters.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self._symbols: Optional[List[sympy.Symbol]] = None
        self._n_params: Optional[int] = None

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of trainable parameters."""
        pass

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Symbolic parameters for the ansatz."""
        if self._symbols is None:
            self._symbols = [sympy.Symbol(f"θ_{i}") for i in range(self.n_params)]
        return self._symbols

    @abstractmethod
    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Build the ansatz circuit."""
        pass

    def get_initial_params(
        self,
        strategy: Literal["random", "zeros", "identity"] = "random",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Get initial parameter values.

        Parameters
        ----------
        strategy : str
            Initialization strategy:
            - 'random': Uniform random in [0, 2π)
            - 'zeros': All zeros
            - 'identity': Values that approximate identity (small random)
        seed : int, optional
            Random seed.

        Returns
        -------
        np.ndarray
            Initial parameter values.
        """
        rng = np.random.RandomState(seed)

        if strategy == "zeros":
            return np.zeros(self.n_params)
        elif strategy == "identity":
            # Small values near zero approximate identity transformations
            return rng.uniform(-0.1, 0.1, self.n_params)
        else:  # random
            return rng.uniform(0, 2 * np.pi, self.n_params)


class HardwareEfficientAnsatz(BaseAnsatz):
    """
    Hardware-efficient ansatz with single-qubit rotations and entangling gates.

    Structure per layer:
    1. RY rotation on each qubit
    2. RZ rotation on each qubit
    3. Entangling layer (CNOT gates)

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    entanglement : str, default='linear'
        Entanglement topology:
        - 'linear': Chain connectivity (qubit i to i+1)
        - 'full': All-to-all connectivity
        - 'circular': Ring topology (linear + last to first)
        - 'star': Hub-and-spoke (all connected to qubit 0)
    rotation_gates : tuple, default=('RY', 'RZ')
        Rotation gates to use per layer.

    Examples
    --------
    >>> ansatz = HardwareEfficientAnsatz(n_qubits=4, n_layers=3)
    >>> circuit = ansatz.build()
    >>> print(f"Parameters: {ansatz.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        entanglement: Literal["linear", "full", "circular", "star"] = "linear",
        rotation_gates: Tuple[str, ...] = ("RY", "RZ"),
    ):
        super().__init__(n_qubits, n_layers)
        self.entanglement = entanglement
        self.rotation_gates = tuple(g.upper() for g in rotation_gates)

        # Validate rotation gates
        valid_gates = {"RX", "RY", "RZ"}
        for gate in self.rotation_gates:
            if gate not in valid_gates:
                raise ValueError(f"Invalid rotation gate: {gate}")

    @property
    def n_params(self) -> int:
        """Number of parameters: rotations per qubit per layer."""
        if self._n_params is None:
            rotations_per_layer = self.n_qubits * len(self.rotation_gates)
            self._n_params = rotations_per_layer * self.n_layers
        return self._n_params

    def _get_entangling_pairs(self) -> List[Tuple[int, int]]:
        """Get qubit pairs for entanglement based on topology."""
        pairs = []

        if self.entanglement == "linear":
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))

        elif self.entanglement == "circular":
            for i in range(self.n_qubits - 1):
                pairs.append((i, i + 1))
            if self.n_qubits > 2:
                pairs.append((self.n_qubits - 1, 0))

        elif self.entanglement == "full":
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    pairs.append((i, j))

        elif self.entanglement == "star":
            for i in range(1, self.n_qubits):
                pairs.append((0, i))

        return pairs

    def _rotation_gate(
        self,
        gate_name: str,
        angle: Union[float, sympy.Symbol],
    ) -> cirq.Gate:
        """Get rotation gate by name."""
        gates = {
            "RX": cirq.rx,
            "RY": cirq.ry,
            "RZ": cirq.rz,
        }
        return gates[gate_name](angle)

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Build the hardware-efficient ansatz circuit."""
        circuit = cirq.Circuit()
        params = symbols or self.symbols
        param_idx = 0

        entangling_pairs = self._get_entangling_pairs()

        for layer in range(self.n_layers):
            # Rotation layers
            for gate_name in self.rotation_gates:
                rotation_ops = []
                for qubit_idx in range(self.n_qubits):
                    gate = self._rotation_gate(gate_name, params[param_idx])
                    rotation_ops.append(gate.on(self.qubits[qubit_idx]))
                    param_idx += 1
                circuit.append(rotation_ops)

            # Entangling layer
            for i, j in entangling_pairs:
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[j]))

        return circuit


class StronglyEntanglingAnsatz(BaseAnsatz):
    """
    Strongly entangling ansatz with full rotations and shifted entanglement.

    Each layer has:
    1. Full rotation (RX, RY, RZ) on each qubit
    2. CNOT entanglement with layer-dependent offset

    This provides high expressivity but may be prone to barren plateaus.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__(n_qubits, n_layers)

    @property
    def n_params(self) -> int:
        """3 rotations per qubit per layer."""
        if self._n_params is None:
            self._n_params = 3 * self.n_qubits * self.n_layers
        return self._n_params

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Build strongly entangling ansatz."""
        circuit = cirq.Circuit()
        params = symbols or self.symbols
        param_idx = 0

        for layer in range(self.n_layers):
            # Full rotation on each qubit
            for qubit_idx in range(self.n_qubits):
                qubit = self.qubits[qubit_idx]
                circuit.append(cirq.rx(params[param_idx]).on(qubit))
                param_idx += 1
                circuit.append(cirq.ry(params[param_idx]).on(qubit))
                param_idx += 1
                circuit.append(cirq.rz(params[param_idx]).on(qubit))
                param_idx += 1

            # Entanglement with layer-dependent offset
            offset = layer % self.n_qubits
            for i in range(self.n_qubits):
                control = self.qubits[i]
                target = self.qubits[(i + 1 + offset) % self.n_qubits]
                if control != target:
                    circuit.append(cirq.CNOT(control, target))

        return circuit


class VQC:
    """
    Variational Quantum Classifier.

    Combines encoding, variational ansatz, and measurement into a
    complete classification circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.
    ansatz : str or BaseAnsatz, default='hardware_efficient'
        Ansatz type or instance.
    entanglement : str, default='linear'
        Entanglement topology for built-in ansatze.

    Attributes
    ----------
    ansatz : BaseAnsatz
        The variational ansatz.
    qubits : list
        Qubit objects.
    symbols : list
        Trainable parameters.

    Examples
    --------
    >>> vqc = VQC(n_qubits=4, n_layers=3, entanglement='full')
    >>> circuit = vqc.build()
    >>> print(f"Trainable parameters: {vqc.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        ansatz: Union[str, BaseAnsatz] = "hardware_efficient",
        entanglement: Literal["linear", "full", "circular", "star"] = "linear",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Create ansatz
        if isinstance(ansatz, BaseAnsatz):
            self.ansatz = ansatz
        elif ansatz == "hardware_efficient":
            self.ansatz = HardwareEfficientAnsatz(
                n_qubits, n_layers, entanglement=entanglement
            )
        elif ansatz == "strongly_entangling":
            self.ansatz = StronglyEntanglingAnsatz(n_qubits, n_layers)
        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        self.qubits = self.ansatz.qubits
        self._circuit: Optional[cirq.Circuit] = None

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Trainable parameters."""
        return self.ansatz.symbols

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self.ansatz.n_params

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Build the VQC circuit.

        Parameters
        ----------
        symbols : sequence of sympy.Symbol, optional
            Custom symbols for parameters.

        Returns
        -------
        cirq.Circuit
            The variational circuit.
        """
        return self.ansatz.build(symbols)

    def get_circuit(self) -> cirq.Circuit:
        """Get the cached circuit (builds if needed)."""
        if self._circuit is None:
            self._circuit = self.build()
        return self._circuit

    def get_initial_params(
        self,
        strategy: str = "random",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Get initial parameter values."""
        return self.ansatz.get_initial_params(strategy, seed)

    def resolve(
        self,
        params: np.ndarray,
        circuit: Optional[cirq.Circuit] = None,
    ) -> cirq.Circuit:
        """
        Resolve circuit with concrete parameter values.

        Parameters
        ----------
        params : np.ndarray
            Parameter values.
        circuit : cirq.Circuit, optional
            Circuit to resolve. Uses cached if None.

        Returns
        -------
        cirq.Circuit
            Resolved circuit.
        """
        if circuit is None:
            circuit = self.get_circuit()

        resolver = dict(zip(self.symbols, params))
        return cirq.resolve_parameters(circuit, resolver)

    def __repr__(self) -> str:
        return (
            f"VQC(n_qubits={self.n_qubits}, n_layers={self.n_layers}, "
            f"ansatz={self.ansatz.__class__.__name__})"
        )
