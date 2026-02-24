"""
Quantum Convolutional Neural Network (QCNN)
===========================================

Implements quantum CNN architecture with:
- Convolutional layers (two-qubit gates with translation invariance)
- Pooling layers (measurement-based or trace-out)
- Fully connected layers

References
----------
Cong et al., "Quantum Convolutional Neural Networks",
Nature Physics 15, 1273-1278 (2019).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Sequence, Tuple

import cirq
import numpy as np
import sympy


class QCNNLayer(ABC):
    """Abstract base class for QCNN layers."""

    @abstractmethod
    def build(
        self,
        qubits: List[cirq.GridQubit],
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> Tuple[cirq.Circuit, int]:
        """
        Build the layer circuit.

        Returns
        -------
        circuit : cirq.Circuit
            The layer circuit.
        n_params : int
            Number of parameters used.
        """
        pass


class ConvolutionalLayer(QCNNLayer):
    """
    Quantum convolutional layer.

    Applies parameterized two-qubit gates in a translationally invariant
    pattern across all qubits.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolutional kernel (number of qubits).
    stride : int, default=1
        Stride of the convolution.
    periodic : bool, default=True
        Whether to use periodic boundary conditions.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        periodic: bool = True,
    ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.periodic = periodic

    def _two_qubit_unitary(
        self,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        params: Sequence[sympy.Symbol],
        param_offset: int,
    ) -> List[cirq.Operation]:
        """Create parameterized two-qubit unitary."""
        ops = []

        # U3 on first qubit
        ops.append(cirq.rz(params[param_offset]).on(qubit1))
        ops.append(cirq.ry(params[param_offset + 1]).on(qubit1))
        ops.append(cirq.rz(params[param_offset + 2]).on(qubit1))

        # U3 on second qubit
        ops.append(cirq.rz(params[param_offset + 3]).on(qubit2))
        ops.append(cirq.ry(params[param_offset + 4]).on(qubit2))
        ops.append(cirq.rz(params[param_offset + 5]).on(qubit2))

        # Entangling gate
        ops.append(cirq.CNOT(qubit1, qubit2))

        # Post-entanglement rotations
        ops.append(cirq.ry(params[param_offset + 6]).on(qubit1))
        ops.append(cirq.ry(params[param_offset + 7]).on(qubit2))

        return ops

    def build(
        self,
        qubits: List[cirq.GridQubit],
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> Tuple[cirq.Circuit, int]:
        """Build convolutional layer."""
        circuit = cirq.Circuit()
        n_qubits = len(qubits)

        # Parameters per kernel application
        params_per_kernel = 8

        # Create symbols if needed
        n_applications = (n_qubits - self.kernel_size) // self.stride + 1
        if self.periodic:
            n_applications = n_qubits // self.stride

        # For translation invariance, reuse same parameters
        total_params = params_per_kernel
        if symbols is None:
            symbols = [sympy.Symbol(f"conv_{i}") for i in range(total_params)]

        # Apply kernels
        for i in range(0, n_qubits, self.stride):
            if self.kernel_size == 2:
                j = (i + 1) % n_qubits if self.periodic else i + 1
                if j < n_qubits:
                    ops = self._two_qubit_unitary(qubits[i], qubits[j], symbols, 0)
                    circuit.append(ops)

        return circuit, total_params


class PoolingLayer(QCNNLayer):
    """
    Quantum pooling layer.

    Reduces the number of qubits by measuring or tracing out every other qubit,
    conditioned on measurement outcomes.

    Parameters
    ----------
    pooling_type : str, default='measure'
        Type of pooling:
        - 'measure': Measure and conditionally rotate
        - 'trace': Controlled rotation then trace out
    """

    def __init__(
        self,
        pooling_type: Literal["measure", "trace"] = "trace",
    ):
        self.pooling_type = pooling_type

    def build(
        self,
        qubits: List[cirq.GridQubit],
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> Tuple[cirq.Circuit, int]:
        """Build pooling layer."""
        circuit = cirq.Circuit()
        n_qubits = len(qubits)

        # Parameters per pooling operation
        params_per_pool = 2
        n_pools = n_qubits // 2
        total_params = params_per_pool

        if symbols is None:
            symbols = [sympy.Symbol(f"pool_{i}") for i in range(total_params)]

        # Pair up qubits and apply pooling
        for i in range(0, n_qubits - 1, 2):
            source = qubits[i]
            target = qubits[i + 1]

            if self.pooling_type == "trace":
                # Controlled rotation: transfer info from source to target
                circuit.append(cirq.CNOT(source, target))
                circuit.append(cirq.ry(symbols[0]).on(target).controlled_by(source))
                circuit.append(cirq.rz(symbols[1]).on(target).controlled_by(source))

            elif self.pooling_type == "measure":
                # Simpler version: just CNOT and rotation
                circuit.append(cirq.CNOT(source, target))
                circuit.append(cirq.ry(symbols[0]).on(target))

        return circuit, total_params


class FullyConnectedLayer(QCNNLayer):
    """
    Fully connected quantum layer.

    Applies all-to-all entanglement followed by single-qubit rotations.
    Used at the end of QCNN for classification.

    Parameters
    ----------
    n_layers : int, default=1
        Number of repetitions.
    """

    def __init__(self, n_layers: int = 1):
        self.n_layers = n_layers

    def build(
        self,
        qubits: List[cirq.GridQubit],
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> Tuple[cirq.Circuit, int]:
        """Build fully connected layer."""
        circuit = cirq.Circuit()
        n_qubits = len(qubits)

        # 3 rotations per qubit per layer
        total_params = 3 * n_qubits * self.n_layers

        if symbols is None:
            symbols = [sympy.Symbol(f"fc_{i}") for i in range(total_params)]

        param_idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for qubit in qubits:
                circuit.append(cirq.rx(symbols[param_idx]).on(qubit))
                param_idx += 1
                circuit.append(cirq.ry(symbols[param_idx]).on(qubit))
                param_idx += 1
                circuit.append(cirq.rz(symbols[param_idx]).on(qubit))
                param_idx += 1

            # All-to-all entanglement
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    circuit.append(cirq.CZ(qubits[i], qubits[j]))

        return circuit, total_params


class QCNN:
    """
    Quantum Convolutional Neural Network.

    Hierarchical architecture with convolutional and pooling layers
    that progressively reduce the number of active qubits.

    Parameters
    ----------
    n_qubits : int
        Number of input qubits (should be power of 2).
    n_conv_layers : int, default=2
        Number of convolutional layers per block.
    pooling_type : str, default='trace'
        Type of pooling layer.

    Attributes
    ----------
    qubits : list
        Qubit objects.
    n_params : int
        Total number of trainable parameters.

    Examples
    --------
    >>> qcnn = QCNN(n_qubits=8, n_conv_layers=2)
    >>> circuit = qcnn.build()
    >>> print(f"Parameters: {qcnn.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        n_conv_layers: int = 2,
        pooling_type: Literal["measure", "trace"] = "trace",
    ):
        self.n_qubits = n_qubits
        self.n_conv_layers = n_conv_layers
        self.pooling_type = pooling_type

        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self._symbols: Optional[List[sympy.Symbol]] = None
        self._n_params: Optional[int] = None

    @property
    def n_params(self) -> int:
        """Total number of parameters."""
        if self._n_params is None:
            # Compute by building
            self.build()
        return self._n_params

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Trainable parameters."""
        if self._symbols is None:
            self.build()
        return self._symbols

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Build the QCNN circuit.

        Parameters
        ----------
        symbols : sequence of sympy.Symbol, optional
            Custom symbols for parameters.

        Returns
        -------
        cirq.Circuit
            The QCNN circuit.
        """
        circuit = cirq.Circuit()
        active_qubits = list(self.qubits)
        all_symbols = []
        symbol_idx = 0

        # Number of hierarchical levels
        n_levels = int(np.log2(self.n_qubits))

        for level in range(n_levels):
            n_active = len(active_qubits)
            if n_active < 2:
                break

            # Convolutional layers
            for conv_layer in range(self.n_conv_layers):
                conv = ConvolutionalLayer(kernel_size=2, periodic=True)

                if symbols is None:
                    layer_symbols = [
                        sympy.Symbol(f"θ_{symbol_idx + i}")
                        for i in range(8)  # params per conv kernel
                    ]
                else:
                    layer_symbols = symbols[symbol_idx : symbol_idx + 8]

                layer_circuit, n_params = conv.build(active_qubits, layer_symbols)
                circuit += layer_circuit
                all_symbols.extend(layer_symbols[:n_params])
                symbol_idx += n_params

            # Pooling layer (except at last level)
            if level < n_levels - 1 and n_active >= 4:
                pool = PoolingLayer(pooling_type=self.pooling_type)

                if symbols is None:
                    layer_symbols = [
                        sympy.Symbol(f"θ_{symbol_idx + i}")
                        for i in range(2)
                    ]
                else:
                    layer_symbols = symbols[symbol_idx : symbol_idx + 2]

                layer_circuit, n_params = pool.build(active_qubits, layer_symbols)
                circuit += layer_circuit
                all_symbols.extend(layer_symbols[:n_params])
                symbol_idx += n_params

                # Reduce active qubits (keep every other)
                active_qubits = active_qubits[1::2]

        # Final fully connected layer
        if len(active_qubits) >= 1:
            fc = FullyConnectedLayer(n_layers=1)

            if symbols is None:
                layer_symbols = [
                    sympy.Symbol(f"θ_{symbol_idx + i}")
                    for i in range(3 * len(active_qubits))
                ]
            else:
                layer_symbols = symbols[symbol_idx : symbol_idx + 3 * len(active_qubits)]

            layer_circuit, n_params = fc.build(active_qubits, layer_symbols)
            circuit += layer_circuit
            all_symbols.extend(layer_symbols[:n_params])

        self._symbols = all_symbols
        self._n_params = len(all_symbols)

        return circuit

    def get_initial_params(
        self,
        strategy: str = "random",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Get initial parameter values."""
        rng = np.random.RandomState(seed)

        if self._n_params is None:
            self.build()

        if strategy == "zeros":
            return np.zeros(self._n_params)
        elif strategy == "identity":
            return rng.uniform(-0.1, 0.1, self._n_params)
        else:  # random
            return rng.uniform(0, 2 * np.pi, self._n_params)

    def __repr__(self) -> str:
        return f"QCNN(n_qubits={self.n_qubits}, n_conv_layers={self.n_conv_layers})"
