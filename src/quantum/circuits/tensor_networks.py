"""
Tensor Network Quantum Circuits
================================

Tensor network-inspired ansatze for variational quantum circuits.

- MPS (Matrix Product State): Linear chain of two-qubit gates
- TTN (Tree Tensor Network): Binary tree structure

These architectures have bounded entanglement and are more
resistant to barren plateaus than general circuits.

References
----------
Huggins et al., "Towards Quantum Machine Learning with
Tensor Networks", Quantum Sci. Technol. 4, 024001 (2019).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cirq
import numpy as np
import sympy


class MPSAnsatz:
    """
    Matrix Product State (MPS) ansatz.

    Linear chain of two-qubit unitary blocks, each parameterized.
    Bond dimension is implicitly 2 (single-qubit bond).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int, default=1
        Number of MPS sweeps (left-to-right passes).
    block_type : str, default='general'
        Type of two-qubit block:
        - 'general': Full U3 + entangling + U3
        - 'simple': RY + CNOT + RY

    Examples
    --------
    >>> mps = MPSAnsatz(n_qubits=6, n_layers=2)
    >>> circuit = mps.build()
    >>> print(f"Parameters: {mps.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        block_type: str = "general",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.block_type = block_type
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

        # Parameters per block
        if block_type == "general":
            self._params_per_block = 6  # 3 rotations per qubit
        else:
            self._params_per_block = 2  # 1 rotation per qubit

        self._n_blocks = (n_qubits - 1) * n_layers
        self._n_params = self._n_blocks * self._params_per_block
        self._symbols = [sympy.Symbol(f"mps_{i}") for i in range(self._n_params)]

    @property
    def n_params(self) -> int:
        """Number of trainable parameters."""
        return self._n_params

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Trainable parameters."""
        return self._symbols

    def _general_block(
        self,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        params: Sequence[sympy.Symbol],
    ) -> List[cirq.Operation]:
        """Full parameterized two-qubit block."""
        return [
            cirq.ry(params[0]).on(qubit1),
            cirq.rz(params[1]).on(qubit1),
            cirq.ry(params[2]).on(qubit2),
            cirq.CNOT(qubit1, qubit2),
            cirq.ry(params[3]).on(qubit1),
            cirq.rz(params[4]).on(qubit1),
            cirq.ry(params[5]).on(qubit2),
        ]

    def _simple_block(
        self,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        params: Sequence[sympy.Symbol],
    ) -> List[cirq.Operation]:
        """Simple parameterized two-qubit block."""
        return [
            cirq.ry(params[0]).on(qubit1),
            cirq.CNOT(qubit1, qubit2),
            cirq.ry(params[1]).on(qubit2),
        ]

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Build the MPS ansatz circuit.

        Parameters
        ----------
        symbols : sequence of sympy.Symbol, optional
            Custom parameter symbols.

        Returns
        -------
        cirq.Circuit
            MPS ansatz circuit.
        """
        circuit = cirq.Circuit()
        params = symbols or self._symbols
        param_idx = 0

        block_fn = (
            self._general_block if self.block_type == "general"
            else self._simple_block
        )

        for layer in range(self.n_layers):
            # Left-to-right sweep
            for i in range(self.n_qubits - 1):
                block_params = params[param_idx:param_idx + self._params_per_block]
                ops = block_fn(self.qubits[i], self.qubits[i + 1], block_params)
                circuit.append(ops)
                param_idx += self._params_per_block

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
        elif strategy == "identity":
            return rng.uniform(-0.1, 0.1, self._n_params)
        return rng.uniform(0, 2 * np.pi, self._n_params)

    def __repr__(self) -> str:
        return f"MPSAnsatz(n_qubits={self.n_qubits}, n_layers={self.n_layers})"


class TTNAnsatz:
    """
    Tree Tensor Network (TTN) ansatz.

    Binary tree structure that hierarchically entangles qubits.
    At each level, pairs of qubits are entangled, then one qubit
    from each pair carries information to the next level.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (should be a power of 2).
    block_type : str, default='general'
        Type of two-qubit block.

    Examples
    --------
    >>> ttn = TTNAnsatz(n_qubits=8)
    >>> circuit = ttn.build()
    >>> print(f"Parameters: {ttn.n_params}")
    """

    def __init__(
        self,
        n_qubits: int,
        block_type: str = "general",
    ):
        self.n_qubits = n_qubits
        self.block_type = block_type
        self.qubits = cirq.GridQubit.rect(1, n_qubits)

        if block_type == "general":
            self._params_per_block = 6
        else:
            self._params_per_block = 2

        # Count total blocks across all tree levels
        self._n_blocks = 0
        n = n_qubits
        while n > 1:
            self._n_blocks += n // 2
            n = n // 2

        self._n_params = self._n_blocks * self._params_per_block
        self._symbols = [sympy.Symbol(f"ttn_{i}") for i in range(self._n_params)]

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def symbols(self) -> List[sympy.Symbol]:
        return self._symbols

    def _general_block(
        self,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        params: Sequence[sympy.Symbol],
    ) -> List[cirq.Operation]:
        """Full parameterized two-qubit block."""
        return [
            cirq.ry(params[0]).on(qubit1),
            cirq.rz(params[1]).on(qubit1),
            cirq.ry(params[2]).on(qubit2),
            cirq.CNOT(qubit1, qubit2),
            cirq.ry(params[3]).on(qubit1),
            cirq.rz(params[4]).on(qubit1),
            cirq.ry(params[5]).on(qubit2),
        ]

    def _simple_block(
        self,
        qubit1: cirq.GridQubit,
        qubit2: cirq.GridQubit,
        params: Sequence[sympy.Symbol],
    ) -> List[cirq.Operation]:
        """Simple parameterized two-qubit block."""
        return [
            cirq.ry(params[0]).on(qubit1),
            cirq.CNOT(qubit1, qubit2),
            cirq.ry(params[1]).on(qubit2),
        ]

    def build(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Build the TTN ansatz circuit.

        Returns
        -------
        cirq.Circuit
            TTN ansatz circuit with binary tree structure.
        """
        circuit = cirq.Circuit()
        params = symbols or self._symbols
        param_idx = 0

        block_fn = (
            self._general_block if self.block_type == "general"
            else self._simple_block
        )

        # Build tree levels
        active_qubits = list(self.qubits)

        while len(active_qubits) > 1:
            next_level = []
            for i in range(0, len(active_qubits) - 1, 2):
                q1 = active_qubits[i]
                q2 = active_qubits[i + 1]

                block_params = params[param_idx:param_idx + self._params_per_block]
                ops = block_fn(q1, q2, block_params)
                circuit.append(ops)
                param_idx += self._params_per_block

                # Keep second qubit for next level (carries info)
                next_level.append(q2)

            # Handle odd qubit
            if len(active_qubits) % 2 == 1:
                next_level.append(active_qubits[-1])

            active_qubits = next_level

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
        elif strategy == "identity":
            return rng.uniform(-0.1, 0.1, self._n_params)
        return rng.uniform(0, 2 * np.pi, self._n_params)

    def __repr__(self) -> str:
        return f"TTNAnsatz(n_qubits={self.n_qubits}, blocks={self._n_blocks})"
