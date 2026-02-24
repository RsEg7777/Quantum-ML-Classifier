"""
Base Encoding Class
===================

Abstract base class for all quantum encoding schemes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

import cirq
import numpy as np
import sympy


class BaseEncoding(ABC):
    """
    Abstract base class for quantum data encoding schemes.

    All encoding implementations should inherit from this class and
    implement the `encode` method.

    Parameters
    ----------
    n_qubits : int
        Number of qubits to use for encoding.
    n_features : int, optional
        Number of input features. If None, determined from data.

    Attributes
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int or None
        Number of features to encode.
    qubits : list of cirq.GridQubit
        Qubit objects used in the circuit.
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.n_features = n_features
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        self._symbols: Optional[List[sympy.Symbol]] = None

    @property
    def symbols(self) -> List[sympy.Symbol]:
        """Get the symbolic parameters for this encoding."""
        if self._symbols is None:
            n_params = self.n_features or self.n_qubits
            self._symbols = [sympy.Symbol(f"x_{i}") for i in range(n_params)]
        return self._symbols

    @abstractmethod
    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Create an encoding circuit.

        Parameters
        ----------
        data : np.ndarray, optional
            Input data to encode. If None, returns parametric circuit.
        symbols : sequence of sympy.Symbol, optional
            Symbolic parameters. If None, uses default symbols.

        Returns
        -------
        cirq.Circuit
            Quantum circuit that encodes the data.
        """
        pass

    def get_circuit(
        self,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Get the parametric encoding circuit.

        Parameters
        ----------
        symbols : sequence of sympy.Symbol, optional
            Symbolic parameters to use.

        Returns
        -------
        cirq.Circuit
            Parametric encoding circuit.
        """
        return self.encode(data=None, symbols=symbols)

    def resolve(
        self,
        circuit: cirq.Circuit,
        data: np.ndarray,
    ) -> cirq.Circuit:
        """
        Resolve a parametric circuit with concrete data values.

        Parameters
        ----------
        circuit : cirq.Circuit
            Parametric circuit to resolve.
        data : np.ndarray
            Data values to substitute.

        Returns
        -------
        cirq.Circuit
            Resolved circuit with concrete values.
        """
        resolver = dict(zip(self.symbols, data))
        return cirq.resolve_parameters(circuit, resolver)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_qubits={self.n_qubits})"

    def __str__(self) -> str:
        return self.__repr__()


class RepeatedEncoding(BaseEncoding):
    """
    Encoding that repeats a base encoding multiple times.

    This can increase expressivity by re-encoding data at different
    points in the circuit.

    Parameters
    ----------
    base_encoding : BaseEncoding
        The base encoding to repeat.
    n_reps : int
        Number of repetitions.
    """

    def __init__(
        self,
        base_encoding: BaseEncoding,
        n_reps: int = 2,
    ):
        super().__init__(
            n_qubits=base_encoding.n_qubits,
            n_features=base_encoding.n_features,
        )
        self.base_encoding = base_encoding
        self.n_reps = n_reps

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create repeated encoding circuit."""
        circuit = cirq.Circuit()
        base_circuit = self.base_encoding.encode(data, symbols)

        for _ in range(self.n_reps):
            circuit += base_circuit

        return circuit


class CompositeEncoding(BaseEncoding):
    """
    Combine multiple encoding schemes sequentially.

    Parameters
    ----------
    encodings : list of BaseEncoding
        Encodings to combine.
    """

    def __init__(self, encodings: List[BaseEncoding]):
        if not encodings:
            raise ValueError("At least one encoding required")

        # All encodings must have same n_qubits
        n_qubits = encodings[0].n_qubits
        if not all(e.n_qubits == n_qubits for e in encodings):
            raise ValueError("All encodings must have same n_qubits")

        super().__init__(n_qubits=n_qubits)
        self.encodings = encodings

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create composite encoding circuit."""
        circuit = cirq.Circuit()

        for encoding in self.encodings:
            circuit += encoding.encode(data, symbols)

        return circuit
