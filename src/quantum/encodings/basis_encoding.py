"""
Basis Encoding
==============

Encodes classical data into computational basis states.

For binary data, each bit is directly mapped to a qubit state.
For continuous data, discretization is applied first.
"""

from __future__ import annotations

from typing import Optional, Sequence, Literal

import cirq
import numpy as np
import sympy

from src.quantum.encodings.base import BaseEncoding


class BasisEncoding(BaseEncoding):
    """
    Basis encoding for computational basis states.

    Maps binary/integer data directly to qubit states using X gates.
    For a binary string b = (b_0, b_1, ..., b_{n-1}):
    |0...0⟩ → |b_0 b_1 ... b_{n-1}⟩

    Parameters
    ----------
    n_qubits : int
        Number of qubits (determines maximum encodable value).
    encoding_type : str, default='binary'
        How to interpret input data:
        - 'binary': Input is already binary (0/1)
        - 'integer': Input is integer, converted to binary
        - 'threshold': Input is continuous, thresholded at 0.5

    Examples
    --------
    >>> encoding = BasisEncoding(n_qubits=4)
    >>> # Encode binary [1, 0, 1, 1] = state |1011⟩
    >>> circuit = encoding.encode(data=np.array([1, 0, 1, 1]))
    """

    def __init__(
        self,
        n_qubits: int,
        encoding_type: Literal["binary", "integer", "threshold"] = "binary",
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_qubits)
        self.encoding_type = encoding_type

    def _to_binary(self, data: np.ndarray) -> np.ndarray:
        """Convert data to binary representation."""
        if self.encoding_type == "binary":
            return np.round(data).astype(int) % 2

        elif self.encoding_type == "integer":
            # Take first element as integer, convert to binary
            if len(data) == 1:
                value = int(data[0])
                binary = []
                for _ in range(self.n_qubits):
                    binary.append(value % 2)
                    value //= 2
                return np.array(binary[::-1])
            else:
                return np.round(data).astype(int) % 2

        elif self.encoding_type == "threshold":
            return (data > 0.5).astype(int)

        return np.zeros(self.n_qubits, dtype=int)

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Create basis encoding circuit.

        Parameters
        ----------
        data : np.ndarray, optional
            Input data to encode. Must be provided for basis encoding.
        symbols : sequence of sympy.Symbol, optional
            Not used for basis encoding (encoding is not parametric).

        Returns
        -------
        cirq.Circuit
            Circuit with X gates where needed to prepare basis state.
        """
        circuit = cirq.Circuit()

        if data is not None:
            binary = self._to_binary(np.array(data))

            # Apply X gate where bit is 1
            for i, bit in enumerate(binary):
                if i < self.n_qubits and bit == 1:
                    circuit.append(cirq.X.on(self.qubits[i]))

        # Note: For parametric version, basis encoding doesn't make sense
        # as X gates are not parameterized. Return empty circuit if no data.

        return circuit


class BinaryAmplitudeEncoding(BaseEncoding):
    """
    Encodes binary strings using uniform superposition over marked states.

    Creates a state where all specified binary patterns have equal amplitude.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Examples
    --------
    >>> encoding = BinaryAmplitudeEncoding(n_qubits=3)
    >>> # Create equal superposition of |001⟩, |010⟩, |100⟩
    >>> circuit = encoding.encode(data=np.array([[0,0,1], [0,1,0], [1,0,0]]))
    """

    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits)

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create binary amplitude encoding circuit."""
        circuit = cirq.Circuit()

        if data is None:
            # Default: uniform superposition over all basis states
            circuit.append(cirq.H.on_each(*self.qubits))
            return circuit

        data = np.atleast_2d(data)
        n_patterns = len(data)

        if n_patterns == 1:
            # Single pattern: just basis encoding
            for i, bit in enumerate(data[0]):
                if i < self.n_qubits and bit == 1:
                    circuit.append(cirq.X.on(self.qubits[i]))
        else:
            # Multiple patterns: create superposition
            # This is a simplified version - exact implementation
            # would require Grover-like preparation
            circuit.append(cirq.H.on_each(*self.qubits))

        return circuit


class ThermometerEncoding(BaseEncoding):
    """
    Thermometer encoding for ordered/continuous data.

    Maps a value x ∈ [0, 1] to a thermometer code where
    the first k qubits are |1⟩ and remaining are |0⟩,
    where k = floor(x * n_qubits).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (resolution of encoding).
    n_features : int, optional
        Number of features. Each feature uses n_qubits/n_features qubits.

    Examples
    --------
    >>> encoding = ThermometerEncoding(n_qubits=4)
    >>> # x=0.75 → |1110⟩ (3 out of 4 qubits are 1)
    >>> circuit = encoding.encode(data=np.array([0.75]))
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int = 1,
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)
        self.qubits_per_feature = n_qubits // n_features

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create thermometer encoding circuit."""
        circuit = cirq.Circuit()

        if data is None:
            # Parametric version not supported
            return circuit

        data = np.array(data).flatten()

        for f, value in enumerate(data[:self.n_features]):
            # Number of 1s for this feature
            k = int(np.clip(value, 0, 1) * self.qubits_per_feature)

            # Apply X gates to first k qubits for this feature
            start_idx = f * self.qubits_per_feature
            for i in range(k):
                qubit_idx = start_idx + i
                if qubit_idx < self.n_qubits:
                    circuit.append(cirq.X.on(self.qubits[qubit_idx]))

        return circuit


class OneHotEncoding(BaseEncoding):
    """
    One-hot encoding for categorical data.

    Maps a categorical value k ∈ {0, 1, ..., n-1} to state |0...010...0⟩
    where only qubit k is in state |1⟩.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (= number of categories).

    Examples
    --------
    >>> encoding = OneHotEncoding(n_qubits=4)
    >>> # Category 2 → |0010⟩
    >>> circuit = encoding.encode(data=np.array([2]))
    """

    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits, n_features=1)

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create one-hot encoding circuit."""
        circuit = cirq.Circuit()

        if data is None:
            return circuit

        category = int(data[0]) % self.n_qubits
        circuit.append(cirq.X.on(self.qubits[category]))

        return circuit


class GrayCodeEncoding(BaseEncoding):
    """
    Gray code encoding for ordinal data.

    Uses Gray code representation where adjacent values differ
    by exactly one bit, reducing Hamming distance for nearby values.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.

    Examples
    --------
    >>> encoding = GrayCodeEncoding(n_qubits=3)
    >>> # Value 5 in Gray code: 5 → 7 XOR 3 = 100 → |100⟩
    """

    def __init__(self, n_qubits: int):
        super().__init__(n_qubits=n_qubits)

    def _to_gray(self, n: int) -> int:
        """Convert integer to Gray code."""
        return n ^ (n >> 1)

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create Gray code encoding circuit."""
        circuit = cirq.Circuit()

        if data is None:
            return circuit

        value = int(data[0])
        gray_value = self._to_gray(value)

        # Convert to binary and apply X gates
        for i in range(self.n_qubits):
            if (gray_value >> i) & 1:
                circuit.append(cirq.X.on(self.qubits[self.n_qubits - 1 - i]))

        return circuit
