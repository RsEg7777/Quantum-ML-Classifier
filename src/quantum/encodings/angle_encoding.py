"""
Angle Encoding
==============

Rotation-based encoding that maps classical features to rotation angles.

This is the most common encoding scheme, using RX, RY, or RZ gates.
Each feature value θ_i is encoded as a rotation: R(θ_i)|0⟩.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import cirq
import numpy as np
import sympy

from src.quantum.encodings.base import BaseEncoding


class AngleEncoding(BaseEncoding):
    """
    Angle encoding using rotation gates.

    Maps each feature x_i to a rotation angle, encoding it as R_axis(x_i).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    rotation : str, default='Y'
        Rotation axis: 'X', 'Y', 'Z', or 'XYZ' for all three.
    scaling : float, default=1.0
        Scaling factor for angles (applied as x * scaling).
    n_features : int, optional
        Number of features. If None, equals n_qubits.
    repeated : bool, default=False
        If True and n_features > n_qubits, reuse qubits cyclically.

    Examples
    --------
    >>> encoding = AngleEncoding(n_qubits=4)
    >>> circuit = encoding.encode()  # Parametric circuit
    >>> print(circuit)

    >>> import numpy as np
    >>> data = np.array([0.1, 0.2, 0.3, 0.4])
    >>> resolved = encoding.resolve(circuit, data)
    """

    def __init__(
        self,
        n_qubits: int,
        rotation: Literal["X", "Y", "Z", "XYZ"] = "Y",
        scaling: float = 1.0,
        n_features: Optional[int] = None,
        repeated: bool = False,
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)
        self.rotation = rotation.upper()
        self.scaling = scaling
        self.repeated = repeated

        # Validate rotation
        valid_rotations = {"X", "Y", "Z", "XYZ"}
        if self.rotation not in valid_rotations:
            raise ValueError(f"rotation must be one of {valid_rotations}")

    def _get_rotation_gate(
        self,
        angle: Union[float, sympy.Symbol],
    ) -> cirq.Gate:
        """Get the rotation gate for the specified axis."""
        scaled_angle = angle * self.scaling

        if self.rotation == "X":
            return cirq.rx(scaled_angle)
        elif self.rotation == "Y":
            return cirq.ry(scaled_angle)
        elif self.rotation == "Z":
            return cirq.rz(scaled_angle)
        else:
            raise ValueError(f"Unknown rotation: {self.rotation}")

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """
        Create angle encoding circuit.

        Parameters
        ----------
        data : np.ndarray, optional
            Input data of shape (n_features,). If None, creates parametric circuit.
        symbols : sequence of sympy.Symbol, optional
            Custom symbols to use. If None, uses default x_0, x_1, ...

        Returns
        -------
        cirq.Circuit
            Encoding circuit.
        """
        circuit = cirq.Circuit()

        # Determine number of features
        if data is not None:
            n_features = len(data)
            params = data
        else:
            n_features = self.n_features or self.n_qubits
            params = symbols or self.symbols[:n_features]

        if self.rotation == "XYZ":
            # Use all three rotations for each feature
            circuit = self._encode_xyz(params, n_features)
        else:
            # Single rotation axis
            circuit = self._encode_single_axis(params, n_features)

        return circuit

    def _encode_single_axis(
        self,
        params: Sequence[Union[float, sympy.Symbol]],
        n_features: int,
    ) -> cirq.Circuit:
        """Encode using a single rotation axis."""
        circuit = cirq.Circuit()

        for i, param in enumerate(params):
            qubit_idx = i % self.n_qubits
            gate = self._get_rotation_gate(param)
            circuit.append(gate.on(self.qubits[qubit_idx]))

        return circuit

    def _encode_xyz(
        self,
        params: Sequence[Union[float, sympy.Symbol]],
        n_features: int,
    ) -> cirq.Circuit:
        """Encode using RX, RY, RZ sequence for each feature."""
        circuit = cirq.Circuit()

        for i, param in enumerate(params):
            qubit_idx = i % self.n_qubits
            qubit = self.qubits[qubit_idx]
            scaled = param * self.scaling

            # Apply RX, RY, RZ in sequence
            circuit.append(cirq.rx(scaled).on(qubit))
            circuit.append(cirq.ry(scaled).on(qubit))
            circuit.append(cirq.rz(scaled).on(qubit))

        return circuit

    def __repr__(self) -> str:
        return (
            f"AngleEncoding(n_qubits={self.n_qubits}, "
            f"rotation='{self.rotation}', scaling={self.scaling})"
        )


class DenseAngleEncoding(BaseEncoding):
    """
    Dense angle encoding that packs more features per qubit.

    Uses multiple rotation gates per qubit to encode more features
    than qubits available.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_features : int
        Number of features to encode.
    rotations_per_qubit : int, default=3
        Number of rotation gates per qubit (max 3 for RX, RY, RZ).

    Examples
    --------
    >>> encoding = DenseAngleEncoding(n_qubits=2, n_features=6)
    >>> # Encodes 6 features using 2 qubits with 3 rotations each
    """

    def __init__(
        self,
        n_qubits: int,
        n_features: int,
        rotations_per_qubit: int = 3,
    ):
        super().__init__(n_qubits=n_qubits, n_features=n_features)

        if rotations_per_qubit > 3:
            raise ValueError("rotations_per_qubit cannot exceed 3 (RX, RY, RZ)")

        self.rotations_per_qubit = rotations_per_qubit
        self.max_features = n_qubits * rotations_per_qubit

        if n_features > self.max_features:
            raise ValueError(
                f"Cannot encode {n_features} features with {n_qubits} qubits "
                f"and {rotations_per_qubit} rotations per qubit. "
                f"Maximum is {self.max_features}."
            )

        # Create symbols for all features
        self._symbols = [sympy.Symbol(f"x_{i}") for i in range(n_features)]

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create dense angle encoding circuit."""
        circuit = cirq.Circuit()

        params = data if data is not None else (symbols or self.symbols)
        rotation_gates = [cirq.rx, cirq.ry, cirq.rz]

        param_idx = 0
        for qubit_idx in range(self.n_qubits):
            qubit = self.qubits[qubit_idx]

            for rot_idx in range(self.rotations_per_qubit):
                if param_idx >= len(params):
                    break

                gate_fn = rotation_gates[rot_idx]
                circuit.append(gate_fn(params[param_idx]).on(qubit))
                param_idx += 1

        return circuit


class HadamardAngleEncoding(BaseEncoding):
    """
    Angle encoding with Hadamard preprocessing.

    Applies Hadamard gates before rotations to create superposition states,
    which can increase expressivity.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    rotation : str, default='Z'
        Rotation axis: 'X', 'Y', or 'Z'.

    Notes
    -----
    The circuit structure is: H - Rz(x) for each qubit, which creates
    states of the form cos(x/2)|0⟩ + sin(x/2)|1⟩.
    """

    def __init__(
        self,
        n_qubits: int,
        rotation: Literal["X", "Y", "Z"] = "Z",
    ):
        super().__init__(n_qubits=n_qubits)
        self.rotation = rotation.upper()

    def encode(
        self,
        data: Optional[np.ndarray] = None,
        symbols: Optional[Sequence[sympy.Symbol]] = None,
    ) -> cirq.Circuit:
        """Create Hadamard-angle encoding circuit."""
        circuit = cirq.Circuit()

        params = data if data is not None else (symbols or self.symbols)

        # Hadamard layer
        circuit.append(cirq.H.on_each(*self.qubits))

        # Rotation layer
        rotation_gate = {"X": cirq.rx, "Y": cirq.ry, "Z": cirq.rz}[self.rotation]

        for i, param in enumerate(params):
            if i >= self.n_qubits:
                break
            circuit.append(rotation_gate(param).on(self.qubits[i]))

        return circuit
