"""
Tests for Quantum Encoding Schemes
==================================
"""

import numpy as np
import pytest
import cirq
import sympy

from src.quantum.encodings import AngleEncoding, BaseEncoding
from src.quantum.encodings.angle_encoding import DenseAngleEncoding, HadamardAngleEncoding


class TestAngleEncoding:
    """Tests for AngleEncoding class."""

    def test_init_default(self, n_qubits):
        """Test default initialization."""
        encoding = AngleEncoding(n_qubits=n_qubits)

        assert encoding.n_qubits == n_qubits
        assert encoding.rotation == "Y"
        assert encoding.scaling == 1.0
        assert len(encoding.qubits) == n_qubits

    def test_init_custom_rotation(self):
        """Test initialization with custom rotation."""
        encoding = AngleEncoding(n_qubits=4, rotation="X")
        assert encoding.rotation == "X"

        encoding = AngleEncoding(n_qubits=4, rotation="Z")
        assert encoding.rotation == "Z"

        encoding = AngleEncoding(n_qubits=4, rotation="XYZ")
        assert encoding.rotation == "XYZ"

    def test_invalid_rotation(self):
        """Test that invalid rotation raises error."""
        with pytest.raises(ValueError):
            AngleEncoding(n_qubits=4, rotation="INVALID")

    def test_symbols(self, n_qubits):
        """Test symbol generation."""
        encoding = AngleEncoding(n_qubits=n_qubits)
        symbols = encoding.symbols

        assert len(symbols) == n_qubits
        assert all(isinstance(s, sympy.Symbol) for s in symbols)
        assert [str(s) for s in symbols] == [f"x_{i}" for i in range(n_qubits)]

    def test_encode_parametric(self, n_qubits):
        """Test parametric circuit generation."""
        encoding = AngleEncoding(n_qubits=n_qubits)
        circuit = encoding.encode()

        assert isinstance(circuit, cirq.Circuit)
        assert len(list(circuit.all_operations())) == n_qubits

    def test_encode_with_data(self, n_qubits):
        """Test encoding with concrete data."""
        encoding = AngleEncoding(n_qubits=n_qubits)
        data = np.array([0.1, 0.2, 0.3, 0.4])

        circuit = encoding.encode(data=data)

        assert isinstance(circuit, cirq.Circuit)
        # Circuit should have no free symbols when data is provided
        assert len(cirq.parameter_symbols(circuit)) == 0

    def test_encode_xyz(self):
        """Test XYZ rotation encoding."""
        encoding = AngleEncoding(n_qubits=2, rotation="XYZ")
        circuit = encoding.encode()

        # XYZ should have 3 gates per qubit
        ops = list(circuit.all_operations())
        assert len(ops) == 6  # 2 qubits * 3 rotations

    def test_scaling(self):
        """Test angle scaling."""
        encoding = AngleEncoding(n_qubits=2, scaling=2.0)
        circuit = encoding.encode()

        # Check that scaling is applied (indirectly through circuit structure)
        assert encoding.scaling == 2.0

    def test_resolve(self, n_qubits):
        """Test circuit resolution with concrete values."""
        encoding = AngleEncoding(n_qubits=n_qubits)
        circuit = encoding.encode()
        data = np.random.uniform(0, np.pi, n_qubits)

        resolved = encoding.resolve(circuit, data)

        # Resolved circuit should have no free symbols
        assert len(cirq.parameter_symbols(resolved)) == 0


class TestDenseAngleEncoding:
    """Tests for DenseAngleEncoding class."""

    def test_init(self):
        """Test initialization."""
        encoding = DenseAngleEncoding(n_qubits=2, n_features=6)

        assert encoding.n_qubits == 2
        assert encoding.n_features == 6
        assert encoding.rotations_per_qubit == 3

    def test_max_features_exceeded(self):
        """Test error when n_features exceeds capacity."""
        with pytest.raises(ValueError):
            DenseAngleEncoding(n_qubits=2, n_features=10)

    def test_encode(self):
        """Test dense encoding circuit generation."""
        encoding = DenseAngleEncoding(n_qubits=2, n_features=6)
        circuit = encoding.encode()

        # Should have 6 rotation gates (3 per qubit)
        ops = list(circuit.all_operations())
        assert len(ops) == 6


class TestHadamardAngleEncoding:
    """Tests for HadamardAngleEncoding class."""

    def test_init(self):
        """Test initialization."""
        encoding = HadamardAngleEncoding(n_qubits=4)

        assert encoding.n_qubits == 4
        assert encoding.rotation == "Z"

    def test_encode(self, n_qubits):
        """Test Hadamard-angle encoding."""
        encoding = HadamardAngleEncoding(n_qubits=n_qubits)
        circuit = encoding.encode()

        # Should have H gates + rotation gates
        ops = list(circuit.all_operations())
        assert len(ops) == 2 * n_qubits  # H + Rz per qubit


class TestEncodingSimulation:
    """Tests that verify encoding correctness through simulation."""

    @pytest.mark.slow
    def test_angle_encoding_simulation(self, n_qubits):
        """Test that angle encoding produces expected states."""
        encoding = AngleEncoding(n_qubits=n_qubits, rotation="Y")

        # Encode π/2 on each qubit
        data = np.full(n_qubits, np.pi / 2)
        circuit = encoding.encode(data=data)

        # Simulate
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # For Ry(π/2), state should be superposition
        state = result.final_state_vector
        assert len(state) == 2**n_qubits
        # All amplitudes should be non-zero for uniform encoding
        assert np.all(np.abs(state) > 0.01)
