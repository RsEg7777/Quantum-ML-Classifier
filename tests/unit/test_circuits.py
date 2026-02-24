"""
Tests for Quantum Circuits
==========================
"""

import numpy as np
import pytest
import cirq
import sympy

from src.quantum.circuits import VQC, HardwareEfficientAnsatz, StronglyEntanglingAnsatz


class TestHardwareEfficientAnsatz:
    """Tests for HardwareEfficientAnsatz class."""

    def test_init_default(self, n_qubits, n_layers):
        """Test default initialization."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)

        assert ansatz.n_qubits == n_qubits
        assert ansatz.n_layers == n_layers
        assert ansatz.entanglement == "linear"
        assert ansatz.rotation_gates == ("RY", "RZ")

    def test_n_params(self, n_qubits, n_layers):
        """Test parameter count calculation."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)

        # 2 rotations per qubit per layer
        expected = 2 * n_qubits * n_layers
        assert ansatz.n_params == expected

    def test_symbols(self, n_qubits, n_layers):
        """Test symbol generation."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        symbols = ansatz.symbols

        assert len(symbols) == ansatz.n_params
        assert all(isinstance(s, sympy.Symbol) for s in symbols)

    def test_build(self, n_qubits, n_layers):
        """Test circuit building."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        circuit = ansatz.build()

        assert isinstance(circuit, cirq.Circuit)
        # Check that circuit has operations
        assert len(list(circuit.all_operations())) > 0

    @pytest.mark.parametrize("entanglement", ["linear", "full", "circular", "star"])
    def test_entanglement_topologies(self, n_qubits, entanglement):
        """Test different entanglement topologies."""
        ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits,
            n_layers=2,
            entanglement=entanglement,
        )
        circuit = ansatz.build()

        assert isinstance(circuit, cirq.Circuit)

    def test_custom_rotation_gates(self, n_qubits, n_layers):
        """Test custom rotation gates."""
        ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_gates=("RX", "RY", "RZ"),
        )

        # Should have 3 rotations per qubit per layer
        expected = 3 * n_qubits * n_layers
        assert ansatz.n_params == expected

    def test_invalid_rotation_gate(self, n_qubits):
        """Test error on invalid rotation gate."""
        with pytest.raises(ValueError):
            HardwareEfficientAnsatz(
                n_qubits=n_qubits,
                n_layers=1,
                rotation_gates=("INVALID",),
            )

    def test_initial_params_random(self, n_qubits, n_layers, seed):
        """Test random parameter initialization."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        params = ansatz.get_initial_params(strategy="random", seed=seed)

        assert params.shape == (ansatz.n_params,)
        assert np.all(params >= 0)
        assert np.all(params < 2 * np.pi)

    def test_initial_params_zeros(self, n_qubits, n_layers):
        """Test zero parameter initialization."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        params = ansatz.get_initial_params(strategy="zeros")

        assert np.allclose(params, 0)

    def test_initial_params_identity(self, n_qubits, n_layers, seed):
        """Test identity (near-zero) parameter initialization."""
        ansatz = HardwareEfficientAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        params = ansatz.get_initial_params(strategy="identity", seed=seed)

        assert params.shape == (ansatz.n_params,)
        assert np.all(np.abs(params) < 0.2)


class TestStronglyEntanglingAnsatz:
    """Tests for StronglyEntanglingAnsatz class."""

    def test_init(self, n_qubits, n_layers):
        """Test initialization."""
        ansatz = StronglyEntanglingAnsatz(n_qubits=n_qubits, n_layers=n_layers)

        assert ansatz.n_qubits == n_qubits
        assert ansatz.n_layers == n_layers

    def test_n_params(self, n_qubits, n_layers):
        """Test parameter count."""
        ansatz = StronglyEntanglingAnsatz(n_qubits=n_qubits, n_layers=n_layers)

        # 3 rotations per qubit per layer
        expected = 3 * n_qubits * n_layers
        assert ansatz.n_params == expected

    def test_build(self, n_qubits, n_layers):
        """Test circuit building."""
        ansatz = StronglyEntanglingAnsatz(n_qubits=n_qubits, n_layers=n_layers)
        circuit = ansatz.build()

        assert isinstance(circuit, cirq.Circuit)


class TestVQC:
    """Tests for VQC class."""

    def test_init_default(self, n_qubits, n_layers):
        """Test default initialization."""
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)

        assert vqc.n_qubits == n_qubits
        assert vqc.n_layers == n_layers
        assert isinstance(vqc.ansatz, HardwareEfficientAnsatz)

    def test_init_strongly_entangling(self, n_qubits, n_layers):
        """Test initialization with strongly entangling ansatz."""
        vqc = VQC(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz="strongly_entangling",
        )

        assert isinstance(vqc.ansatz, StronglyEntanglingAnsatz)

    def test_init_custom_ansatz(self, n_qubits, n_layers):
        """Test initialization with custom ansatz instance."""
        custom_ansatz = HardwareEfficientAnsatz(
            n_qubits=n_qubits,
            n_layers=n_layers,
            entanglement="full",
        )
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers, ansatz=custom_ansatz)

        assert vqc.ansatz is custom_ansatz

    def test_build(self, n_qubits, n_layers):
        """Test circuit building."""
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)
        circuit = vqc.build()

        assert isinstance(circuit, cirq.Circuit)

    def test_get_circuit_caching(self, n_qubits, n_layers):
        """Test that get_circuit caches the circuit."""
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)

        circuit1 = vqc.get_circuit()
        circuit2 = vqc.get_circuit()

        assert circuit1 is circuit2

    def test_resolve(self, n_qubits, n_layers, seed):
        """Test circuit resolution."""
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)
        params = vqc.get_initial_params(seed=seed)

        resolved = vqc.resolve(params)

        # Resolved circuit should have no free symbols
        assert len(cirq.parameter_symbols(resolved)) == 0

    def test_repr(self, n_qubits, n_layers):
        """Test string representation."""
        vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)

        repr_str = repr(vqc)
        assert "VQC" in repr_str
        assert str(n_qubits) in repr_str
        assert str(n_layers) in repr_str


class TestVQCSimulation:
    """Simulation tests for VQC."""

    @pytest.mark.slow
    def test_simulation_runs(self, n_qubits, seed):
        """Test that VQC circuit can be simulated."""
        vqc = VQC(n_qubits=n_qubits, n_layers=2)
        params = vqc.get_initial_params(seed=seed)
        circuit = vqc.resolve(params)

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        assert result.final_state_vector is not None
        assert len(result.final_state_vector) == 2**n_qubits

    @pytest.mark.slow
    def test_different_params_different_states(self, n_qubits, seed):
        """Test that different parameters produce different states."""
        vqc = VQC(n_qubits=n_qubits, n_layers=2)

        params1 = vqc.get_initial_params(seed=seed)
        params2 = vqc.get_initial_params(seed=seed + 1)

        circuit1 = vqc.resolve(params1)
        circuit2 = vqc.resolve(params2)

        simulator = cirq.Simulator()
        state1 = simulator.simulate(circuit1).final_state_vector
        state2 = simulator.simulate(circuit2).final_state_vector

        # States should be different
        assert not np.allclose(state1, state2)
