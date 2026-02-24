"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for all tests.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def seed():
    """Random seed for reproducibility."""
    return 42


@pytest.fixture(scope="session")
def n_qubits():
    """Default number of qubits for tests."""
    return 4


@pytest.fixture(scope="session")
def n_layers():
    """Default number of layers for tests."""
    return 2


@pytest.fixture
def sample_data(seed):
    """Generate sample data for testing."""
    np.random.seed(seed)
    n_samples = 100
    n_features = 4

    X = np.random.uniform(0, np.pi, (n_samples, n_features)).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int32)

    return X, y


@pytest.fixture
def iris_data():
    """Load Iris dataset for integration tests."""
    from src.data.loaders import load_iris

    return load_iris(test_size=0.2, seed=42)


@pytest.fixture
def small_sample_data(seed):
    """Small dataset for quick tests."""
    np.random.seed(seed)
    n_samples = 20
    n_features = 4

    X = np.random.uniform(0, np.pi, (n_samples, n_features)).astype(np.float32)
    y = np.random.randint(0, 2, n_samples).astype(np.int32)

    return X, y


@pytest.fixture
def cirq_qubits(n_qubits):
    """Create Cirq qubits."""
    import cirq

    return cirq.GridQubit.rect(1, n_qubits)


@pytest.fixture
def sympy_symbols(n_qubits):
    """Create sympy symbols for circuit parameters."""
    import sympy

    return [sympy.Symbol(f"θ_{i}") for i in range(n_qubits * 2)]


# Markers configuration
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "hardware: marks tests requiring quantum hardware")
    config.addinivalue_line("markers", "benchmark: marks benchmark tests")
