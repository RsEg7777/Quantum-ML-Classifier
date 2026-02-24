"""
Experiment 1.1: Circuit Depth Scaling
=====================================

Measures how VQC performance and expressibility change with circuit depth.
Sweeps n_layers from 1 to 10 on the Iris dataset using Cirq simulation.
"""

from __future__ import annotations

from typing import Any, Dict

import cirq
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from src.data.loaders.classification_datasets import load_iris
from src.quantum.circuits.vqc import VQC
from src.quantum.encodings.angle_encoding import AngleEncoding


def _build_quantum_features(
    X: np.ndarray,
    n_qubits: int,
    n_layers: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract quantum features by simulating a VQC on each sample.

    For each input x, builds encoding(x) + ansatz(theta) circuit,
    simulates, and collects Z-expectation values as features.
    """
    rng = np.random.RandomState(seed)
    encoding = AngleEncoding(n_qubits=n_qubits, rotation="Y")
    vqc = VQC(n_qubits=n_qubits, n_layers=n_layers, entanglement="linear")
    params = vqc.get_initial_params(seed=seed)
    simulator = cirq.Simulator()

    qubits = encoding.qubits
    z_ops = [cirq.Z(q) for q in qubits]

    features = np.zeros((len(X), n_qubits))
    for i, x in enumerate(X):
        enc_circuit = encoding.encode(data=x[:n_qubits])
        ansatz_circuit = vqc.resolve(params)
        full_circuit = enc_circuit + ansatz_circuit

        result = simulator.simulate(full_circuit)
        sv = result.final_state_vector

        for q_idx, op in enumerate(z_ops):
            exp_val = op.on(qubits[q_idx])
            # <Z> = <psi|Z|psi>
            dm = np.outer(sv, sv.conj())
            z_matrix = cirq.unitary(cirq.Z)
            n = len(qubits)
            full_z = np.eye(1)
            for j in range(n):
                if j == q_idx:
                    full_z = np.kron(full_z, z_matrix)
                else:
                    full_z = np.kron(full_z, np.eye(2))
            features[i, q_idx] = np.real(np.trace(dm @ full_z))

    return features


def run_depth_scaling(config: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """
    Run depth scaling experiment.

    Uses quantum-enhanced features + classical SVM for fast evaluation.
    """
    n_qubits = config.get("n_qubits", 4)
    n_layers = config.get("n_layers", 3)

    # Load data
    data = load_iris(seed=seed, n_features=n_qubits)

    # Extract quantum features
    X_train_q = _build_quantum_features(
        data.X_train, n_qubits, n_layers, seed=seed
    )
    X_test_q = _build_quantum_features(
        data.X_test, n_qubits, n_layers, seed=seed
    )

    # Train SVM on quantum features
    svm = SVC(kernel="rbf", random_state=seed)
    svm.fit(X_train_q, data.y_train)

    train_acc = accuracy_score(data.y_train, svm.predict(X_train_q))
    test_acc = accuracy_score(data.y_test, svm.predict(X_test_q))

    # Circuit stats
    vqc = VQC(n_qubits=n_qubits, n_layers=n_layers)
    circuit = vqc.build()

    return {
        "n_layers": n_layers,
        "n_qubits": n_qubits,
        "n_params": vqc.n_params,
        "circuit_depth": len(circuit),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
    }


def register_experiments(runner) -> None:
    """Register depth scaling sweep with the experiment runner."""
    base_config = {"n_qubits": 4}

    for depth in [1, 2, 3, 5]:
        config = {**base_config, "n_layers": depth}
        runner.register(
            f"depth_scaling_L{depth}",
            lambda cfg, seed=42, _c=config: run_depth_scaling({**_c, **cfg}, seed),
        )
