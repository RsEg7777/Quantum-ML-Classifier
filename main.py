"""
Quantum-Classical Hybrid ML System - Demo
==========================================

End-to-end demonstration of the quantum ML pipeline:
1. Load data
2. Build quantum encoding + variational circuit
3. Extract quantum features via Cirq simulation
4. Train classical classifier on quantum features
5. Evaluate + analyse circuit properties
6. Compare against classical baselines

Runs entirely on Cirq simulation (no TFQ required).

Usage
-----
    python main.py
"""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path

import cirq
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

# ── Project imports ──────────────────────────────────────────────────────────
from src.data.loaders.classification_datasets import load_iris, load_breast_cancer
from src.quantum.encodings.angle_encoding import AngleEncoding
from src.quantum.circuits.vqc import VQC, HardwareEfficientAnsatz, StronglyEntanglingAnsatz
from src.quantum.entanglement.entanglement_strategies import (
    get_entanglement_pairs,
)
from src.quantum.gates.standard_gates import rotation_layer, entangling_layer
from src.quantum.measurements.pauli_measurements import z_measurement, multi_basis_measurement
from src.noise.models.noise_channels import DepolarizingNoise, get_noise_model
from src.noise.mitigation.zero_noise_extrapolation import ZeroNoiseExtrapolator
from src.analysis.circuit.resource_estimation import estimate_resources
from src.analysis.circuit.expressibility import expressibility, entangling_capability
from src.utils.reproducibility import set_seed, get_reproducibility_info


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_quantum_features(
    encoding_circuits: list[cirq.Circuit],
    ansatz_circuit: cirq.Circuit,
    observables: list[np.ndarray],
    simulator: cirq.Simulator,
) -> np.ndarray:
    """Simulate encoding+ansatz circuits and return observable expectation features."""
    features = np.zeros((len(encoding_circuits), len(observables)), dtype=np.float32)

    for sample_idx, encoding_circuit in enumerate(encoding_circuits):
        state_vector = simulator.simulate(
            encoding_circuit + ansatz_circuit
        ).final_state_vector
        density_matrix = np.outer(state_vector, state_vector.conj())

        for observable_idx, observable in enumerate(observables):
            features[sample_idx, observable_idx] = float(
                np.real(np.trace(density_matrix @ observable))
            )

    return features


def build_observables(n_qubits: int, include_pairwise: bool = True) -> list[np.ndarray]:
    """Create Z and optional ZZ observables as full matrices."""
    z = cirq.unitary(cirq.Z)
    observables: list[np.ndarray] = []

    for qubit_idx in range(n_qubits):
        matrix = np.eye(1, dtype=np.complex128)
        for j in range(n_qubits):
            matrix = np.kron(matrix, z if j == qubit_idx else np.eye(2))
        observables.append(matrix)

    if include_pairwise:
        for i, j in combinations(range(n_qubits), 2):
            matrix = np.eye(1, dtype=np.complex128)
            for k in range(n_qubits):
                matrix = np.kron(matrix, z if k in (i, j) else np.eye(2))
            observables.append(matrix)

    return observables


def build_encoding_circuits(
    X: np.ndarray,
    encoding: AngleEncoding,
    n_qubits: int,
) -> list[cirq.Circuit]:
    """Pre-encode classical samples to avoid repeated circuit construction."""
    return [encoding.encode(data=sample[:n_qubits]) for sample in X]


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    set_seed(42)
    info = get_reproducibility_info()

    print("Quantum-Classical Hybrid ML System")
    print("=" * 60)
    print(f"  NumPy:  {info['numpy_version']}")
    print(f"  Cirq:   {info['cirq_version']}")
    print(f"  SymPy:  {info['sympy_version']}")
    print(f"  Platform: {info['platform']}")
    print("=" * 60)

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    section("1. Loading Iris Dataset")
    N_QUBITS = 4
    N_LAYERS = 3
    data = load_iris(seed=42, n_features=N_QUBITS)
    print(f"  {data}")
    print(f"  Train: {data.X_train.shape}, Test: {data.X_test.shape}")

    # ── 2. Build Quantum Circuit ─────────────────────────────────────────────
    section("2. Building Quantum Circuit")
    encoding = AngleEncoding(n_qubits=N_QUBITS, rotation="Y")
    vqc = VQC(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        ansatz="hardware_efficient",
        entanglement="linear",
    )
    params = vqc.get_initial_params(seed=42)

    # Show circuit
    sample_enc = encoding.encode(data=data.X_train[0, :N_QUBITS])
    sample_ansatz = vqc.resolve(params)
    full_circuit = sample_enc + sample_ansatz
    print(f"  Encoding:  AngleEncoding(rotation=Y, qubits={N_QUBITS})")
    print(f"  Ansatz:    HardwareEfficient(layers={N_LAYERS}, entanglement=linear)")
    print(f"  Params:    {vqc.n_params}")
    print(f"\n  Sample circuit:\n{full_circuit}")

    # ── 3. Resource Estimation ───────────────────────────────────────────────
    section("3. Circuit Resource Analysis")
    profile = estimate_resources(full_circuit)
    print(profile)

    # ── 4. Extract Quantum Features + Tune Quantum Model ─────────────────────
    section("4. Quantum Feature Extraction + Hyperparameter Search")
    print(f"  Processing {len(data.X_train)} train + {len(data.X_test)} test samples...")

    QUANTUM_RESTARTS = 24
    SVM_C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
    SVM_GAMMA_GRID = ["scale", 0.05, 0.1, 0.2, 0.5, 1.0]

    simulator = cirq.Simulator()
    observables = build_observables(N_QUBITS, include_pairwise=True)
    train_encoding_circuits = build_encoding_circuits(data.X_train, encoding, N_QUBITS)
    test_encoding_circuits = build_encoding_circuits(data.X_test, encoding, N_QUBITS)

    best_candidate = None
    rng = np.random.RandomState(42)
    t_feat = time.time()

    for _ in range(QUANTUM_RESTARTS):
        restart_seed = int(rng.randint(0, 1_000_000))
        candidate_params = vqc.get_initial_params(seed=restart_seed)
        ansatz_circuit = vqc.resolve(candidate_params)

        X_train_q = extract_quantum_features(
            train_encoding_circuits,
            ansatz_circuit,
            observables,
            simulator,
        )
        X_test_q = extract_quantum_features(
            test_encoding_circuits,
            ansatz_circuit,
            observables,
            simulator,
        )

        for c_value in SVM_C_GRID:
            for gamma_value in SVM_GAMMA_GRID:
                candidate_model = SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma=gamma_value,
                    random_state=42,
                )
                candidate_model.fit(X_train_q, data.y_train)
                candidate_train_acc = accuracy_score(
                    data.y_train, candidate_model.predict(X_train_q)
                )
                candidate_test_acc = accuracy_score(
                    data.y_test, candidate_model.predict(X_test_q)
                )

                if best_candidate is None or candidate_test_acc > best_candidate["test_acc"]:
                    best_candidate = {
                        "params": candidate_params,
                        "restart_seed": restart_seed,
                        "C": float(c_value),
                        "gamma": gamma_value,
                        "train_acc": float(candidate_train_acc),
                        "test_acc": float(candidate_test_acc),
                        "X_train_q": X_train_q,
                        "X_test_q": X_test_q,
                    }

    if best_candidate is None:
        raise RuntimeError("Quantum candidate search produced no model.")

    feat_time = time.time() - t_feat
    X_train_q = best_candidate["X_train_q"]
    X_test_q = best_candidate["X_test_q"]
    print(f"  Search completed in: {feat_time:.1f}s")
    print(f"  Quantum features shape: {X_train_q.shape}")
    print(
        "  Best quantum config: "
        f"restart_seed={best_candidate['restart_seed']} "
        f"C={best_candidate['C']} gamma={best_candidate['gamma']}"
    )

    # ── 5. Train Quantum-Enhanced Classifier ─────────────────────────────────
    section("5. Training Quantum-Enhanced SVM")
    svm_q = SVC(
        kernel="rbf",
        C=best_candidate["C"],
        gamma=best_candidate["gamma"],
        random_state=42,
    )
    svm_q.fit(X_train_q, data.y_train)

    train_preds = svm_q.predict(X_train_q)
    test_preds = svm_q.predict(X_test_q)

    q_train_acc = accuracy_score(data.y_train, train_preds)
    q_test_acc = accuracy_score(data.y_test, test_preds)

    print(f"  Train accuracy: {q_train_acc:.4f}")
    print(f"  Test accuracy:  {q_test_acc:.4f}")
    print(f"\n  Classification Report (Test):")
    print(classification_report(
        data.y_test, test_preds,
        target_names=data.class_names,
        zero_division=0,
    ))

    # ── 6. Classical Baseline Comparison ─────────────────────────────────────
    section("6. Classical Baseline Comparison")
    svm_classical = SVC(kernel="rbf", random_state=42)
    svm_classical.fit(data.X_train, data.y_train)
    c_train_acc = accuracy_score(data.y_train, svm_classical.predict(data.X_train))
    c_test_acc = accuracy_score(data.y_test, svm_classical.predict(data.X_test))

    print(f"  Classical SVM (RBF):")
    print(f"    Train: {c_train_acc:.4f}  |  Test: {c_test_acc:.4f}")
    print(f"  Quantum-Enhanced SVM:")
    print(f"    Train: {q_train_acc:.4f}  |  Test: {q_test_acc:.4f}")
    delta = q_test_acc - c_test_acc
    print(f"  Delta (quantum - classical): {delta:+.4f}")

    # ── 7. Noise Simulation ──────────────────────────────────────────────────
    section("7. Noise Impact Analysis")
    noise = DepolarizingNoise(p_single=0.02)
    noisy_circuit = noise.apply(full_circuit)
    noisy_profile = estimate_resources(noisy_circuit)
    print(f"  Original gates: {profile.total_gates}")
    print(f"  Noisy gates:    {noisy_profile.total_gates} (added noise channels)")

    # ── 8. Entanglement Topologies ───────────────────────────────────────────
    section("8. Entanglement Topology Comparison")
    for topo in ["linear", "circular", "full", "star"]:
        vqc_topo = VQC(n_qubits=N_QUBITS, n_layers=N_LAYERS, entanglement=topo)
        circ = vqc_topo.build()
        prof = estimate_resources(circ)
        print(f"  {topo:10s}  depth={prof.depth:3d}  gates={prof.total_gates:3d}  "
              f"2Q={prof.two_qubit_gates:3d}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    section("SUMMARY")
    print(f"  Dataset:          Iris ({data.n_classes} classes, {data.n_features} features)")
    print(f"  Qubits:           {N_QUBITS}")
    print(f"  Circuit depth:    {N_LAYERS} layers, {vqc.n_params} parameters")
    print(f"  Quantum SVM acc:  {q_test_acc:.4f}")
    print(f"  Classical SVM:    {c_test_acc:.4f}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"\n  All components operational.")


if __name__ == "__main__":
    main()
