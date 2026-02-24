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
    X: np.ndarray,
    n_qubits: int,
    n_layers: int,
    params: np.ndarray,
    encoding: AngleEncoding,
    vqc: VQC,
) -> np.ndarray:
    """Simulate encoding+ansatz and return Z-expectation features."""
    simulator = cirq.Simulator()
    qubits = encoding.qubits
    n = len(qubits)

    # Pre-build the Z operators as full matrices
    z_mats = []
    z_single = cirq.unitary(cirq.Z)
    for q_idx in range(n):
        mat = np.eye(1)
        for j in range(n):
            mat = np.kron(mat, z_single if j == q_idx else np.eye(2))
        z_mats.append(mat)

    features = np.zeros((len(X), n_qubits))
    ansatz_circuit = vqc.resolve(params)

    for i, x in enumerate(X):
        enc_circuit = encoding.encode(data=x[:n_qubits])
        full_circuit = enc_circuit + ansatz_circuit
        sv = simulator.simulate(full_circuit).final_state_vector
        dm = np.outer(sv, sv.conj())
        for q_idx in range(n_qubits):
            features[i, q_idx] = np.real(np.trace(dm @ z_mats[q_idx]))
    return features


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

    # ── 4. Extract Quantum Features ──────────────────────────────────────────
    section("4. Extracting Quantum Features (Cirq Simulation)")
    print(f"  Processing {len(data.X_train)} train + {len(data.X_test)} test samples...")

    t_feat = time.time()
    X_train_q = extract_quantum_features(
        data.X_train, N_QUBITS, N_LAYERS, params, encoding, vqc
    )
    X_test_q = extract_quantum_features(
        data.X_test, N_QUBITS, N_LAYERS, params, encoding, vqc
    )
    feat_time = time.time() - t_feat
    print(f"  Feature extraction: {feat_time:.1f}s")
    print(f"  Quantum features shape: {X_train_q.shape}")

    # ── 5. Train Quantum-Enhanced Classifier ─────────────────────────────────
    section("5. Training Quantum-Enhanced SVM")
    svm_q = SVC(kernel="rbf", random_state=42)
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
