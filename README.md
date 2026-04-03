# Quantum-Classical Hybrid ML Classifier

A modular, research-grade framework for **quantum-classical hybrid machine learning** built on [Google Cirq](https://quantumai.google/cirq) and [TensorFlow Quantum](https://www.tensorflow.org/quantum). It provides a complete pipeline — from quantum data encoding through variational circuit training to classical post-processing — designed for experimentation, benchmarking, and education.

> **Status:** 40/40 tests passing · end-to-end pipeline verified · runs in ~4 s on CPU

---

## Key Results (Iris Dataset)

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Quantum-Enhanced SVM | 97.50% | **100.00%** |
| Classical SVM (RBF) | 97.50% | 96.67% |

Full output: [`output/pipeline_output.txt`](output/pipeline_output.txt)

---

## Features

### Quantum Components
- **5 Encoding Schemes** — Angle, Amplitude, IQP, Basis, Hamiltonian
- **6 Circuit Architectures** — VQC (hardware-efficient & strongly-entangling ansätze), QCNN, QAOA, Tensor Networks (MPS/TTN), QRNN, QGNN
- **4 Entanglement Topologies** — Linear, Circular, Full, Star
- **Parametric & Standard Gates** — Rx, Ry, Rz, CNOT, CZ, SWAP, and more
- **Measurements** — Pauli (X/Y/Z) expectation values, projective measurements
- **Quantum Kernels** — Fidelity-based kernel with scikit-learn estimator compatibility

### Classical Components
- **Baseline Models** — SVM, Gaussian Processes, XGBoost/LightGBM ensembles, MLP
- **Hybrid Pipeline** — Quantum feature extraction → classical classifier

### Noise & Error Mitigation
- **Noise Channels** — Depolarizing, amplitude damping, phase damping, readout error
- **Mitigation** — Zero-Noise Extrapolation (ZNE), measurement error mitigation

### Optimization
- **Quantum-Aware Optimizers** — SPSA, Rotosolve
- **Training Loop** — Configurable training with logging and checkpointing

### Analysis & Visualization
- **Circuit Analysis** — Expressibility, entangling capability, resource estimation, gradient analysis, loss landscape
- **Visualization** — Training curves, circuit diagrams

### Experiment Framework
- **YAML-Driven Experiments** — Define and run reproducible experiment suites
- **Built-in Experiments** — Depth scaling, encoding comparison

---

## Project Structure

```
Quantum-ML-Classifier/
├── main.py                          # End-to-end demo pipeline
├── pyproject.toml                   # Project metadata & dependencies
│
├── src/
│   ├── quantum/
│   │   ├── encodings/               # Angle, Amplitude, IQP, Basis, Hamiltonian
│   │   ├── circuits/                # VQC, QCNN, QAOA, Tensor Networks, QRNN, QGNN
│   │   ├── gates/                   # Standard & parametric gate libraries
│   │   ├── measurements/            # Pauli & projective measurements
│   │   ├── kernels/                 # Fidelity-based quantum kernel (sklearn API)
│   │   └── entanglement/            # Entanglement topology strategies
│   │
│   ├── classical/
│   │   ├── baselines/               # SVM, Gaussian Processes, Ensembles
│   │   └── networks/                # MLP
│   │
│   ├── hybrid/
│   │   ├── models/                  # HybridModel, QuantumLayer
│   │   ├── optimization/            # SPSA, Rotosolve
│   │   └── training/                # Training loop
│   │
│   ├── noise/
│   │   ├── models/                  # Depolarizing, amplitude/phase damping, readout
│   │   └── mitigation/              # ZNE, measurement error mitigation
│   │
│   ├── analysis/
│   │   ├── circuit/                 # Expressibility, resource estimation
│   │   └── optimization/            # Gradient analysis, loss landscape
│   │
│   ├── visualization/
│   │   ├── circuits/                # Circuit drawer
│   │   └── ml/                      # Training curves
│   │
│   ├── data/loaders/                # Iris, Breast Cancer dataset loaders
│   └── utils/                       # Config, logging, reproducibility
│
├── experiments/
│   ├── run_all_experiments.py        # Experiment runner framework
│   └── suite_1_architecture/         # Depth scaling experiment
│
├── configs/
│   ├── experiments/                  # Depth scaling & encoding comparison configs
│   └── models/                       # VQC config
│
├── tests/unit/                       # 40 unit tests (circuits, encodings)
├── output/                           # Documented pipeline & test outputs
├── docs/                             # Documentation
├── notebooks/                        # Jupyter notebooks
├── scripts/                          # Utility scripts
├── docker/                           # Docker configuration
└── ci_cd/                            # CI/CD configuration
```

---

## Quick Start

### Prerequisites

- **Python 3.9–3.11** (TensorFlow Quantum requires < 3.12)
- **Linux or WSL2** (TFQ is Linux-only; Windows users use WSL2)

### Installation

```bash
# Clone the repository
git clone https://github.com/RsEg7777/Quantum-ML-Classifier.git
cd Quantum-ML-Classifier

# Create virtual environment (Python 3.10 recommended)
python3.10 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install cirq==1.3.0 tensorflow==2.15.0 tensorflow-quantum==0.7.2
pip install scikit-learn numpy scipy sympy matplotlib seaborn pyyaml tqdm

# Install test dependencies
pip install pytest
```

### Run the Demo

```bash
python main.py
```

This executes the full pipeline on the Iris dataset:
1. Loads and preprocesses data (4 features, 3 classes)
2. Builds a 4-qubit VQC with angle encoding + hardware-efficient ansatz (3 layers, 24 parameters)
3. Extracts quantum features via Cirq statevector simulation
4. Trains a quantum-enhanced SVM classifier
5. Compares against a classical SVM baseline
6. Analyzes noise impact and entanglement topologies

### Run Tests

```bash
python -m pytest tests/ -v
```

Expected: **40/40 passed** — see [`output/test_results.txt`](output/test_results.txt)

---

## How It Works

### Pipeline

```
Raw Data → Encoding → |ψ(x)⟩ → Variational Ansatz → U(θ)|ψ(x)⟩ → Measurement → Features → Classical ML → Prediction
```

1. **Data Encoding** — Classical features are mapped to quantum states via angle encoding (Ry rotations proportional to feature values)
2. **Variational Circuit** — A parameterized ansatz transforms the encoded state; the hardware-efficient ansatz alternates Ry/Rz rotation layers with CNOT entangling layers
3. **Measurement** — Pauli-Z expectation values are extracted per qubit, producing a quantum feature vector
4. **Classical Post-Processing** — An SVM (or other classifier) operates on the quantum features

### Sample Circuit (4 qubits, 3 layers)

```
q0: ──Ry(x₀)──Ry(θ₀)──Rz(θ₁)──@────────────────────────────────── ···
                                │
q1: ──Ry(x₁)──Ry(θ₂)──Rz(θ₃)──X──@─────────────────────────────── ···
                                    │
q2: ──Ry(x₂)──Ry(θ₄)──Rz(θ₅)─────X──@──────────────────────────── ···
                                       │
q3: ──Ry(x₃)──Ry(θ₆)──Rz(θ₇)─────────X──────────────────────────── ···
      ╰─encoding─╯     ╰──── variational ansatz (× 3 layers) ────╯
```

### Entanglement Topologies

- **Linear** — nearest-neighbor: (0,1), (1,2), (2,3)
- **Circular** — linear + wrap-around: adds (3,0)
- **Full** — all-to-all: all C(n,2) pairs
- **Star** — hub-spoke from qubit 0: (0,1), (0,2), (0,3)

Topology impact on circuit resources (4 qubits, 3 layers):

| Topology | Depth | Gates | 2-Qubit Gates |
|---|---|---|---|
| Linear | 13 | 33 | 9 |
| Circular | 18 | 36 | 12 |
| Full | 19 | 42 | 18 |
| Star | 15 | 33 | 9 |

---

## Configuration

Experiments are configured via YAML files in `configs/`:

```yaml
# configs/experiments/depth_scaling.yaml
experiment:
  name: depth_scaling
  dataset: iris
  n_qubits: 4
  depths: [1, 2, 3, 4, 5, 6]
  encoding: angle
  ansatz: hardware_efficient
  entanglement: linear
```

Run experiments:

```bash
python -m experiments.run_all_experiments
```

---

## Tech Stack

| Component | Library | Version |
|---|---|---|
| Quantum Simulation | Cirq | 1.3.0 |
| Quantum ML | TensorFlow Quantum | 0.7.2 |
| Deep Learning | TensorFlow | 2.15.0 |
| Classical ML | scikit-learn | ≥ 1.3 |
| Symbolic Math | SymPy | ≥ 1.12 |
| Numerical | NumPy | ≥ 1.24 |

---

## Output

The [`output/`](output/) directory contains documented results from verified runs:

- **[`pipeline_output.txt`](output/pipeline_output.txt)** — Full `main.py` output including circuit diagrams, accuracy metrics, resource analysis, noise simulation, and topology comparison
- **[`test_results.txt`](output/test_results.txt)** — Complete pytest output showing all 40 tests passing

---

## License

MIT License — see [`pyproject.toml`](pyproject.toml) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## Acknowledgments

- [Google Cirq](https://quantumai.google/cirq) — Quantum circuit framework
- [TensorFlow Quantum](https://www.tensorflow.org/quantum) — Hybrid quantum-classical ML
- [scikit-learn](https://scikit-learn.org/) — Classical ML baselines
