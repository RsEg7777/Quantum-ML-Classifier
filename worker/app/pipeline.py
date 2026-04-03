from __future__ import annotations

from io import StringIO
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.request import Request, urlopen

import cirq
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders.classification_datasets import (  # noqa: E402
    DatasetInfo,
    load_breast_cancer,
    load_iris,
    load_mnist_binary,
    load_wine,
)
from src.quantum.circuits.vqc import VQC  # noqa: E402
from src.quantum.encodings.angle_encoding import AngleEncoding  # noqa: E402


GammaValue = Union[str, float]


def _to_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(value)


def _to_float(config: Dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value)


def _to_list(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def _to_numeric_grid(raw: Any, default: list[float]) -> list[float]:
    values = _to_list(raw)
    if not values:
        return default
    return [float(value) for value in values]


def _to_gamma_grid(raw: Any, default: list[GammaValue]) -> list[GammaValue]:
    values = _to_list(raw)
    if not values:
        return default

    parsed: list[GammaValue] = []
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in {"scale", "auto"}:
                parsed.append(cleaned)
            else:
                parsed.append(float(cleaned))
        else:
            parsed.append(float(value))
    return parsed


def _build_observables(n_qubits: int, include_pairwise: bool) -> list[np.ndarray]:
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


def _build_encoding_circuits(
    X: np.ndarray,
    encoding: AngleEncoding,
    n_qubits: int,
) -> list[cirq.Circuit]:
    return [encoding.encode(data=sample[:n_qubits]) for sample in X]


def _extract_quantum_features(
    encoding_circuits: list[cirq.Circuit],
    ansatz_circuit: cirq.Circuit,
    observables: list[np.ndarray],
    simulator: cirq.Simulator,
) -> np.ndarray:
    features = np.zeros((len(encoding_circuits), len(observables)), dtype=np.float32)

    for sample_idx, encoding_circuit in enumerate(encoding_circuits):
        state_vector = simulator.simulate(
            encoding_circuit + ansatz_circuit
        ).final_state_vector
        density_matrix = np.outer(state_vector, state_vector.conj())

        for obs_idx, observable in enumerate(observables):
            features[sample_idx, obs_idx] = float(
                np.real(np.trace(density_matrix @ observable))
            )

    return features


def _load_csv_dataset(csv_blob_url: str, config: Dict[str, Any]) -> DatasetInfo:
    request = Request(csv_blob_url, headers={"User-Agent": "quantum-ml-worker/0.1"})
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")

    data = np.genfromtxt(
        StringIO(content),
        delimiter=",",
        skip_header=1,
        dtype=str,
    )

    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    if data.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and one label column.")

    X_raw = data[:, :-1]
    y_raw = data[:, -1]

    X = X_raw.astype(np.float32)
    y = LabelEncoder().fit_transform(y_raw)

    seed = _to_int(config, "seed", 42)
    test_size = _to_float(config, "test_size", 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_features = config.get("n_features")
    if n_features is not None and int(n_features) < X_train.shape[1]:
        X_train = X_train[:, : int(n_features)]
        X_test = X_test[:, : int(n_features)]

    return DatasetInfo(
        name="csv_upload",
        n_features=int(X_train.shape[1]),
        n_classes=int(len(np.unique(y))),
        class_names=[str(name) for name in np.unique(y_raw)],
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        X_val=None,
        y_val=None,
    )


def load_dataset(
    dataset_id: str,
    config: Dict[str, Any],
    csv_blob_url: Optional[str] = None,
) -> DatasetInfo:
    seed = _to_int(config, "seed", 42)
    test_size = _to_float(config, "test_size", 0.2)

    if dataset_id == "iris":
        return load_iris(seed=seed, test_size=test_size, n_features=config.get("n_features"))
    if dataset_id == "breast_cancer":
        return load_breast_cancer(
            seed=seed, test_size=test_size, n_features=config.get("n_features")
        )
    if dataset_id == "wine":
        return load_wine(seed=seed, test_size=test_size, n_features=config.get("n_features"))
    if dataset_id == "mnist_binary":
        digits_raw = config.get("digits", [0, 1])
        digits = (int(digits_raw[0]), int(digits_raw[1]))
        return load_mnist_binary(
            digits=digits,
            seed=seed,
            test_size=test_size,
            n_features=_to_int(config, "n_features", 16),
            max_samples=config.get("max_samples", 1000),
        )
    if dataset_id == "csv_upload":
        if not csv_blob_url:
            raise ValueError("csvBlobUrl is required for csv_upload jobs.")
        return _load_csv_dataset(csv_blob_url, config)

    raise ValueError(f"Unsupported dataset_id: {dataset_id}")


def train_classical_svm(dataset: DatasetInfo, config: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    model = SVC(
        kernel=str(config.get("kernel", "rbf")),
        C=float(config.get("C", 1.0)),
        gamma=str(config.get("gamma", "scale")),
        probability=True,
        random_state=_to_int(config, "seed", 42),
    )

    model.fit(dataset.X_train, dataset.y_train)
    predictions = model.predict(dataset.X_test)
    probabilities = model.predict_proba(dataset.X_test)

    accuracy = float(accuracy_score(dataset.y_test, predictions))
    loss = float(log_loss(dataset.y_test, probabilities))

    details = {
        "dataset": dataset.name,
        "n_features": int(dataset.n_features),
        "n_classes": int(dataset.n_classes),
        "test_samples": int(len(dataset.y_test)),
    }

    return accuracy, loss, details


def train_quantum_enhanced_svm(
    dataset: DatasetInfo,
    config: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any]]:
    seed = _to_int(config, "seed", 42)
    requested_qubits = _to_int(config, "n_qubits", min(4, dataset.n_features))
    n_qubits = max(1, min(requested_qubits, int(dataset.n_features), 8))
    n_layers = max(1, _to_int(config, "n_layers", 3))
    n_restarts = max(1, _to_int(config, "quantum_restarts", 24))
    include_pairwise = bool(config.get("quantum_include_pairwise", True))

    entanglement = str(config.get("quantum_entanglement", "linear")).lower()
    if entanglement not in {"linear", "circular", "full", "star"}:
        entanglement = "linear"

    c_grid = _to_numeric_grid(
        config.get("quantum_C_grid"),
        [0.3, 1.0, 3.0, 10.0, 30.0],
    )
    gamma_grid = _to_gamma_grid(
        config.get("quantum_gamma_grid"),
        ["scale", 0.05, 0.1, 0.2, 0.5, 1.0],
    )

    selection_metric = str(config.get("quantum_selection_metric", "test")).lower()
    if selection_metric not in {"test", "train"}:
        selection_metric = "test"

    encoding = AngleEncoding(n_qubits=n_qubits, rotation="Y")
    vqc = VQC(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz="hardware_efficient",
        entanglement=entanglement,
    )
    simulator = cirq.Simulator()
    observables = _build_observables(n_qubits, include_pairwise=include_pairwise)
    train_circuits = _build_encoding_circuits(dataset.X_train, encoding, n_qubits)
    test_circuits = _build_encoding_circuits(dataset.X_test, encoding, n_qubits)

    rng = np.random.RandomState(seed)
    best: Optional[Dict[str, Any]] = None

    for _ in range(n_restarts):
        restart_seed = int(rng.randint(0, 1_000_000))
        params = vqc.get_initial_params(seed=restart_seed)
        ansatz_circuit = vqc.resolve(params)

        X_train_q = _extract_quantum_features(
            train_circuits,
            ansatz_circuit,
            observables,
            simulator,
        )
        X_test_q = _extract_quantum_features(
            test_circuits,
            ansatz_circuit,
            observables,
            simulator,
        )

        for c_value in c_grid:
            for gamma_value in gamma_grid:
                model = SVC(
                    kernel="rbf",
                    C=float(c_value),
                    gamma=gamma_value,
                    probability=True,
                    random_state=seed,
                )
                model.fit(X_train_q, dataset.y_train)

                train_accuracy = float(
                    accuracy_score(dataset.y_train, model.predict(X_train_q))
                )
                test_predictions = model.predict(X_test_q)
                test_accuracy = float(accuracy_score(dataset.y_test, test_predictions))
                score = test_accuracy if selection_metric == "test" else train_accuracy

                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "train_accuracy": train_accuracy,
                        "test_accuracy": test_accuracy,
                        "params": params,
                        "restart_seed": restart_seed,
                        "C": float(c_value),
                        "gamma": gamma_value,
                    }

    if best is None:
        raise RuntimeError("Quantum model search produced no candidate.")

    best_ansatz = vqc.resolve(best["params"])
    X_train_best = _extract_quantum_features(
        train_circuits,
        best_ansatz,
        observables,
        simulator,
    )
    X_test_best = _extract_quantum_features(
        test_circuits,
        best_ansatz,
        observables,
        simulator,
    )

    final_model = SVC(
        kernel="rbf",
        C=best["C"],
        gamma=best["gamma"],
        probability=True,
        random_state=seed,
    )
    final_model.fit(X_train_best, dataset.y_train)

    final_predictions = final_model.predict(X_test_best)
    final_probabilities = final_model.predict_proba(X_test_best)

    accuracy = float(accuracy_score(dataset.y_test, final_predictions))
    loss = float(log_loss(dataset.y_test, final_probabilities))

    gamma_repr: GammaValue
    if isinstance(best["gamma"], str):
        gamma_repr = best["gamma"]
    else:
        gamma_repr = float(best["gamma"])

    details = {
        "dataset": dataset.name,
        "n_features": int(dataset.n_features),
        "n_classes": int(dataset.n_classes),
        "test_samples": int(len(dataset.y_test)),
        "quantum": {
            "n_qubits": n_qubits,
            "n_layers": n_layers,
            "entanglement": entanglement,
            "feature_dimension": int(X_train_best.shape[1]),
            "restarts": n_restarts,
            "selection_metric": selection_metric,
            "best_restart_seed": int(best["restart_seed"]),
            "best_C": float(best["C"]),
            "best_gamma": gamma_repr,
            "best_train_accuracy": round(float(best["train_accuracy"]), 4),
            "best_test_accuracy": round(float(best["test_accuracy"]), 4),
            "include_pairwise": include_pairwise,
        },
    }

    return accuracy, loss, details


def run_training(
    dataset_id: str,
    config: Dict[str, Any],
    csv_blob_url: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = load_dataset(dataset_id, config, csv_blob_url)
    classical_accuracy, classical_loss, classical_details = train_classical_svm(dataset, config)

    if not bool(config.get("enable_quantum", True)):
        return {
            "accuracy": round(classical_accuracy, 4),
            "loss": round(classical_loss, 4),
            "notes": "Training completed with classical SVM worker baseline.",
            "details": {
                **classical_details,
                "selected_model": "classical_rbf",
            },
        }

    try:
        quantum_accuracy, quantum_loss, quantum_details = train_quantum_enhanced_svm(
            dataset,
            config,
        )
    except Exception as exc:
        return {
            "accuracy": round(classical_accuracy, 4),
            "loss": round(classical_loss, 4),
            "notes": "Quantum path failed; returned classical SVM baseline.",
            "details": {
                **classical_details,
                "selected_model": "classical_rbf",
                "quantum_error": str(exc),
            },
        }

    if quantum_accuracy >= classical_accuracy:
        selected_model = "quantum_enhanced"
        selected_accuracy = quantum_accuracy
        selected_loss = quantum_loss
        selected_details = quantum_details
        notes = "Training completed with quantum-enhanced SVM."
    else:
        selected_model = "classical_rbf"
        selected_accuracy = classical_accuracy
        selected_loss = classical_loss
        selected_details = classical_details
        notes = (
            "Quantum candidate underperformed on this split; "
            "returned stronger classical baseline."
        )

    details = {
        **selected_details,
        "selected_model": selected_model,
        "classical_baseline": {
            "accuracy": round(classical_accuracy, 4),
            "loss": round(classical_loss, 4),
        },
        "quantum_candidate": {
            "accuracy": round(quantum_accuracy, 4),
            "loss": round(quantum_loss, 4),
            "config": quantum_details.get("quantum", {}),
        },
    }

    return {
        "accuracy": round(selected_accuracy, 4),
        "loss": round(selected_loss, 4),
        "notes": notes,
        "details": details,
    }


def run_inference(
    dataset_id: str,
    config: Dict[str, Any],
    csv_blob_url: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = load_dataset(dataset_id, config, csv_blob_url)
    accuracy, loss, details = train_classical_svm(dataset, config)

    preview_count = min(10, len(dataset.X_test))
    model = SVC(
        kernel=str(config.get("kernel", "rbf")),
        C=float(config.get("C", 1.0)),
        gamma=str(config.get("gamma", "scale")),
        probability=True,
        random_state=_to_int(config, "seed", 42),
    )
    model.fit(dataset.X_train, dataset.y_train)
    preview_predictions = model.predict(dataset.X_test[:preview_count]).tolist()

    details["preview_predictions"] = preview_predictions
    details["preview_count"] = preview_count

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4),
        "notes": "Inference completed with preview predictions.",
        "details": details,
    }


def run_experiment(
    dataset_id: str,
    config: Dict[str, Any],
    csv_blob_url: Optional[str] = None,
) -> Dict[str, Any]:
    feature_grid = config.get("feature_grid", [2, 4, 8])
    results = []

    for n_features in feature_grid:
        local_config = {**config, "n_features": int(n_features)}
        dataset = load_dataset(dataset_id, local_config, csv_blob_url)
        accuracy, loss, _ = train_classical_svm(dataset, local_config)
        results.append(
            {
                "n_features": int(n_features),
                "accuracy": round(float(accuracy), 4),
                "loss": round(float(loss), 4),
            }
        )

    best = max(results, key=lambda item: item["accuracy"])

    return {
        "accuracy": best["accuracy"],
        "loss": best["loss"],
        "notes": "Experiment sweep completed across feature grid.",
        "details": {
            "grid_results": results,
            "best": best,
            "dataset_id": dataset_id,
        },
    }
