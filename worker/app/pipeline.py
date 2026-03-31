from __future__ import annotations

from io import StringIO
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen

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


def _to_int(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    return int(value)


def _to_float(config: Dict[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    return float(value)


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


def run_training(
    dataset_id: str,
    config: Dict[str, Any],
    csv_blob_url: Optional[str] = None,
) -> Dict[str, Any]:
    dataset = load_dataset(dataset_id, config, csv_blob_url)
    accuracy, loss, details = train_classical_svm(dataset, config)

    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss, 4),
        "notes": "Training completed with classical SVM worker baseline.",
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
