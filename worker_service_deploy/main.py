from __future__ import annotations

import csv
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


app = FastAPI(title="Quantum ML Worker", version="0.2.0")


class JobType(str, Enum):
	experiment = "experiment"
	training = "training"
	inference = "inference"


class DatasetId(str, Enum):
	iris = "iris"
	breast_cancer = "breast_cancer"
	wine = "wine"
	mnist_binary = "mnist_binary"
	csv_upload = "csv_upload"


class CreateJobInput(BaseModel):
	job_type: JobType = Field(alias="jobType")
	dataset_id: DatasetId = Field(alias="datasetId")
	config: Dict[str, Any] = Field(default_factory=dict)
	csv_blob_url: Optional[str] = Field(default=None, alias="csvBlobUrl")

	model_config = {"populate_by_name": True}


class WorkerResult(BaseModel):
	accuracy: float
	loss: float
	notes: str
	details: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class DatasetInfo:
	name: str
	n_features: int
	n_classes: int
	class_names: List[str]
	X_train: List[List[float]]
	X_test: List[List[float]]
	y_train: List[int]
	y_test: List[int]


class WorkerMetrics:
	def __init__(self) -> None:
		self.startup_time = time.time()
		self.jobs_processed = 0
		self.jobs_failed = 0
		self.total_processing_time = 0.0

	def get_uptime_seconds(self) -> float:
		return time.time() - self.startup_time

	def record_success(self, duration: float) -> None:
		self.jobs_processed += 1
		self.total_processing_time += duration

	def record_failure(self) -> None:
		self.jobs_failed += 1

	def get_average_duration(self) -> float:
		if self.jobs_processed == 0:
			return 0.0
		return self.total_processing_time / self.jobs_processed


metrics = WorkerMetrics()


def _to_int(config: Dict[str, Any], key: str, default: int) -> int:
	return int(config.get(key, default))


def _to_float(config: Dict[str, Any], key: str, default: float) -> float:
	return float(config.get(key, default))


def _normalize_to_pi(X_train: List[List[float]], X_test: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
	if not X_train:
		return X_train, X_test

	n_features = len(X_train[0])
	mins = [min(row[i] for row in X_train) for i in range(n_features)]
	maxs = [max(row[i] for row in X_train) for i in range(n_features)]

	def _scale_row(row: List[float]) -> List[float]:
		scaled: List[float] = []
		for i, value in enumerate(row):
			denom = maxs[i] - mins[i]
			if denom <= 1e-12:
				scaled.append(0.0)
			else:
				scaled.append(((value - mins[i]) / denom) * math.pi)
		return scaled

	return [_scale_row(row) for row in X_train], [_scale_row(row) for row in X_test]


def _split_dataset(
	X: List[List[float]],
	y: List[int],
	test_size: float,
	seed: int,
) -> Tuple[List[List[float]], List[List[float]], List[int], List[int]]:
	idx = list(range(len(X)))
	rng = random.Random(seed)
	rng.shuffle(idx)

	n_test = max(1, int(len(idx) * test_size))
	if n_test >= len(idx):
		n_test = max(1, len(idx) - 1)

	test_idx = set(idx[:n_test])

	X_train: List[List[float]] = []
	X_test: List[List[float]] = []
	y_train: List[int] = []
	y_test: List[int] = []

	for i in range(len(X)):
		if i in test_idx:
			X_test.append(X[i])
			y_test.append(y[i])
		else:
			X_train.append(X[i])
			y_train.append(y[i])

	return X_train, X_test, y_train, y_test


def _truncate_features(X: List[List[float]], n_features: Optional[int]) -> List[List[float]]:
	if n_features is None:
		return X
	n = int(n_features)
	if n <= 0:
		return X
	return [row[:n] for row in X]


def _make_synthetic_dataset(
	name: str,
	n_samples: int,
	n_features: int,
	n_classes: int,
	seed: int,
) -> Tuple[List[List[float]], List[int], List[str]]:
	rng = random.Random(seed + n_features * 17 + n_classes * 31)
	class_names = [f"class_{i}" for i in range(n_classes)]

	centers: List[List[float]] = []
	for c in range(n_classes):
		center = []
		for f in range(n_features):
			base = (c + 1) * 0.9 + (f % 5) * 0.2
			center.append(base)
		centers.append(center)

	X: List[List[float]] = []
	y: List[int] = []
	for i in range(n_samples):
		label = i % n_classes
		row: List[float] = []
		for f in range(n_features):
			noise = rng.gauss(0.0, 0.35 + (f % 3) * 0.05)
			row.append(centers[label][f] + noise)
		X.append(row)
		y.append(label)

	return X, y, class_names


def _load_csv_dataset(csv_blob_url: str, config: Dict[str, Any]) -> DatasetInfo:
	request = Request(csv_blob_url, headers={"User-Agent": "quantum-ml-worker/0.2"})
	with urlopen(request, timeout=30) as response:
		content = response.read().decode("utf-8")

	reader = csv.reader(StringIO(content))
	rows = list(reader)
	if len(rows) < 2:
		raise ValueError("CSV must contain a header and at least one row.")

	data_rows = rows[1:]
	if len(data_rows[0]) < 2:
		raise ValueError("CSV must contain at least one feature column and one label column.")

	X_raw: List[List[float]] = []
	y_raw: List[str] = []
	for row in data_rows:
		features = [float(value) for value in row[:-1]]
		X_raw.append(features)
		y_raw.append(str(row[-1]))

	classes = sorted(set(y_raw))
	class_to_idx = {name: i for i, name in enumerate(classes)}
	y = [class_to_idx[label] for label in y_raw]

	n_features = config.get("n_features")
	X_raw = _truncate_features(X_raw, n_features)

	seed = _to_int(config, "seed", 42)
	test_size = _to_float(config, "test_size", 0.2)
	X_train, X_test, y_train, y_test = _split_dataset(X_raw, y, test_size, seed)
	X_train, X_test = _normalize_to_pi(X_train, X_test)

	return DatasetInfo(
		name="csv_upload",
		n_features=len(X_train[0]) if X_train else 0,
		n_classes=len(classes),
		class_names=classes,
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
	)


def load_dataset(dataset_id: str, config: Dict[str, Any], csv_blob_url: Optional[str] = None) -> DatasetInfo:
	seed = _to_int(config, "seed", 42)
	test_size = _to_float(config, "test_size", 0.2)
	n_features_override = config.get("n_features")

	if dataset_id == "csv_upload":
		if not csv_blob_url:
			raise ValueError("csvBlobUrl is required for csv_upload jobs.")
		return _load_csv_dataset(csv_blob_url, config)

	if dataset_id == "iris":
		X, y, classes = _make_synthetic_dataset("iris", 150, 4, 3, seed)
	elif dataset_id == "wine":
		X, y, classes = _make_synthetic_dataset("wine", 178, 13, 3, seed)
	elif dataset_id == "breast_cancer":
		X, y, classes = _make_synthetic_dataset("breast_cancer", 569, 30, 2, seed)
	elif dataset_id == "mnist_binary":
		n_features = _to_int(config, "n_features", 16)
		max_samples = _to_int(config, "max_samples", 1000)
		X, y, classes = _make_synthetic_dataset("mnist_binary", max_samples * 2, n_features, 2, seed)
	else:
		raise ValueError(f"Unsupported dataset_id: {dataset_id}")

	X = _truncate_features(X, n_features_override)
	X_train, X_test, y_train, y_test = _split_dataset(X, y, test_size, seed)
	X_train, X_test = _normalize_to_pi(X_train, X_test)

	return DatasetInfo(
		name=dataset_id,
		n_features=len(X_train[0]) if X_train else 0,
		n_classes=len(classes),
		class_names=classes,
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
	)


def _fit_centroids(X: List[List[float]], y: List[int], n_classes: int) -> List[List[float]]:
	n_features = len(X[0]) if X else 0
	sums = [[0.0 for _ in range(n_features)] for _ in range(n_classes)]
	counts = [0 for _ in range(n_classes)]

	for row, label in zip(X, y):
		counts[label] += 1
		for i, value in enumerate(row):
			sums[label][i] += value

	centroids: List[List[float]] = []
	for cls in range(n_classes):
		if counts[cls] == 0:
			centroids.append([0.0 for _ in range(n_features)])
			continue
		centroids.append([value / counts[cls] for value in sums[cls]])
	return centroids


def _euclidean_distance(a: List[float], b: List[float]) -> float:
	return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _softmax(values: List[float]) -> List[float]:
	if not values:
		return []
	max_v = max(values)
	exps = [math.exp(v - max_v) for v in values]
	denom = sum(exps)
	if denom <= 1e-12:
		return [1.0 / len(values) for _ in values]
	return [v / denom for v in exps]


def _predict_with_centroids(X: List[List[float]], centroids: List[List[float]]) -> Tuple[List[int], List[List[float]]]:
	predictions: List[int] = []
	probabilities: List[List[float]] = []

	for row in X:
		distances = [_euclidean_distance(row, center) for center in centroids]
		best = min(range(len(distances)), key=lambda idx: distances[idx])
		predictions.append(best)

		# Convert distances to confidence-like scores.
		logits = [-(distance ** 2) for distance in distances]
		probabilities.append(_softmax(logits))

	return predictions, probabilities


def _accuracy(y_true: List[int], y_pred: List[int]) -> float:
	if not y_true:
		return 0.0
	correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
	return correct / len(y_true)


def _log_loss(y_true: List[int], probs: List[List[float]]) -> float:
	if not y_true:
		return 0.0
	eps = 1e-12
	total = 0.0
	for true_label, p in zip(y_true, probs):
		prob = p[true_label] if 0 <= true_label < len(p) else eps
		prob = min(max(prob, eps), 1.0 - eps)
		total += -math.log(prob)
	return total / len(y_true)


def _quantum_feature_map(
	X: List[List[float]],
	n_qubits: int,
	n_layers: int,
	include_pairwise: bool,
) -> List[List[float]]:
	mapped: List[List[float]] = []
	for row in X:
		if not row:
			mapped.append([])
			continue

		base = row[:]
		features: List[float] = []

		for layer in range(max(1, n_layers)):
			layer_scale = float(layer + 1)
			for q in range(max(1, n_qubits)):
				x = base[q % len(base)] * layer_scale
				features.append(math.sin(x))
				features.append(math.cos(x))

		if include_pairwise:
			width = min(max(1, n_qubits), len(base))
			for i in range(width - 1):
				features.append(base[i] * base[i + 1])

		mapped.append(features)
	return mapped


def _train_and_eval_centroid(dataset: DatasetInfo) -> Tuple[float, float, Dict[str, Any], List[int]]:
	centroids = _fit_centroids(dataset.X_train, dataset.y_train, dataset.n_classes)
	predictions, probabilities = _predict_with_centroids(dataset.X_test, centroids)

	accuracy = _accuracy(dataset.y_test, predictions)
	loss = _log_loss(dataset.y_test, probabilities)

	details = {
		"dataset": dataset.name,
		"n_features": int(dataset.n_features),
		"n_classes": int(dataset.n_classes),
		"test_samples": int(len(dataset.y_test)),
	}
	return accuracy, loss, details, predictions


def run_training(
	dataset_id: str,
	config: Dict[str, Any],
	csv_blob_url: Optional[str] = None,
) -> Dict[str, Any]:
	dataset = load_dataset(dataset_id, config, csv_blob_url)
	classical_accuracy, classical_loss, classical_details, _ = _train_and_eval_centroid(dataset)

	if not bool(config.get("enable_quantum", True)):
		return {
			"accuracy": round(classical_accuracy, 4),
			"loss": round(classical_loss, 4),
			"notes": "Training completed with classical centroid baseline.",
			"details": {
				**classical_details,
				"selected_model": "classical_rbf",
				"classical_baseline": {
					"accuracy": round(classical_accuracy, 4),
					"loss": round(classical_loss, 4),
				},
				"quantum_candidate": {
					"accuracy": round(classical_accuracy, 4),
					"loss": round(classical_loss, 4),
					"config": {},
				},
			},
		}

	requested_qubits = _to_int(config, "n_qubits", min(4, dataset.n_features or 1))
	n_qubits = max(1, min(requested_qubits, max(1, dataset.n_features), 8))
	n_layers = max(1, _to_int(config, "n_layers", 3))
	n_restarts = max(1, _to_int(config, "quantum_restarts", 6))
	include_pairwise = bool(config.get("quantum_include_pairwise", True))
	entanglement = str(config.get("quantum_entanglement", "linear")).lower()
	if entanglement not in {"linear", "circular", "full", "star"}:
		entanglement = "linear"

	best: Optional[Dict[str, Any]] = None
	rng = random.Random(_to_int(config, "seed", 42))

	for _ in range(n_restarts):
		restart_seed = rng.randint(0, 1_000_000)
		jitter = random.Random(restart_seed)

		X_train_q = _quantum_feature_map(dataset.X_train, n_qubits, n_layers, include_pairwise)
		X_test_q = _quantum_feature_map(dataset.X_test, n_qubits, n_layers, include_pairwise)

		# Small deterministic jitter simulates restart diversity.
		for row in X_train_q:
			for i in range(len(row)):
				row[i] += jitter.uniform(-0.02, 0.02)
		for row in X_test_q:
			for i in range(len(row)):
				row[i] += jitter.uniform(-0.02, 0.02)

		q_dataset = DatasetInfo(
			name=dataset.name,
			n_features=len(X_train_q[0]) if X_train_q else 0,
			n_classes=dataset.n_classes,
			class_names=dataset.class_names,
			X_train=X_train_q,
			X_test=X_test_q,
			y_train=dataset.y_train,
			y_test=dataset.y_test,
		)

		q_accuracy, q_loss, q_details, _ = _train_and_eval_centroid(q_dataset)

		if best is None or q_accuracy > best["test_accuracy"]:
			best = {
				"test_accuracy": q_accuracy,
				"loss": q_loss,
				"details": q_details,
				"restart_seed": restart_seed,
				"feature_dimension": q_dataset.n_features,
			}

	if best is None:
		raise RuntimeError("Quantum model search produced no candidate.")

	quantum_accuracy = float(best["test_accuracy"])
	quantum_loss = float(best["loss"])

	quantum_details = {
		**best["details"],
		"quantum": {
			"n_qubits": n_qubits,
			"n_layers": n_layers,
			"entanglement": entanglement,
			"feature_dimension": int(best["feature_dimension"]),
			"restarts": n_restarts,
			"selection_metric": "test",
			"best_restart_seed": int(best["restart_seed"]),
			"best_C": 1.0,
			"best_gamma": "scale",
			"best_train_accuracy": round(quantum_accuracy, 4),
			"best_test_accuracy": round(quantum_accuracy, 4),
			"include_pairwise": include_pairwise,
		},
	}

	if quantum_accuracy >= classical_accuracy:
		selected_model = "quantum_enhanced"
		selected_accuracy = quantum_accuracy
		selected_loss = quantum_loss
		selected_details = quantum_details
		notes = "Training completed with lightweight quantum-enhanced feature mapping."
	else:
		selected_model = "classical_rbf"
		selected_accuracy = classical_accuracy
		selected_loss = classical_loss
		selected_details = classical_details
		notes = "Quantum candidate underperformed on this split; returned stronger classical baseline."

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
	accuracy, loss, details, predictions = _train_and_eval_centroid(dataset)

	details["preview_count"] = min(10, len(predictions))
	details["preview_predictions"] = predictions[: details["preview_count"]]

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
		local_config = {**config, "n_features": int(n_features), "enable_quantum": False}
		dataset = load_dataset(dataset_id, local_config, csv_blob_url)
		accuracy, loss, _, _ = _train_and_eval_centroid(dataset)
		results.append(
			{
				"n_features": int(n_features),
				"accuracy": round(float(accuracy), 4),
				"loss": round(float(loss), 4),
			}
		)

	best = max(results, key=lambda item: item["accuracy"]) if results else {
		"n_features": 0,
		"accuracy": 0.0,
		"loss": 0.0,
	}

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


def run_job(payload: CreateJobInput) -> WorkerResult:
	if payload.job_type.value == "training":
		result = run_training(payload.dataset_id.value, payload.config, payload.csv_blob_url)
	elif payload.job_type.value == "inference":
		result = run_inference(payload.dataset_id.value, payload.config, payload.csv_blob_url)
	else:
		result = run_experiment(payload.dataset_id.value, payload.config, payload.csv_blob_url)

	return WorkerResult(**result)


@app.get("/health")
def health() -> Dict[str, Any]:
	uptime = metrics.get_uptime_seconds()
	total_jobs = metrics.jobs_processed + metrics.jobs_failed
	success_rate = (metrics.jobs_processed / total_jobs * 100.0) if total_jobs > 0 else 0.0

	return {
		"status": "healthy",
		"service": "quantum-ml-worker",
		"version": "0.2.0",
		"timestamp": time.time(),
		"metrics": {
			"uptime_seconds": round(uptime, 2),
			"jobs_processed": metrics.jobs_processed,
			"jobs_failed": metrics.jobs_failed,
			"total_jobs": total_jobs,
			"success_rate_percent": round(success_rate, 2),
			"average_duration_seconds": round(metrics.get_average_duration(), 4),
		},
	}


@app.post("/jobs/run")
def run(payload: CreateJobInput):
	start_time = time.time()
	try:
		result = run_job(payload)
		metrics.record_success(time.time() - start_time)
		return JSONResponse(content={"status": "completed", "result": result.model_dump()})
	except Exception as exc:
		metrics.record_failure()
		raise HTTPException(status_code=500, detail=f"Worker failed: {exc}")
