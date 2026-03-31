"""
Tests for worker pipeline and CSV ingestion.
"""

from __future__ import annotations

import io

from worker.app import pipeline


class _MockResponse:
    def __init__(self, text: str):
        self._text = text

    def read(self) -> bytes:
        return self._text.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_training_builtin_dataset():
    result = pipeline.run_training("iris", {"seed": 42, "n_features": 4})

    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["loss"] >= 0.0
    assert result["details"]["dataset"].startswith("iris")


def test_load_dataset_csv_upload(monkeypatch):
    csv_text = """f1,f2,label
0.1,1.2,A
0.2,1.1,A
1.8,3.0,B
1.9,3.2,B
2.1,2.9,B
0.0,1.0,A
"""

    def _fake_urlopen(_request, timeout=30):
        _ = timeout
        return _MockResponse(csv_text)

    monkeypatch.setattr(pipeline, "urlopen", _fake_urlopen)

    dataset = pipeline.load_dataset(
        "csv_upload",
        {"seed": 42, "test_size": 0.33},
        csv_blob_url="http://local.test/upload.csv?token=abc",
    )

    assert dataset.name == "csv_upload"
    assert dataset.n_features == 2
    assert dataset.n_classes == 2
    assert dataset.X_train.shape[1] == 2


def test_run_experiment_csv_upload(monkeypatch):
    csv_text = """f1,f2,f3,label
0.1,1.2,0.4,A
0.2,1.1,0.3,A
1.8,3.0,1.0,B
1.9,3.2,1.1,B
2.1,2.9,1.2,B
0.0,1.0,0.2,A
2.2,3.3,1.3,B
0.3,0.9,0.1,A
"""

    def _fake_urlopen(_request, timeout=30):
        _ = timeout
        return _MockResponse(csv_text)

    monkeypatch.setattr(pipeline, "urlopen", _fake_urlopen)

    result = pipeline.run_experiment(
        "csv_upload",
        {"seed": 42, "test_size": 0.25, "feature_grid": [1, 2, 3]},
        csv_blob_url="http://local.test/upload.csv?token=abc",
    )

    assert result["details"]["dataset_id"] == "csv_upload"
    assert len(result["details"]["grid_results"]) == 3
    assert 0.0 <= result["accuracy"] <= 1.0
