from __future__ import annotations

from .schemas import CreateJobInput, WorkerResult
from .pipeline import run_experiment, run_inference, run_training


def run_job(payload: CreateJobInput) -> WorkerResult:
    try:
        if payload.job_type.value == "training":
            result = run_training(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
            )
        elif payload.job_type.value == "inference":
            result = run_inference(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
            )
        else:
            result = run_experiment(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
            )

        return WorkerResult(**result)
    except Exception as exc:
        return WorkerResult(
            accuracy=0.0,
            loss=0.0,
            notes=f"Worker failed: {exc}",
            details={"error": str(exc)},
        )
