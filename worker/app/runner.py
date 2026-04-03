from __future__ import annotations

import json
from typing import Callable, Optional
from urllib.request import Request, urlopen

from .schemas import CreateJobInput, WorkerResult
from .pipeline import run_experiment, run_inference, run_training


ProgressEmitter = Callable[[int, str, str | None, list[dict[str, str]] | None], None]

STEP_LABELS = [
    "Validate request",
    "Load and prepare data",
    "Run model execution",
    "Aggregate metrics",
    "Finalize and persist",
]


def _build_steps(
    *,
    active_index: Optional[int] = None,
    error_index: Optional[int] = None,
) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []

    for index, label in enumerate(STEP_LABELS):
        if error_index is not None:
            if index < error_index:
                state = "done"
            elif index == error_index:
                state = "error"
            else:
                state = "pending"
        elif active_index is None:
            state = "pending"
        elif index < active_index:
            state = "done"
        elif index == active_index:
            state = "active"
        else:
            state = "pending"

        steps.append({"label": label, "state": state})

    return steps


def _create_progress_emitter(payload: CreateJobInput) -> Optional[ProgressEmitter]:
    if not payload.progress_callback_url or not payload.job_id:
        return None

    callback_url = payload.progress_callback_url
    callback_token = payload.progress_callback_token
    job_id = payload.job_id

    def emit(
        percent: int,
        stage: str,
        message: str | None = None,
        steps: list[dict[str, str]] | None = None,
    ) -> None:
        bounded_percent = max(0, min(100, int(percent)))
        body = {
            "id": job_id,
            "progress": {
                "percent": bounded_percent,
                "stage": stage,
                "message": message,
                "steps": steps,
            },
        }

        headers = {
            "Content-Type": "application/json",
        }
        if callback_token:
            headers["x-dispatch-secret"] = callback_token

        try:
            request = Request(
                callback_url,
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urlopen(request, timeout=1.5) as response:
                response.read()
        except Exception:
            # Progress callbacks are best-effort and should never fail the job.
            return

    return emit


def run_job(payload: CreateJobInput) -> WorkerResult:
    progress = _create_progress_emitter(payload)

    try:
        if progress is not None:
            progress(
                8,
                "Validate request",
                "Worker accepted the job.",
                _build_steps(active_index=0),
            )

        if payload.job_type.value == "training":
            result = run_training(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
                progress_callback=progress,
            )
        elif payload.job_type.value == "inference":
            result = run_inference(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
                progress_callback=progress,
            )
        else:
            result = run_experiment(
                payload.dataset_id.value,
                payload.config,
                payload.csv_blob_url,
                progress_callback=progress,
            )

        if progress is not None:
            progress(
                100,
                "Completed",
                "Worker execution completed.",
                _build_steps(active_index=len(STEP_LABELS)),
            )

        return WorkerResult(**result)
    except Exception as exc:
        if progress is not None:
            progress(
                100,
                "Execution failed",
                f"Worker failed: {exc}",
                _build_steps(error_index=2),
            )

        return WorkerResult(
            accuracy=0.0,
            loss=0.0,
            notes=f"Worker failed: {exc}",
            details={"error": str(exc)},
        )
