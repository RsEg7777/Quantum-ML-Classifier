from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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
    job_id: Optional[str] = Field(default=None, alias="jobId")
    progress_callback_url: Optional[str] = Field(default=None, alias="progressCallbackUrl")
    progress_callback_token: Optional[str] = Field(default=None, alias="progressCallbackToken")

    model_config = {
        "populate_by_name": True,
    }


class WorkerResult(BaseModel):
    accuracy: float
    loss: float
    notes: str
    details: Dict[str, Any] = Field(default_factory=dict)
