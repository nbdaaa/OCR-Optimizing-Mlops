"""
/training/* endpoints — trigger and poll training jobs.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import (
    TriggerTrainingRequest,
    TriggerTrainingResponse,
    TrainingJobStatus,
)

router = APIRouter(prefix="/training", tags=["training"])

# MLflow run status → API status
_STATUS_MAP = {
    "RUNNING":  "running",
    "FINISHED": "completed",
    "FAILED":   "failed",
    "KILLED":   "failed",
}


@router.post("/trigger", response_model=TriggerTrainingResponse)
def trigger_training(request: TriggerTrainingRequest):
    """
    Start a training job on Vast.ai using the specified data version.

    Launches train.py on the GPU instance, returns MLflow run_id as job_id.
    The job runs asynchronously; poll /training/{job_id}/status for updates.
    """
    # Depends on train.py — implemented in a later phase (requires GPU).
    raise NotImplementedError


@router.get("/{job_id}/status", response_model=TrainingJobStatus)
def get_training_status(job_id: str, client=Depends(get_mlflow_client)):
    """
    Poll status of a running or completed training job.

    job_id = MLflow run_id.
    Returns status, MLflow run URL, W&B run URL, and latest metrics.
    Raises 404 if job_id is not found in MLflow.
    """
    try:
        run = client.get_run(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_url = f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{job_id}"
    wandb_url = run.data.tags.get("wandb_url")

    metrics = dict(run.data.metrics) or None

    return TrainingJobStatus(
        job_id=job_id,
        status=_STATUS_MAP.get(run.info.status, run.info.status.lower()),
        mlflow_run_url=mlflow_url,
        wandb_run_url=wandb_url,
        metrics=metrics,
    )
