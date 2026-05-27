"""
/training/* endpoints — trigger and poll training jobs.
"""
from __future__ import annotations

import os
import subprocess

import mlflow
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import (
    TriggerTrainingRequest,
    TriggerTrainingResponse,
    TrainingJobStatus,
)

router = APIRouter(prefix="/training", tags=["training"])

_STATUS_MAP = {
    "RUNNING":  "running",
    "FINISHED": "completed",
    "FAILED":   "failed",
    "KILLED":   "failed",
}

_EXPERIMENT = "ocr-training"


def _ssh_launch(run_id: str, data_version: str) -> None:
    """Fire-and-forget: SSH into the Vast.ai training instance and start train.py."""
    host     = os.environ["VAST_TRAIN_HOST"]
    port     = os.environ.get("VAST_TRAIN_PORT", "22")
    user     = os.environ.get("VAST_TRAIN_USER", "root")
    key      = os.environ.get("VAST_TRAIN_KEY", os.path.expanduser("~/.ssh/id_rsa"))
    work_dir = os.environ.get("REMOTE_WORK_DIR", "/workspace/OCR-Optimizing-Mlops")

    remote_cmd = (
        f"cd {work_dir} && "
        f"nohup python src/training/train.py "
        f"--data-version {data_version} --run-id {run_id} "
        f"> /tmp/train_{run_id}.log 2>&1 &"
    )
    subprocess.Popen([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-p", port,
        "-i", key,
        f"{user}@{host}",
        remote_cmd,
    ])


@router.post("/trigger", response_model=TriggerTrainingResponse)
def trigger_training(
    request: TriggerTrainingRequest,
    background_tasks: BackgroundTasks,
    client=Depends(get_mlflow_client),
):
    """
    Start a training job on Vast.ai using the specified data version.

    Pre-creates an MLflow run (RUNNING), SSHes into the GPU instance in the
    background, and returns the run_id immediately as job_id.
    Poll /training/{job_id}/status to track progress.
    """
    if not os.environ.get("VAST_TRAIN_HOST"):
        raise HTTPException(status_code=503, detail="VAST_TRAIN_HOST not configured")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment = mlflow.set_experiment(_EXPERIMENT)
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"train-{request.data_version}",
    )
    run_id = run.info.run_id

    background_tasks.add_task(_ssh_launch, run_id, request.data_version)
    return TriggerTrainingResponse(job_id=run_id)


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
