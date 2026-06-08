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

# Env vars forwarded to the training instance's .env file
_FORWARDED_VARS = [
    "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_DATA",
    "MLFLOW_TRACKING_URI", "MLFLOW_S3_ENDPOINT_URL",
    "HF_TOKEN", "HF_REPO_DATA",
    "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY",
]


def _build_env_block() -> str:
    """Collect relevant env vars to write as .env on the remote instance."""
    return "\n".join(
        f"{var}={os.environ[var]}"
        for var in _FORWARDED_VARS
        if os.environ.get(var)
    )


def _provision_and_train(run_id: str, data_version: str) -> None:
    """
    Background task:
      1. Provision a Vast.ai GPU instance from GPU_TRAIN_TEMPLATE_ID.
      2. SSH in, clone repo, install deps, write .env.
      3. Launch train.py with the pre-created run_id (non-blocking).
    Logs flow to MLflow + W&B automatically via train.py.
    """
    from src.serving.scaler import AutoScaler, ScalerConfig

    # 1. Provision Vast.ai training instance
    cfg = ScalerConfig(
        vast_api_key=os.environ["VAST_API_KEY"],
        gpu_template_id=os.environ["GPU_TRAIN_TEMPLATE_ID"],
        nginx_upstream_conf="",   # unused for training
        state_file="",            # unused for training
    )
    train_image = os.environ.get(
        "TRAIN_DOCKER_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    )
    instance = AutoScaler(cfg)._create_vast_instance(image=train_image)
    host     = instance["address"].split(":")[0]
    ssh_port = instance["ssh_port"]

    key      = os.environ.get("VAST_TRAIN_KEY", os.path.expanduser("~/.ssh/id_rsa"))
    work_dir = os.environ.get("REMOTE_WORK_DIR", "/workspace/OCR-Optimizing-Mlops")
    git_repo = os.environ["GIT_REPO_URL"]
    user     = os.environ.get("VAST_TRAIN_USER", "root")

    # 2. Build remote setup + launch script
    env_block = _build_env_block()
    script = f"""set -e
if [ -d {work_dir}/.git ]; then
    git -C {work_dir} pull
else
    git clone {git_repo} {work_dir}
fi
pip install -q -r {work_dir}/requirements-train.txt
cat > {work_dir}/.env << 'ENVEOF'
{env_block}
ENVEOF
cd {work_dir}
nohup python src/training/train.py \\
    --data-version {data_version} \\
    --run-id {run_id} \\
    > /tmp/train_{run_id}.log 2>&1 &
"""

    subprocess.Popen([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-p", ssh_port,
        "-i", key,
        f"{user}@{host}",
        script,
    ])


@router.post("/trigger", response_model=TriggerTrainingResponse)
def trigger_training(
    request: TriggerTrainingRequest,
    background_tasks: BackgroundTasks,
    client=Depends(get_mlflow_client),
):
    """
    Trigger a training job on a fresh Vast.ai GPU instance.

    Flow:
      1. Validate required env vars.
      2. Pre-create an MLflow run (RUNNING) → get run_id.
      3. In background: provision instance → clone repo → install deps → run train.py.
      4. Return run_id immediately as job_id.

    Poll /training/{job_id}/status to track progress via MLflow.
    Training logs also appear in W&B (project: WANDB_PROJECT).
    """
    for var in ("VAST_API_KEY", "GPU_TRAIN_TEMPLATE_ID", "GIT_REPO_URL"):
        if not os.environ.get(var):
            raise HTTPException(status_code=503, detail=f"{var} not configured")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment = mlflow.set_experiment(_EXPERIMENT)
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"train-{request.data_version}",
    )
    run_id = run.info.run_id

    background_tasks.add_task(_provision_and_train, run_id, request.data_version)
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
