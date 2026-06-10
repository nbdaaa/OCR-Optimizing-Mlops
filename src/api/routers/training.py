"""
/training/* endpoints — trigger and poll training jobs.

Provisioning uses Vast.ai `onstart`: the instance self-bootstraps on boot
(clone → install → run train.py) — no SSH from the control plane.
"""
from __future__ import annotations

import os
import time

import mlflow
import requests
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import (
    TriggerTrainingRequest,
    TriggerTrainingResponse,
    TrainingJobStatus,
    TrainingLogsResponse,
    TrainingJobBrief,
    TrainingJobsResponse,
)
from src.serving.scaler import _VAST_BASE

router = APIRouter(prefix="/training", tags=["training"])

_STATUS_MAP = {
    "RUNNING":  "running",
    "FINISHED": "completed",
    "FAILED":   "failed",
    "KILLED":   "failed",
}

_EXPERIMENT = "ocr-training"

# Vars forwarded to the remote .env unchanged (same name + value).
# WANDB_ENTITY intentionally NOT forwarded — let wandb resolve the API key's
# default entity (a hardcoded name causes "entity not found").
_FORWARD_AS_IS = [
    "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET_DATA",
    "HF_TOKEN", "HF_REPO_DATA",
    "WANDB_API_KEY", "WANDB_PROJECT",
]

# Endpoint vars: the remote can't resolve docker network names (minio/mlflow),
# so forward the PUBLIC_* value under the standard name the remote train.py reads.
_FORWARD_PUBLIC = {
    "MLFLOW_TRACKING_URI":    "PUBLIC_MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL": "PUBLIC_MINIO_ENDPOINT",
    "MINIO_ENDPOINT":         "PUBLIC_MINIO_ENDPOINT",
}


def _build_env_block() -> str:
    """Build the .env contents written on the remote instance (public endpoints)."""
    lines = [
        f"{var}={os.environ[var]}"
        for var in _FORWARD_AS_IS
        if os.environ.get(var)
    ]
    for remote_name, src in _FORWARD_PUBLIC.items():
        if os.environ.get(src):
            lines.append(f"{remote_name}={os.environ[src]}")
    return "\n".join(lines)


def _build_onstart(
    data_version: str, run_id: str, env_block: str,
    init_adapter_version: str | None, hyperparams: dict,
) -> str:
    """Bootstrap script the training instance runs on boot."""
    work_dir = os.environ.get("REMOTE_WORK_DIR", "/workspace/OCR-Optimizing-Mlops")
    git_repo = os.environ["GIT_REPO_URL"]
    args = f"--data-version {data_version} --run-id {run_id}"
    if init_adapter_version:
        args += f" --init-adapter-version {init_adapter_version}"
    for flag, val in hyperparams.items():
        if val is not None:
            args += f" {flag} {val}"
    return f"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update && apt-get install -y git)
rm -rf {work_dir}
git clone --branch dev {git_repo} {work_dir}
cd {work_dir}
pip install -q -r requirements-train.txt
cat > .env <<'ENVEOF'
{env_block}
ENVEOF
python -m src.training.train {args}
"""


def _provision_and_train(
    run_id: str, data_version: str, init_adapter_version: str | None,
    hyperparams: dict, gpu_template_id: str | None = None,
) -> None:
    """
    Background task: provision a Vast.ai GPU instance whose onstart script
    clones the repo, installs deps, and runs train.py with the pre-created
    run_id. Tags the MLflow run with the Vast instance id (for logs/status).
    Logs flow to MLflow + W&B; raw stdout to Vast logs.

    gpu_template_id: per-request offer ID; falls back to env, then auto-select.
    """
    from src.serving.scaler import AutoScaler, ScalerConfig

    # request offer ID > env GPU_TRAIN_TEMPLATE_ID > "" (auto-select)
    offer = (gpu_template_id or os.environ.get("GPU_TRAIN_TEMPLATE_ID", "") or "").strip()
    cfg = ScalerConfig(
        vast_api_key=os.environ["VAST_API_KEY"],
        gpu_template_id=offer,
        nginx_upstream_conf="",   # unused for training
        state_file="",            # unused for training
    )
    train_image = os.environ.get(
        "TRAIN_DOCKER_IMAGE", "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel"
    )
    onstart = _build_onstart(
        data_version, run_id, _build_env_block(), init_adapter_version, hyperparams
    )

    try:
        instance = AutoScaler(cfg)._create_vast_instance(image=train_image, onstart=onstart)
        c = mlflow.MlflowClient()
        c.set_tag(run_id, "vast_instance_id", instance["id"])
        c.set_tag(run_id, "vast_ssh_host", instance.get("ssh_host", ""))
        c.set_tag(run_id, "vast_ssh_port", instance.get("ssh_port", ""))
    except Exception as exc:
        mlflow.MlflowClient().set_tag(run_id, "provision_error", str(exc))
        raise


@router.post("/trigger", response_model=TriggerTrainingResponse)
def trigger_training(
    request: TriggerTrainingRequest,
    background_tasks: BackgroundTasks,
    client=Depends(get_mlflow_client),
):
    """
    Trigger a training job on a fresh Vast.ai GPU instance (onstart bootstrap).

    1. Validate required env vars.
    2. Pre-create an MLflow run → run_id.
    3. In background: provision instance with an onstart that runs train.py.
    4. Return run_id immediately as job_id.

    Poll /training/{job_id}/status for progress, /training/{job_id}/logs for raw output.
    """
    for var in ("VAST_API_KEY", "GIT_REPO_URL"):
        if not os.environ.get(var):
            raise HTTPException(status_code=503, detail=f"{var} not configured")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    experiment = mlflow.set_experiment(_EXPERIMENT)
    run = client.create_run(
        experiment_id=experiment.experiment_id,
        run_name=f"train-{request.data_version}",
    )
    run_id = run.info.run_id

    hyperparams = {
        "--num-epochs":   request.num_epochs,
        "--batch-size":   request.batch_size,
        "--grad-accum":   request.grad_accum,
        "--learning-rate": request.learning_rate,
    }
    background_tasks.add_task(
        _provision_and_train, run_id, request.data_version,
        request.init_adapter_version, hyperparams, request.gpu_template_id,
    )
    return TriggerTrainingResponse(job_id=run_id)


@router.get("/jobs", response_model=TrainingJobsResponse)
def list_jobs(client=Depends(get_mlflow_client)):
    """List recent training runs (job_id = run_id) in the ocr-training experiment,
    newest first — so the UI can find a running job from any session."""
    exp = client.get_experiment_by_name(_EXPERIMENT)
    if exp is None:
        return TrainingJobsResponse(jobs=[])
    runs = client.search_runs([exp.experiment_id], max_results=50)
    runs = sorted(runs, key=lambda r: r.info.start_time or 0, reverse=True)
    jobs = [
        TrainingJobBrief(
            job_id=r.info.run_id,
            status=_STATUS_MAP.get(r.info.status, r.info.status.lower()),
            data_version=r.data.params.get("data_version"),
        )
        for r in runs
    ]
    return TrainingJobsResponse(jobs=jobs)


@router.get("/{job_id}/status", response_model=TrainingJobStatus)
def get_training_status(job_id: str, client=Depends(get_mlflow_client)):
    """
    Poll status of a training job (job_id = MLflow run_id).
    Returns status, MLflow/W&B URLs, metrics, and the Vast instance id.
    Raises 404 if job_id is not found.
    """
    try:
        run = client.get_run(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_url = f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{job_id}"

    # Ready-to-run SSH command for LIVE logs (no lag, unlike the snapshot API)
    ssh_host = run.data.tags.get("vast_ssh_host")
    ssh_port = run.data.tags.get("vast_ssh_port")
    ssh_cmd = (
        f'ssh -p {ssh_port} root@{ssh_host} "tail -f /var/log/onstart.log"'
        if ssh_host and ssh_port else None
    )

    return TrainingJobStatus(
        job_id=job_id,
        status=_STATUS_MAP.get(run.info.status, run.info.status.lower()),
        mlflow_run_url=mlflow_url,
        wandb_run_url=run.data.tags.get("wandb_url"),
        metrics=dict(run.data.metrics) or None,
        vast_instance_id=run.data.tags.get("vast_instance_id"),
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        ssh_cmd=ssh_cmd,
    )


@router.get("/{job_id}/logs", response_model=TrainingLogsResponse)
def get_training_logs(job_id: str, tail: int = 200, client=Depends(get_mlflow_client)):
    """
    Fetch raw onstart stdout/stderr of the training instance via the Vast.ai
    logs API. Returns the last `tail` lines. Requires the run to have been
    tagged with vast_instance_id (set once provisioning succeeds).
    """
    try:
        run = client.get_run(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    instance_id = run.data.tags.get("vast_instance_id")
    if not instance_id:
        raise HTTPException(status_code=409, detail="Instance not provisioned yet")

    api_key = os.environ["VAST_API_KEY"]
    req = requests.put(
        f"{_VAST_BASE}/instances/request_logs/{instance_id}/",
        params={"api_key": api_key},
        json={"tail": str(tail)},
        timeout=15,
    )
    req.raise_for_status()
    url = req.json().get("result_url")

    logs = ""
    if url:
        # Vast uploads the log asynchronously — poll the result URL briefly.
        for _ in range(10):
            time.sleep(2)
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.text:
                logs = resp.text
                break

    return TrainingLogsResponse(job_id=job_id, instance_id=instance_id, logs=logs)
