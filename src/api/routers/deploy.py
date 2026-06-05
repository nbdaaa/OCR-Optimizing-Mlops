"""
/deploy/* endpoints — trigger deployment and query running instances.
"""
from __future__ import annotations

import json
import os
import subprocess

import mlflow
from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import (
    DeployStatusResponse,
    InstanceInfo,
    TriggerDeployRequest,
    TriggerDeployResponse,
)
from src.serving.scaler import AutoScaler, ScalerConfig

router = APIRouter(prefix="/deploy", tags=["deploy"])

_STATE_FILE = os.environ.get("SCALER_STATE_FILE", "scaler_state.json")
_MODEL_NAME = "granite-docling-adapter"


def _build_scaler() -> AutoScaler:
    """Instantiate AutoScaler from environment variables."""
    cfg = ScalerConfig(
        vast_api_key=os.environ["VAST_API_KEY"],
        gpu_template_id=os.environ["GPU_TEMPLATE_ID"],
        nginx_upstream_conf=os.environ.get(
            "NGINX_UPSTREAM_CONF", "infra/nginx/upstream.conf"
        ),
        state_file=_STATE_FILE,
        prometheus_targets_file=os.environ.get(
            "PROMETHEUS_TARGETS_FILE", "infra/prometheus/targets.json"
        ),
        nginx_container_name=os.environ.get("NGINX_CONTAINER_NAME", "infra-nginx-1"),
    )
    scaler = AutoScaler(cfg)
    scaler.load_state()
    return scaler


def _ssh_start_vllm(host: str, ssh_port: str, instance_id: str) -> None:
    """Fire-and-forget: SSH into the serving instance and start vllm_server.py."""
    user     = os.environ.get("VAST_TRAIN_USER", "root")
    key      = os.environ.get("VAST_TRAIN_KEY", os.path.expanduser("~/.ssh/id_rsa"))
    work_dir = os.environ.get("REMOTE_WORK_DIR", "/workspace/OCR-Optimizing-Mlops")

    remote_cmd = (
        f"cd {work_dir} && "
        f"nohup python src/serving/vllm_server.py --port 8000 "
        f"> /tmp/vllm_{instance_id}.log 2>&1 &"
    )
    subprocess.Popen([
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-p", ssh_port,
        "-i", key,
        f"{user}@{host}",
        remote_cmd,
    ])


@router.post("/trigger", response_model=TriggerDeployResponse)
def trigger_deploy(
    request: TriggerDeployRequest,
    client=Depends(get_mlflow_client),
):
    """
    Deploy the Production model to a new vLLM instance on Vast.ai.

    Flow:
      1. Verify a Production version exists in MLflow Registry.
      2. Provision a Vast.ai GPU instance (blocks until running, ~5 min).
      3. SSH in and start vllm_server.py in background (non-blocking).
      4. Register instance with scaler state, nginx upstream, prometheus targets.
      5. Return instance_id and address.
    """
    for var in ("VAST_API_KEY", "GPU_TEMPLATE_ID"):
        if not os.environ.get(var):
            raise HTTPException(status_code=503, detail=f"{var} not configured")

    # 1. Verify Production version exists
    versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise HTTPException(
            status_code=404,
            detail=f"No Production version found for '{_MODEL_NAME}'. Run ci_gate first.",
        )

    # 2. Provision Vast.ai instance (blocking — waits until "running")
    scaler = _build_scaler()
    instance = scaler._create_vast_instance()   # {"id", "address", "ssh_port"}
    host = instance["address"].split(":")[0]

    # 3. SSH in and start vllm_server.py (fire-and-forget)
    _ssh_start_vllm(host, instance["ssh_port"], instance["id"])

    # 4. Register with scaler state + nginx + prometheus
    scaler.state.instances.append(instance)
    scaler._write_nginx_upstream()
    scaler._write_prometheus_targets()
    scaler.save_state()

    # 5. Return
    return TriggerDeployResponse(
        instance_id=instance["id"],
        address=instance["address"],
    )


@router.get("/status", response_model=DeployStatusResponse)
def get_deploy_status():
    """Return the list of currently running vLLM instances managed by the scaler."""
    if not os.path.exists(_STATE_FILE):
        return DeployStatusResponse(instances=[])

    with open(_STATE_FILE) as f:
        state = json.load(f)

    instances = [
        InstanceInfo(
            instance_id=inst["id"],
            address=inst["address"],
            status="running",
        )
        for inst in state.get("instances", [])
    ]
    return DeployStatusResponse(instances=instances)
