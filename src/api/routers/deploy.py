"""
/deploy/* endpoints — trigger deployment and query running instances.
"""
from __future__ import annotations

import json
import os

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

# Endpoint vars forwarded to the serving instance (public, not docker names)
_FORWARD_PUBLIC = {
    "MLFLOW_TRACKING_URI":    "PUBLIC_MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL": "PUBLIC_MINIO_ENDPOINT",
}
_FORWARD_AS_IS = ["MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"]


def _build_scaler() -> AutoScaler:
    """Instantiate AutoScaler from environment variables."""
    cfg = ScalerConfig(
        vast_api_key=os.environ["VAST_API_KEY"],
        gpu_template_id=os.environ.get("GPU_TEMPLATE_ID", ""),  # ""→auto-select
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


def _build_serve_onstart() -> str:
    """Bootstrap script the serving instance runs on boot (clone + vllm_server)."""
    work_dir = os.environ.get("REMOTE_WORK_DIR", "/workspace/OCR-Optimizing-Mlops")
    git_repo = os.environ["GIT_REPO_URL"]

    lines = [f"{v}={os.environ[v]}" for v in _FORWARD_AS_IS if os.environ.get(v)]
    for remote_name, src in _FORWARD_PUBLIC.items():
        if os.environ.get(src):
            lines.append(f"{remote_name}={os.environ[src]}")
    env_block = "\n".join(lines)

    return f"""#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null || (apt-get update && apt-get install -y git)
rm -rf {work_dir}
git clone --branch dev {git_repo} {work_dir}
cd {work_dir}
pip install -q -r requirements-serve.txt
cat > .env <<'ENVEOF'
{env_block}
ENVEOF
python src/serving/vllm_server.py --port 8000
"""


@router.post("/trigger", response_model=TriggerDeployResponse)
def trigger_deploy(
    request: TriggerDeployRequest,
    client=Depends(get_mlflow_client),
):
    """
    Deploy the Production model to a new vLLM instance on Vast.ai (onstart).

    1. Verify a Production version exists in MLflow Registry.
    2. Provision a Vast.ai GPU instance whose onstart clones + runs vllm_server.py
       (blocks until the instance is running, ~5 min).
    3. Register instance with scaler state, nginx upstream, prometheus targets.
    4. Return instance_id and address (host:mapped_port).
    """
    for var in ("VAST_API_KEY", "GIT_REPO_URL"):
        if not os.environ.get(var):
            raise HTTPException(status_code=503, detail=f"{var} not configured")

    versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
    production = [v for v in versions if v.current_stage == "Production"]
    if not production:
        raise HTTPException(
            status_code=404,
            detail=f"No Production version found for '{_MODEL_NAME}'. Run ci_gate first.",
        )

    scaler = _build_scaler()
    # Serving pulls the multi-GB vLLM image on first boot → require a fast link
    # (default 2000 Mbps floor) so the pull doesn't dominate deploy time.
    serve_min_inet = float(os.environ.get("VAST_SERVE_MIN_INET_MBPS", "2000"))
    instance = scaler._create_vast_instance(
        onstart=_build_serve_onstart(), min_inet=serve_min_inet
    )
    scaler.state.instances.append(instance)
    scaler.save_state()

    # Nginx upstream + Prometheus targets registration is part of the full
    # autoscale wiring (needs the config files + docker socket mounted into this
    # container). Best-effort for now so single-instance serving works; the
    # instance address is returned and can be hit directly until the LB is wired.
    try:
        scaler._write_nginx_upstream()
        scaler._write_prometheus_targets()
    except Exception:  # noqa: BLE001 — LB wiring comes with the autoscale phase
        pass

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
