"""
/deploy/* — control the Colab serving pool.

FastAPI only sets the desired floor in a shared state file; the host-side
`colab_pool_scaler.py` daemon reads it and does the actual provisioning,
teardown, autoscaling (1↔2) and nginx load-balancing.

    POST /deploy/trigger   → desired_floor = 1  (bring up pool, autoscale by load)
    POST /deploy/teardown  → desired_floor = 0  (stop all instances)
    GET  /deploy/status    → current pool status (instances, LB url, state)
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter

from src.api.schemas import DeployControlResponse, DeployStatusResponse, PoolInstance

router = APIRouter(prefix="/deploy", tags=["deploy"])

# Shared with the host scaler (mount the same file into this container).
STATE_FILE = os.environ.get("SERVE_STATE_FILE", "/opt/ocr/serve_state.json")


def _read() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _set_floor(n: int) -> None:
    st = _read()
    st["desired_floor"] = n
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(st, f, indent=2)


@router.post("/trigger", response_model=DeployControlResponse)
def trigger_deploy():
    """Deploy: ask the scaler to bring the pool up to ≥1 instance (autoscale 1↔2)."""
    _set_floor(1)
    return DeployControlResponse(desired_floor=1, status=_read().get("status", "deploying"))


@router.post("/teardown", response_model=DeployControlResponse)
def teardown_deploy():
    """Tear down: ask the scaler to stop all serving instances."""
    _set_floor(0)
    return DeployControlResponse(desired_floor=0, status="tearing_down")


@router.get("/status", response_model=DeployStatusResponse)
def deploy_status():
    """Current pool status as written by the scaler."""
    st = _read()
    return DeployStatusResponse(
        desired_floor=int(st.get("desired_floor", 0)),
        status=st.get("status", "unknown"),
        lb_url=st.get("lb_url", ""),
        instances=[PoolInstance(**i) for i in st.get("instances", [])],
    )
