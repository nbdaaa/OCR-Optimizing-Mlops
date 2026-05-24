"""
/deploy/* endpoints — trigger deployment and query running instances.
"""
from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    DeployStatusResponse,
    InstanceInfo,
    TriggerDeployRequest,
    TriggerDeployResponse,
)

router = APIRouter(prefix="/deploy", tags=["deploy"])

_STATE_FILE = os.environ.get("SCALER_STATE_FILE", "scaler_state.json")


@router.post("/trigger", response_model=TriggerDeployResponse)
def trigger_deploy(request: TriggerDeployRequest):
    """
    Deploy the Production model (or a specific version) to a new vLLM instance.

    Pulls LoRA adapter from MLflow Registry (stage=Production),
    launches vLLM on Vast.ai, registers instance with the scaler.
    """
    # Depends on scaler.py — implemented in a later phase.
    raise NotImplementedError


@router.get("/status", response_model=DeployStatusResponse)
def get_deploy_status():
    """
    Return the list of currently running vLLM instances managed by the scaler.
    """
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
