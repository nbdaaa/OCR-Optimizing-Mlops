"""
Pydantic request/response schemas for all API endpoints.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel


# ── Data versioning ───────────────────────────────────────────────────────────

class CreateVersionRequest(BaseModel):
    version: str
    max_samples: int | None = None


class VersionInfo(BaseModel):
    version: str
    count: int
    hf_repo: str
    split: str
    created_at: str
    filter_stats: dict[str, Any]
    mlflow_run_id: str | None = None


class CreateVersionResponse(BaseModel):
    version: str
    mlflow_run_id: str


class VersionListResponse(BaseModel):
    versions: list[str]


# ── Training ──────────────────────────────────────────────────────────────────

class TriggerTrainingRequest(BaseModel):
    data_version: str


class TriggerTrainingResponse(BaseModel):
    job_id: str


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str                   # "running" | "completed" | "failed"
    mlflow_run_url: str | None = None
    wandb_run_url: str | None = None
    metrics: dict[str, float] | None = None


# ── Model registry ────────────────────────────────────────────────────────────

class ModelVersionInfo(BaseModel):
    version: int
    stage: str
    run_id: str
    metrics: dict[str, float] | None = None
    params: dict[str, str] | None = None
    created_at: str | None = None


class ModelVersionListResponse(BaseModel):
    versions: list[ModelVersionInfo]


# ── Deploy ────────────────────────────────────────────────────────────────────

class TriggerDeployRequest(BaseModel):
    model_version: int | None = None   # None = use current Production


class TriggerDeployResponse(BaseModel):
    instance_id: str
    address: str


class InstanceInfo(BaseModel):
    instance_id: str
    address: str
    status: str


class DeployStatusResponse(BaseModel):
    instances: list[InstanceInfo]
