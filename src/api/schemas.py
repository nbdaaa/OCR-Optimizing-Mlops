"""
Pydantic request/response schemas for all API endpoints.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Data versioning ───────────────────────────────────────────────────────────

class CreateVersionRequest(BaseModel):
    version: str
    max_samples: int | None = None


class VersionInfo(BaseModel):
    version: str
    count: int
    # Optional so the endpoint tolerates minimal metadata (e.g. the benchmark
    # set, whose metadata.json only has version/count/source_version/note).
    hf_repo: str | None = None
    split: str | None = None
    created_at: str | None = None
    filter_stats: dict[str, Any] = Field(default_factory=dict)
    offset: int = 0
    mlflow_run_id: str | None = None
    source_version: str | None = None
    note: str | None = None


class CreateVersionResponse(BaseModel):
    version: str
    mlflow_run_id: str


class VersionListResponse(BaseModel):
    versions: list[str]


# ── Training ──────────────────────────────────────────────────────────────────

class TriggerTrainingRequest(BaseModel):
    data_version: str
    init_adapter_version: str | None = None   # continual warm-start; None = latest
    # Optional hyperparameter overrides (None → use TrainConfig defaults)
    num_epochs: int | None = None
    batch_size: int | None = None
    grad_accum: int | None = None
    learning_rate: float | None = None


class TriggerTrainingResponse(BaseModel):
    job_id: str


class TrainingJobStatus(BaseModel):
    job_id: str
    status: str                   # "running" | "completed" | "failed"
    mlflow_run_url: str | None = None
    wandb_run_url: str | None = None
    metrics: dict[str, float] | None = None
    vast_instance_id: str | None = None
    ssh_host: str | None = None
    ssh_port: str | None = None
    ssh_cmd: str | None = None   # ready-to-run SSH tail for live logs


class TrainingLogsResponse(BaseModel):
    job_id: str
    instance_id: str
    logs: str


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


# ── CI/CD gate ────────────────────────────────────────────────────────────────

class CIGateRequest(BaseModel):
    cer_threshold: float = 0.15
    regression_tolerance: float = 1.05


class CIGateResponse(BaseModel):
    result: str                          # "pass" | "fail_cer" | "fail_regression"
    staging_version: int
    staging_cer: float
    production_cer: float | None = None
    new_stage: str                       # "Production" | "Archived"


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
