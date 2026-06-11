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
    # Vast.ai offer ID to pin a specific machine; None/empty → env default or auto-select
    gpu_template_id: str | None = None
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


class TrainingJobBrief(BaseModel):
    job_id: str
    status: str
    data_version: str | None = None


class TrainingJobsResponse(BaseModel):
    jobs: list[TrainingJobBrief]


class RecoverRequest(BaseModel):
    gpu_template_id: str | None = None   # pin a machine for the recovery instance


class RecoverResponse(BaseModel):
    job_id: str
    action: str          # DONE | REGISTER_ONLY | FINALIZE | RESUME_TRAIN
    provisioned: bool    # True if a GPU instance was launched to finish recovery


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
    staging_cer: float                   # benchmark CER (absolute floor check)
    production_cer: float | None = None
    staging_loss: float | None = None    # val eval_loss (regression check)
    production_loss: float | None = None
    new_stage: str                       # "Production" | "Archived"


# ── Deploy ────────────────────────────────────────────────────────────────────

class DeployControlResponse(BaseModel):
    desired_floor: int        # 1 = deployed (pool min 1, autoscale 1↔2), 0 = sleep
    status: str               # deploying | active | starting | sleeping | tearing_down | ...


class PoolInstance(BaseModel):
    name: str
    tunnel_url: str | None = None
    ready: bool = False
    age_s: int = 0


class DeployStatusResponse(BaseModel):
    desired_floor: int
    status: str
    lb_url: str = ""
    instances: list[PoolInstance] = []
