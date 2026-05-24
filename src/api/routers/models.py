"""
/models/versions/* endpoints — query MLflow Model Registry.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_mlflow_client
from src.api.schemas import ModelVersionInfo, ModelVersionListResponse

router = APIRouter(prefix="/models", tags=["models"])

_MODEL_NAME = "granite-docling-adapter"


def _to_version_info(mv, client) -> ModelVersionInfo:
    metrics, params = None, None
    try:
        run = client.get_run(mv.run_id)
        metrics = dict(run.data.metrics) or None
        params = dict(run.data.params) or None
    except Exception:
        pass

    return ModelVersionInfo(
        version=int(mv.version),
        stage=mv.current_stage,
        run_id=mv.run_id,
        metrics=metrics,
        params=params,
        created_at=str(mv.creation_timestamp),
    )


@router.get("/versions", response_model=ModelVersionListResponse)
def list_model_versions(client=Depends(get_mlflow_client)):
    """
    List all registered versions of granite-docling-adapter from MLflow Registry.
    Includes stage (None / Staging / Production / Archived) and metrics.
    """
    versions = client.search_model_versions(f"name='{_MODEL_NAME}'")
    return ModelVersionListResponse(
        versions=[_to_version_info(mv, client) for mv in versions]
    )


@router.get("/versions/{version_id}", response_model=ModelVersionInfo)
def get_model_version(version_id: int, client=Depends(get_mlflow_client)):
    """
    Return metadata and metrics for a specific model version.
    Raises 404 if version_id does not exist in MLflow Registry.
    """
    try:
        mv = client.get_model_version(name=_MODEL_NAME, version=str(version_id))
    except Exception:
        raise HTTPException(status_code=404, detail=f"Model version {version_id} not found")

    return _to_version_info(mv, client)
