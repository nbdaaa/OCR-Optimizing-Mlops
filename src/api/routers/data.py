"""
/data/versions/* endpoints — trigger and query data versioning.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from src.api.deps import get_minio_client
from src.api.schemas import (
    CreateVersionRequest,
    CreateVersionResponse,
    VersionInfo,
    VersionListResponse,
)
from src.data.data_versioning import create_version as _create_version

router = APIRouter(prefix="/data", tags=["data"])

_BUCKET = "ocr-data"


@router.post("/versions/create", response_model=CreateVersionResponse)
def create_version(request: CreateVersionRequest):
    """
    Trigger creation of a new data version.

    Downloads from HF Hub, filters, deduplicates, uploads to MinIO,
    and logs lineage to MLflow — all in a background task.
    """
    metadata = _create_version(
        version=request.version,
        max_samples=request.max_samples,
    )
    return CreateVersionResponse(
        version=metadata["version"],
        mlflow_run_id=metadata["mlflow_run_id"],
    )


@router.get("/versions", response_model=VersionListResponse)
def list_versions(minio=Depends(get_minio_client)):
    """List all data versions available in MinIO (ocr-data bucket)."""
    response = minio.list_objects_v2(Bucket=_BUCKET, Delimiter="/")
    prefixes = response.get("CommonPrefixes", [])
    versions = [p["Prefix"].rstrip("/") for p in prefixes]
    return VersionListResponse(versions=versions)


@router.get("/versions/{version_id}", response_model=VersionInfo)
def get_version(version_id: str, minio=Depends(get_minio_client)):
    """
    Return metadata for a specific data version.
    Raises 404 if version_id does not exist in MinIO.
    """
    try:
        obj = minio.get_object(Bucket=_BUCKET, Key=f"{version_id}/metadata.json")
        metadata = json.loads(obj["Body"].read())
    except minio.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")
    except Exception:
        raise HTTPException(status_code=404, detail=f"Version '{version_id}' not found")

    return VersionInfo(**metadata)
