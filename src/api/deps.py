"""
Shared FastAPI dependencies — MinIO and MLflow client factories.
"""
from __future__ import annotations

import os

import boto3
import mlflow


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )


def get_mlflow_client() -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    return mlflow.MlflowClient()
