"""
Integration tests: MinIO upload + MLflow log_input round-trip.
Requires docker-compose services (MinIO + MLflow) to be running.
"""
import io
import json
import os
import shutil
import tempfile

import mlflow
import pandas as pd
import pytest

from src.data.data_versioning import build_metadata, log_to_mlflow, upload_to_minio

BUCKET = "ocr-data"
_VERSION_UPLOAD = "test-minio-upload-v9999"
_VERSION_RT = "test-roundtrip-v9999"


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_parquet(tmp_path_factory):
    """Minimal valid parquet written to a temp file."""
    tmp = tmp_path_factory.mktemp("parquet")
    path = str(tmp / "dataset.parquet")
    pd.DataFrame([{
        "sample_id": "s1",
        "output_text": "<doctag><loc_1>hello</loc_1></doctag>",
    }]).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="module")
def sample_metadata():
    return build_metadata(
        version=_VERSION_UPLOAD,
        count=1,
        hf_repo="nbdaaa/all-ocr-data",
        filter_stats={"total": 2, "rejected_invalid": 1, "final_count": 1},
        split="train",
    )


# ── MinIO upload tests ────────────────────────────────────────────────────────

class TestMinIOUpload:
    @pytest.fixture(scope="class", autouse=True)
    def do_upload(self, minio, sample_parquet, sample_metadata):
        """Upload once; delete both objects after all class tests finish."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(sample_parquet, os.path.join(tmp, "dataset.parquet"))
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                json.dump(sample_metadata, f)
            upload_to_minio(tmp, _VERSION_UPLOAD)
        yield
        for key in (f"{_VERSION_UPLOAD}/dataset.parquet", f"{_VERSION_UPLOAD}/metadata.json"):
            try:
                minio.delete_object(Bucket=BUCKET, Key=key)
            except Exception:
                pass

    def test_parquet_exists(self, minio):
        resp = minio.head_object(Bucket=BUCKET, Key=f"{_VERSION_UPLOAD}/dataset.parquet")
        assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_metadata_exists(self, minio):
        resp = minio.head_object(Bucket=BUCKET, Key=f"{_VERSION_UPLOAD}/metadata.json")
        assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_metadata_content_matches(self, minio):
        resp = minio.get_object(Bucket=BUCKET, Key=f"{_VERSION_UPLOAD}/metadata.json")
        content = json.loads(resp["Body"].read())
        assert content["version"] == _VERSION_UPLOAD
        assert content["count"] == 1
        assert content["hf_repo"] == "nbdaaa/all-ocr-data"

    def test_parquet_is_readable(self, minio):
        resp = minio.get_object(Bucket=BUCKET, Key=f"{_VERSION_UPLOAD}/dataset.parquet")
        df = pd.read_parquet(io.BytesIO(resp["Body"].read()))
        assert len(df) == 1

    def test_filter_stats_in_metadata(self, minio):
        resp = minio.get_object(Bucket=BUCKET, Key=f"{_VERSION_UPLOAD}/metadata.json")
        content = json.loads(resp["Body"].read())
        assert "filter_stats" in content
        assert content["filter_stats"]["final_count"] == 1


# ── MLflow log_input tests ────────────────────────────────────────────────────

class TestMLflowLogInput:
    @pytest.fixture(scope="class")
    def run_id(self, mlflow_client, sample_parquet, sample_metadata):
        """Log a real MLflow run; delete it after all class tests finish."""
        rid = log_to_mlflow(_VERSION_UPLOAD, sample_metadata, sample_parquet)
        yield rid
        try:
            mlflow_client.delete_run(rid)
        except Exception:
            pass

    def test_run_is_created(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert run is not None

    def test_run_status_is_finished(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_version_param_stored(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert run.data.params["version"] == _VERSION_UPLOAD

    def test_hf_repo_param_stored(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert run.data.params["hf_repo"] == "nbdaaa/all-ocr-data"

    def test_count_param_stored(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert run.data.params["count"] == "1"

    def test_dataset_input_logged(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        assert len(run.inputs.dataset_inputs) >= 1

    def test_dataset_context_is_data_versioning(self, mlflow_client, run_id):
        run = mlflow_client.get_run(run_id)
        contexts = [
            tag.value
            for di in run.inputs.dataset_inputs
            for tag in di.tags
            if tag.key == "mlflow.data.context"
        ]
        assert "data_versioning" in contexts

    def test_artifact_stored_under_version_path(self, mlflow_client, run_id):
        artifacts = mlflow_client.list_artifacts(run_id)
        names = [a.path for a in artifacts]
        assert any(_VERSION_UPLOAD in name for name in names)


# ── Round-trip test ───────────────────────────────────────────────────────────

class TestRoundTrip:
    """Upload to MinIO + log to MLflow; verify both are mutually consistent."""

    @pytest.fixture(scope="class")
    def rt_artifacts(self, minio, mlflow_client, sample_parquet):
        """Do full round-trip upload; return (run_id, metadata); cleanup after."""
        metadata = build_metadata(
            version=_VERSION_RT,
            count=1,
            hf_repo="nbdaaa/all-ocr-data",
            filter_stats={"total": 1, "rejected_invalid": 0, "final_count": 1},
            split="train",
        )
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(sample_parquet, os.path.join(tmp, "dataset.parquet"))
            with open(os.path.join(tmp, "metadata.json"), "w") as f:
                json.dump(metadata, f)
            upload_to_minio(tmp, _VERSION_RT)
        rid = log_to_mlflow(_VERSION_RT, metadata, sample_parquet)
        yield rid, metadata
        try:
            mlflow_client.delete_run(rid)
        except Exception:
            pass
        for key in (f"{_VERSION_RT}/dataset.parquet", f"{_VERSION_RT}/metadata.json"):
            try:
                minio.delete_object(Bucket=BUCKET, Key=key)
            except Exception:
                pass

    def test_minio_metadata_exists(self, minio, rt_artifacts):
        resp = minio.head_object(Bucket=BUCKET, Key=f"{_VERSION_RT}/metadata.json")
        assert resp["ResponseMetadata"]["HTTPStatusCode"] == 200

    def test_mlflow_run_references_correct_version(self, mlflow_client, rt_artifacts):
        rid, _ = rt_artifacts
        run = mlflow_client.get_run(rid)
        assert run.data.params["version"] == _VERSION_RT

    def test_minio_and_mlflow_version_consistent(self, minio, mlflow_client, rt_artifacts):
        rid, _ = rt_artifacts
        resp = minio.get_object(Bucket=BUCKET, Key=f"{_VERSION_RT}/metadata.json")
        minio_meta = json.loads(resp["Body"].read())
        run = mlflow_client.get_run(rid)
        assert minio_meta["version"] == run.data.params["version"]
        assert minio_meta["hf_repo"] == run.data.params["hf_repo"]

    def test_minio_and_mlflow_count_consistent(self, minio, mlflow_client, rt_artifacts):
        rid, _ = rt_artifacts
        resp = minio.get_object(Bucket=BUCKET, Key=f"{_VERSION_RT}/metadata.json")
        minio_meta = json.loads(resp["Body"].read())
        run = mlflow_client.get_run(rid)
        assert str(minio_meta["count"]) == run.data.params["count"]
