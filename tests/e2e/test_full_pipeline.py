"""
E2E smoke test: data versioning → training → CI gate → model in Production.

Requires:
  - docker-compose stack running (MinIO, MLflow, FastAPI)
  - src/training/train.py implemented (skipped until then)
  - At least one GPU instance available on Vast.ai for the training step

Run with:
    pytest tests/e2e/ -m e2e -v
"""
import os
import time

import mlflow
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from src.api.main import app

load_dotenv()

pytestmark = pytest.mark.e2e

_SMOKE_VERSION = "e2e-smoke-v9999"
_MODEL_NAME = "granite-docling-adapter"
_SMOKE_MAX_SAMPLES = 50
_JOB_POLL_INTERVAL_S = 10
_JOB_TIMEOUT_S = 600  # 10 min — smoke training (1 step) should finish well within this


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def mlflow_client():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.MlflowClient()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wait_for_job(client: TestClient, job_id: str, timeout: int = _JOB_TIMEOUT_S) -> dict:
    """Poll /training/{job_id}/status until status is completed or failed."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/training/{job_id}/status")
        assert resp.status_code == 200, f"Unexpected status code: {resp.status_code}"
        body = resp.json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(_JOB_POLL_INTERVAL_S)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


# ── Step 1: Data versioning ───────────────────────────────────────────────────

@pytest.mark.skip(reason="train.py not yet implemented — enable after Phase 4")
class TestStep1DataVersion:
    @pytest.fixture(scope="class")
    def version_id(self, client):
        resp = client.post(
            "/data/versions/create",
            json={"version": _SMOKE_VERSION, "max_samples": _SMOKE_MAX_SAMPLES},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["version"]

    def test_version_created(self, client, version_id):
        resp = client.get(f"/data/versions/{version_id}")
        assert resp.status_code == 200

    def test_version_has_correct_count(self, client, version_id):
        resp = client.get(f"/data/versions/{version_id}")
        assert resp.json()["count"] <= _SMOKE_MAX_SAMPLES

    def test_version_filter_stats_present(self, client, version_id):
        resp = client.get(f"/data/versions/{version_id}")
        assert "filter_stats" in resp.json()


# ── Step 2: Training trigger ──────────────────────────────────────────────────

@pytest.mark.skip(reason="train.py not yet implemented — enable after Phase 4")
class TestStep2Training:
    @pytest.fixture(scope="class")
    def job(self, client):
        resp = client.post(
            "/training/trigger",
            json={"data_version": _SMOKE_VERSION},
        )
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        result = _wait_for_job(client, job_id)
        return result

    def test_job_completed(self, job):
        assert job["status"] == "completed", f"Training failed: {job}"

    def test_job_has_mlflow_url(self, job):
        assert job["mlflow_run_url"] is not None

    def test_job_has_cer_metric(self, job):
        assert job["metrics"] is not None
        assert "cer" in job["metrics"]

    def test_job_has_eval_loss_metric(self, job):
        assert "eval_loss" in job["metrics"]


# ── Step 3: MLflow Staging transition ────────────────────────────────────────

@pytest.mark.skip(reason="train.py not yet implemented — enable after Phase 4")
class TestStep3StagingTransition:
    @pytest.fixture(scope="class")
    def staging_version(self, mlflow_client):
        """Find the most recent Staging version of granite-docling-adapter."""
        versions = mlflow_client.search_model_versions(f"name='{_MODEL_NAME}'")
        staging = [v for v in versions if v.current_stage == "Staging"]
        assert staging, "No Staging version found after training"
        return staging[-1]

    def test_staging_version_exists(self, staging_version):
        assert staging_version is not None

    def test_staging_version_has_run_id(self, staging_version):
        assert staging_version.run_id is not None


# ── Step 4: CI gate → Production ─────────────────────────────────────────────

@pytest.mark.skip(reason="train.py not yet implemented — enable after Phase 4")
class TestStep4CIGate:
    @pytest.fixture(scope="class")
    def post_gate_version(self, mlflow_client):
        """
        Run ci_gate directly (same as CI would). Returns the version object
        after the gate has executed.
        """
        from src.cicd.ci_gate import run_ci_gate

        versions = mlflow_client.search_model_versions(f"name='{_MODEL_NAME}'")
        staging = [v for v in versions if v.current_stage == "Staging"]
        assert staging, "No Staging version to gate"
        version_number = int(staging[-1].version)

        run_ci_gate(version_number)

        # Re-fetch to see updated stage
        return mlflow_client.get_model_version(_MODEL_NAME, str(version_number))

    def test_version_not_in_staging_after_gate(self, post_gate_version):
        assert post_gate_version.current_stage != "Staging"

    def test_version_transitioned_to_production_or_archived(self, post_gate_version):
        assert post_gate_version.current_stage in ("Production", "Archived")

    def test_production_version_exists(self, mlflow_client):
        versions = mlflow_client.search_model_versions(f"name='{_MODEL_NAME}'")
        production = [v for v in versions if v.current_stage == "Production"]
        assert production, "Expected at least one Production version after CI gate"
