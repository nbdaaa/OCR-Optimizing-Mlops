"""
Integration tests for /data/* endpoints — calls real MinIO service.
"""
import json

import pytest

BUCKET = "ocr-data"
TEST_VERSION = "test-integration-v9999"
TEST_METADATA = {
    "version": TEST_VERSION,
    "count": 5,
    "hf_repo": "nbdaaa/all-ocr-data",
    "split": "train",
    "created_at": "2026-01-01T00:00:00+00:00",
    "filter_stats": {"total": 10, "rejected_invalid": 5, "final_count": 5},
    "mlflow_run_id": "test-run-integration-001",
}


@pytest.fixture(autouse=True, scope="module")
def upload_test_version(minio):
    """Upload test metadata to MinIO once; remove after all tests in this module."""
    minio.put_object(
        Bucket=BUCKET,
        Key=f"{TEST_VERSION}/metadata.json",
        Body=json.dumps(TEST_METADATA).encode(),
    )
    yield
    try:
        minio.delete_object(Bucket=BUCKET, Key=f"{TEST_VERSION}/metadata.json")
    except Exception:
        pass


class TestListVersions:
    def test_returns_200(self, client):
        resp = client.get("/data/versions")
        assert resp.status_code == 200

    def test_response_has_versions_key(self, client):
        resp = client.get("/data/versions")
        assert "versions" in resp.json()

    def test_includes_test_version(self, client):
        resp = client.get("/data/versions")
        assert TEST_VERSION in resp.json()["versions"]


class TestGetVersion:
    def test_returns_200(self, client):
        resp = client.get(f"/data/versions/{TEST_VERSION}")
        assert resp.status_code == 200

    def test_returns_correct_metadata(self, client):
        resp = client.get(f"/data/versions/{TEST_VERSION}")
        body = resp.json()
        assert body["version"] == TEST_VERSION
        assert body["count"] == 5
        assert body["hf_repo"] == "nbdaaa/all-ocr-data"

    def test_filter_stats_present(self, client):
        resp = client.get(f"/data/versions/{TEST_VERSION}")
        assert resp.json()["filter_stats"]["final_count"] == 5

    def test_returns_404_for_missing_version(self, client):
        resp = client.get("/data/versions/nonexistent-xyz-version-999")
        assert resp.status_code == 404


class TestCreateVersion:
    def test_missing_version_returns_422(self, client):
        resp = client.post("/data/versions/create", json={})
        assert resp.status_code == 422
