"""
Integration tests for /models/versions/* endpoints — calls real MLflow Model Registry.
"""
import pytest


class TestListModelVersions:
    def test_returns_200(self, client):
        resp = client.get("/models/versions")
        assert resp.status_code == 200

    def test_response_has_versions_key(self, client):
        resp = client.get("/models/versions")
        assert "versions" in resp.json()

    def test_versions_is_a_list(self, client):
        resp = client.get("/models/versions")
        assert isinstance(resp.json()["versions"], list)

    def test_each_version_has_required_fields(self, client):
        resp = client.get("/models/versions")
        for version in resp.json()["versions"]:
            assert "version" in version
            assert "stage" in version
            assert "run_id" in version


class TestGetModelVersion:
    def test_returns_404_for_nonexistent_version(self, client):
        resp = client.get("/models/versions/99999")
        assert resp.status_code == 404
