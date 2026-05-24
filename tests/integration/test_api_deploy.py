"""
Integration tests for /deploy/* endpoints.
The deploy-status endpoint reads a local state file, so tests patch
the module-level _STATE_FILE path rather than calling a remote service.
"""
import json

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestGetDeployStatus:
    def test_returns_empty_when_no_state_file(self, tmp_path):
        with patch("src.api.routers.deploy._STATE_FILE", str(tmp_path / "nonexistent.json")):
            resp = client.get("/deploy/status")
        assert resp.status_code == 200
        assert resp.json()["instances"] == []

    def test_returns_instances_from_state_file(self, tmp_path):
        state = {
            "instances": [
                {"id": "inst-1", "address": "10.0.0.1:8000"},
                {"id": "inst-2", "address": "10.0.0.2:8000"},
            ],
            "last_scale_time": 0.0,
        }
        state_file = tmp_path / "scaler_state.json"
        state_file.write_text(json.dumps(state))
        with patch("src.api.routers.deploy._STATE_FILE", str(state_file)):
            resp = client.get("/deploy/status")
        assert resp.status_code == 200
        instances = resp.json()["instances"]
        assert len(instances) == 2
        assert instances[0]["instance_id"] == "inst-1"
        assert instances[1]["address"] == "10.0.0.2:8000"

    def test_all_instances_have_running_status(self, tmp_path):
        state = {
            "instances": [{"id": "inst-1", "address": "10.0.0.1:8000"}],
            "last_scale_time": 0.0,
        }
        state_file = tmp_path / "scaler_state.json"
        state_file.write_text(json.dumps(state))
        with patch("src.api.routers.deploy._STATE_FILE", str(state_file)):
            resp = client.get("/deploy/status")
        assert resp.json()["instances"][0]["status"] == "running"
