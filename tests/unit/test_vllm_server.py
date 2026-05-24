"""
Unit tests for src/serving/vllm_server.py

Tests cover the three pure-logic functions that don't need a GPU:
  - get_production_run_id()
  - download_adapter()
  - build_vllm_command()
"""
import sys
import pytest
from unittest.mock import MagicMock, patch

from src.serving.vllm_server import (
    get_production_run_id,
    download_adapter,
    build_vllm_command,
    BASE_MODEL,
    LORA_MODULE_NAME,
)


def _make_version(version: str, stage: str, run_id: str = "run-abc"):
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.run_id = run_id
    return mv


# ── get_production_run_id ─────────────────────────────────────────────────────

class TestGetProductionRunId:
    def test_returns_version_and_run_id_for_production(self):
        client = MagicMock()
        client.search_model_versions.return_value = [
            _make_version("1", "Production", "run-prod-123"),
        ]
        version, run_id = get_production_run_id(client)
        assert version == "1"
        assert run_id == "run-prod-123"

    def test_skips_non_production_versions(self):
        client = MagicMock()
        client.search_model_versions.return_value = [
            _make_version("1", "Archived", "run-old"),
            _make_version("2", "Staging", "run-staging"),
            _make_version("3", "Production", "run-prod"),
        ]
        version, run_id = get_production_run_id(client)
        assert version == "3"
        assert run_id == "run-prod"

    def test_raises_when_no_production_version(self):
        client = MagicMock()
        client.search_model_versions.return_value = [
            _make_version("1", "Staging", "run-staging"),
        ]
        with pytest.raises(RuntimeError, match="No Production version"):
            get_production_run_id(client)

    def test_raises_when_registry_empty(self):
        client = MagicMock()
        client.search_model_versions.return_value = []
        with pytest.raises(RuntimeError):
            get_production_run_id(client)


# ── download_adapter ──────────────────────────────────────────────────────────

class TestDownloadAdapter:
    def test_calls_download_artifacts_with_correct_args(self, tmp_path):
        client = MagicMock()
        client.download_artifacts.return_value = str(tmp_path / "adapter")
        download_adapter(client, "run-abc", str(tmp_path))
        client.download_artifacts.assert_called_once_with("run-abc", "adapter", str(tmp_path))

    def test_returns_local_path(self, tmp_path):
        expected = str(tmp_path / "adapter")
        client = MagicMock()
        client.download_artifacts.return_value = expected
        result = download_adapter(client, "run-abc", str(tmp_path))
        assert result == expected


# ── build_vllm_command ────────────────────────────────────────────────────────

class TestBuildVllmCommand:
    def test_contains_base_model(self):
        cmd = build_vllm_command("/tmp/adapter", port=8000)
        assert BASE_MODEL in cmd

    def test_contains_enable_lora_flag(self):
        cmd = build_vllm_command("/tmp/adapter", port=8000)
        assert "--enable-lora" in cmd

    def test_contains_lora_module_with_path(self):
        cmd = build_vllm_command("/tmp/adapter", port=8000)
        lora_idx = cmd.index("--lora-modules")
        lora_arg = cmd[lora_idx + 1]
        assert LORA_MODULE_NAME in lora_arg
        assert "/tmp/adapter" in lora_arg

    def test_contains_correct_port(self):
        cmd = build_vllm_command("/tmp/adapter", port=9999)
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == "9999"

    def test_port_is_string(self):
        cmd = build_vllm_command("/tmp/adapter", port=8000)
        port_idx = cmd.index("--port")
        assert isinstance(cmd[port_idx + 1], str)

    def test_returns_list_of_strings(self):
        cmd = build_vllm_command("/tmp/adapter", port=8000)
        assert isinstance(cmd, list)
        assert all(isinstance(arg, str) for arg in cmd)

    def test_custom_port_reflected(self):
        cmd8 = build_vllm_command("/tmp/adapter", port=8000)
        cmd9 = build_vllm_command("/tmp/adapter", port=9000)
        idx8 = cmd8.index("--port")
        idx9 = cmd9.index("--port")
        assert cmd8[idx8 + 1] == "8000"
        assert cmd9[idx9 + 1] == "9000"
