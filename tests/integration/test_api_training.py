"""
Integration tests for /training/* endpoints — calls real MLflow service.
"""
import os

import mlflow
import pytest


@pytest.fixture
def finished_run(mlflow_client):
    """Create a real MLflow run in FINISHED state with known metrics; delete after test."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment("integration-test-training")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_metric("cer", 0.1)
        mlflow.log_metric("eval_loss", 0.5)
        run_id = run.info.run_id
    yield run_id
    try:
        mlflow_client.delete_run(run_id)
    except Exception:
        pass


@pytest.fixture
def run_with_wandb_tag(mlflow_client):
    """Create a real MLflow run that has a wandb_url tag; delete after test."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment = mlflow.set_experiment("integration-test-training")
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.set_tag("wandb_url", "https://wandb.ai/ducanhcttp/chandra-ocr/runs/test-abc")
        run_id = run.info.run_id
    yield run_id
    try:
        mlflow_client.delete_run(run_id)
    except Exception:
        pass


class TestGetTrainingStatus:
    def test_returns_200_for_finished_run(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        assert resp.status_code == 200

    def test_status_is_completed_for_finished_run(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        assert resp.json()["status"] == "completed"

    def test_job_id_matches_run_id(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        assert resp.json()["job_id"] == finished_run

    def test_returns_metrics(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        metrics = resp.json()["metrics"]
        assert metrics["cer"] == pytest.approx(0.1)
        assert metrics["eval_loss"] == pytest.approx(0.5)

    def test_mlflow_url_is_present(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        assert resp.json()["mlflow_run_url"] is not None

    def test_mlflow_url_contains_run_id(self, client, finished_run):
        resp = client.get(f"/training/{finished_run}/status")
        assert finished_run in resp.json()["mlflow_run_url"]

    def test_wandb_url_returned_when_tag_present(self, client, run_with_wandb_tag):
        resp = client.get(f"/training/{run_with_wandb_tag}/status")
        assert "wandb.ai" in resp.json()["wandb_run_url"]

    def test_returns_404_for_unknown_job(self, client):
        resp = client.get("/training/nonexistent-run-id-xyz-999/status")
        assert resp.status_code == 404
