"""
Seed MLflow Model Registry with the existing LoRA adapter from HF Hub.

Downloads adapter_config.json + adapter_model.safetensors from nbdaaa/docling-ocr,
logs them as MLflow artifacts, registers as granite-docling-adapter, and transitions
to Production stage.

Usage:
    python scripts/seed_production_model.py

Run once to bootstrap the registry before vLLM serving is set up.
Guard: exits early if a Production version already exists.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import mlflow
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

# MLflow's internal boto3 client reads AWS_* vars, not MINIO_* vars
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ["MINIO_ACCESS_KEY"])
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ["MINIO_SECRET_KEY"])

MODEL_NAME = "granite-docling-adapter"
HF_REPO = "nbdaaa/docling-ocr"
HF_REVISION = "13bc462"
BASE_MODEL = "ibm-granite/granite-docling-258M"

# Only files needed for vLLM serving
ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors"]
# Extra file for extracting training metrics
TRAINER_STATE_FILE = "trainer_state.json"


def _get_production_version(client: mlflow.MlflowClient) -> str | None:
    """Return the current Production version number, or None if none exists."""
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception:
        return None
    for v in versions:
        if v.current_stage == "Production":
            return v.version
    return None


def _extract_metrics(trainer_state_path: str) -> dict[str, float]:
    """Pull the last eval_loss from trainer_state.json if available."""
    try:
        with open(trainer_state_path) as f:
            state = json.load(f)
        history = state.get("log_history", [])
        # Last entry that has eval_loss
        for entry in reversed(history):
            if "eval_loss" in entry:
                return {"eval_loss": entry["eval_loss"]}
    except Exception:
        pass
    return {}


def _extract_params(adapter_config_path: str) -> dict[str, str]:
    """Pull LoRA hyper-params from adapter_config.json."""
    try:
        with open(adapter_config_path) as f:
            cfg = json.load(f)
        return {
            "lora_r": str(cfg.get("r", "")),
            "lora_alpha": str(cfg.get("lora_alpha", "")),
            "lora_dropout": str(cfg.get("lora_dropout", "")),
            "target_modules": ",".join(cfg.get("target_modules", [])),
            "bias": str(cfg.get("bias", "")),
        }
    except Exception:
        return {}


def seed() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()

    # Guard: skip if Production already exists
    existing = _get_production_version(client)
    if existing:
        print(f"[seed] Production version {existing} already exists — skipping.")
        sys.exit(0)

    hf_token = os.environ.get("HF_TOKEN")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"[seed] Downloading adapter files from {HF_REPO} @ {HF_REVISION} ...")
        for filename in ADAPTER_FILES:
            hf_hub_download(
                repo_id=HF_REPO,
                filename=filename,
                revision=HF_REVISION,
                local_dir=tmpdir,
                token=hf_token,
            )
            print(f"  ✓ {filename}")

        # Download trainer_state.json for metrics (best-effort)
        trainer_state_path = None
        try:
            trainer_state_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=TRAINER_STATE_FILE,
                revision=HF_REVISION,
                local_dir=tmpdir,
                token=hf_token,
            )
            print(f"  ✓ {TRAINER_STATE_FILE}")
        except Exception:
            print(f"  ! {TRAINER_STATE_FILE} not found — skipping metrics")

        lora_params = _extract_params(os.path.join(tmpdir, "adapter_config.json"))
        metrics = _extract_metrics(trainer_state_path) if trainer_state_path else {}

        mlflow.set_experiment("seed-production-model")
        with mlflow.start_run(run_name=f"seed-{HF_REVISION[:7]}") as run:
            run_id = run.info.run_id

            mlflow.log_params({
                "base_model": BASE_MODEL,
                "hf_repo": HF_REPO,
                "hf_revision": HF_REVISION,
                "source": "seed_script",
                **lora_params,
            })

            if metrics:
                mlflow.log_metrics(metrics)

            print("[seed] Uploading adapter artifacts to MLflow artifact store ...")
            mlflow.log_artifacts(tmpdir, artifact_path="adapter")
            print("  ✓ artifacts uploaded")

        # Register model
        print(f"[seed] Registering '{MODEL_NAME}' ...")
        mv = mlflow.register_model(f"runs:/{run_id}/adapter", MODEL_NAME)
        print(f"  ✓ version {mv.version} registered (stage: None)")

        # Transition to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=mv.version,
            stage="Production",
        )
        print(f"  ✓ version {mv.version} → Production")

    print(f"\n[seed] Done. '{MODEL_NAME}' version {mv.version} is now Production.")
    print(f"       MLflow UI: {tracking_uri}/#/models/{MODEL_NAME}")


if __name__ == "__main__":
    seed()
