"""
vLLM serving wrapper for granite-docling-258M + LoRA adapter.

Startup flow:
  1. Query MLflow Registry for the Production version of granite-docling-adapter
  2. Download adapter artifacts from MLflow (stored in MinIO mlflow-artifacts bucket)
  3. Launch vLLM's OpenAI-compatible server with the adapter as a LoRA module

The vLLM process exposes:
  POST /v1/chat/completions  — inference (OpenAI-compatible)
  GET  /metrics              — Prometheus metrics (auto-exposed by vLLM)
  GET  /health               — health check

Usage:
  python src/serving/vllm_server.py [--port 8000] [--adapter-dir /tmp/adapter]

Request format:
  {
    "model": "ocr-adapter",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text",      "text":      "Convert this page to docling format."}
      ]}
    ],
    "max_tokens": 6000,
    "temperature": 0.0
  }
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()

# MLflow's internal boto3 client reads AWS_* vars for MinIO access
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))

MODEL_NAME = os.environ.get("MODEL_NAME", "granite-docling-adapter")
BASE_MODEL = "ibm-granite/granite-docling-258M"
LORA_MODULE_NAME = "ocr-adapter"


# ── Pure logic (unit-testable without GPU) ────────────────────────────────────

def get_production_run_id(client: mlflow.MlflowClient) -> tuple[str, str]:
    """
    Query MLflow Registry for the current Production version.
    Returns (version, run_id).
    Raises RuntimeError if no Production version exists.
    """
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.current_stage == "Production":
            return v.version, v.run_id
    raise RuntimeError(
        f"No Production version found for '{MODEL_NAME}'. "
        "Run scripts/seed_production_model.py first."
    )


def download_adapter(
    client: mlflow.MlflowClient,
    run_id: str,
    dst_dir: str,
) -> str:
    """
    Download the adapter artifact from MLflow to dst_dir.
    Returns the local path to the downloaded adapter directory.
    """
    return client.download_artifacts(run_id, "adapter", dst_dir)


def build_vllm_command(adapter_path: str, port: int) -> list[str]:
    """
    Build the argument list for launching vLLM's OpenAI-compatible server.

    Key flags:
      --enable-lora            activates LoRA support in vLLM
      --max-lora-rank 16       matches the r=16 used during training
      --lora-modules           registers the adapter under LORA_MODULE_NAME
      --max-model-len 6000     matches MAX_LENGTH from training
      --dtype bfloat16         matches torch_dtype used during training
    """
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL,
        "--enable-lora",
        "--max-lora-rank", "16",
        "--lora-modules", f"{LORA_MODULE_NAME}={adapter_path}",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", "6000",
        "--trust-remote-code",
    ]


# ── Main entry point ──────────────────────────────────────────────────────────

def serve(port: int = 8000, adapter_dir: str | None = None) -> None:
    """
    Download Production adapter from MLflow and launch vLLM server.
    Blocks until the vLLM subprocess exits.
    """
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()

    print(f"[vllm_server] Querying MLflow for Production '{MODEL_NAME}' ...")
    version, run_id = get_production_run_id(client)
    print(f"  ✓ version {version}  run_id={run_id}")

    dst = adapter_dir or os.environ.get("ADAPTER_LOCAL_DIR", "/tmp/granite-adapter")
    Path(dst).mkdir(parents=True, exist_ok=True)

    print(f"[vllm_server] Downloading adapter to {dst} ...")
    adapter_path = download_adapter(client, run_id, dst)
    print(f"  ✓ adapter at {adapter_path}")

    cmd = build_vllm_command(adapter_path, port)
    print(f"[vllm_server] Launching vLLM on port {port} ...")
    print(f"  cmd: {' '.join(cmd)}\n")

    proc = subprocess.run(cmd)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Serve granite-docling-258M + LoRA adapter via vLLM"
    )
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("VLLM_PORT", 8000)),
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--adapter-dir", type=str, default=None,
        help="Local directory to download adapter into (default: ADAPTER_LOCAL_DIR env or /tmp/granite-adapter)",
    )
    args = parser.parse_args()
    serve(port=args.port, adapter_dir=args.adapter_dir)
