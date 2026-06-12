"""
Fast CER eval via vLLM offline — runs as a SECOND phase on the training Vast
instance AFTER train.py exits (so the GPU is free) and registers the adapter.

Why: the transformers generate loop took ~2-3h for 100 samples; vLLM does the
same in ~10-15 min. This script runs under the cached vLLM venv (its own python),
loads the just-trained adapter from MLflow, batch-generates over the benchmark,
logs `cer` to the SAME run, then self-destructs the Vast instance.

Self-contained (no heavy src imports) so it runs cleanly under the vLLM venv.

Usage (under the vLLM venv python):
    python eval_cer_vllm.py --run-id <RUN_ID>
Env: MLFLOW_TRACKING_URI, MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
     MINIO_BUCKET_DATA, BENCHMARK_VERSION, VAST_API_KEY (optional, self-destruct).
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys

BASE_MODEL = "ibm-granite/granite-docling-258M"
PROMPT = "Convert this page to docling format."


def _pip(pkgs: str) -> None:
    subprocess.run(f"{sys.executable} -m pip install -q {pkgs}", shell=True, check=True)


def _strip(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _img_url(field) -> str:
    """parquet image value (bytes / {'bytes':..}) → base64 data URI for vLLM.chat."""
    import base64
    b = field.get("bytes") if isinstance(field, dict) else field
    return "data:image/png;base64," + base64.b64encode(b).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=4500)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    print("[eval] installing light deps into venv ...", flush=True)
    _pip("mlflow-skinny boto3 pandas pyarrow pillow jiwer")

    import boto3
    import jiwer
    import mlflow
    import pandas as pd

    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = mlflow.MlflowClient()

    # 1. adapter of this run (the LoRA only, ~23MB)
    print(f"[eval] downloading adapter for run {args.run_id[:8]} ...", flush=True)
    adapter_dir = client.download_artifacts(args.run_id, "adapter", "/tmp/eval_adapter")

    # 2. benchmark parquet from MinIO
    bench = os.environ.get("BENCHMARK_VERSION", "benchmark")
    s3 = boto3.client("s3", endpoint_url=os.environ["MINIO_ENDPOINT"],
                      aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["MINIO_SECRET_KEY"])
    bucket = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    print(f"[eval] loading benchmark '{bench}' ...", flush=True)
    obj = s3.get_object(Bucket=bucket, Key=f"{bench}/dataset.parquet")
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    rows = df.to_dict("records")
    print(f"[eval] {len(rows)} benchmark samples", flush=True)

    # 3. vLLM offline batch generate (LoRA on the LLM layers — vLLM applies it fully)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.base_model, enable_lora=True,
              max_lora_rank=args.max_lora_rank, max_model_len=args.max_model_len,
              dtype="bfloat16", gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                        skip_special_tokens=True)
    msgs = [[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _img_url(r["image"])}},
                {"type": "text", "text": PROMPT}]}] for r in rows]
    print("[eval] generating (vLLM batched) ...", flush=True)
    outs = llm.chat(msgs, sp, lora_request=LoRARequest("ocr", 1, adapter_dir))

    preds = [_strip(o.outputs[0].text) for o in outs]
    gts = [_strip(r["output_text"]) for r in rows]
    cer = float(jiwer.cer(gts, preds))
    client.log_metric(args.run_id, "cer", cer)
    print(f"[eval] CER = {cer:.4f} (logged to run {args.run_id[:8]})", flush=True)

    # 4. self-destruct the Vast instance (job fully done)
    api_key = os.environ.get("VAST_API_KEY")
    if api_key and os.environ.get("TRAIN_AUTO_DESTROY", "1") == "1":
        try:
            import requests
            iid = client.get_run(args.run_id).data.tags.get("vast_instance_id")
            if iid:
                print(f"[eval] destroying Vast instance {iid}", flush=True)
                requests.delete(f"https://console.vast.ai/api/v0/instances/{iid}/",
                                params={"api_key": api_key}, timeout=20)
        except Exception as exc:  # noqa: BLE001
            print(f"[eval] self-destruct failed (destroy manually): {exc}", flush=True)


if __name__ == "__main__":
    main()
