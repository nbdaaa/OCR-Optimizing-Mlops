"""
LoRA fine-tune ibm-granite/granite-docling-258M on Vietnamese OCR data.

Usage:
    python src/training/train.py --data-version v1
    python src/training/train.py --data-version v1 --smoke-test
"""
from __future__ import annotations

import argparse
import io
import os

import boto3
import mlflow
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from src.training.config import TrainConfig

# Load .env so MLFLOW_TRACKING_URI / MINIO_* / HF_TOKEN / WANDB_* are available
load_dotenv()

# Reduce CUDA fragmentation OOM on long sequences
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_USER_PROMPT = "Convert this page to docling format."


def _decode_image(img) -> Image.Image:
    """Coerce a parquet image value (bytes / HF dict / PIL) to an RGB PIL.Image."""
    if isinstance(img, dict):
        img = img.get("bytes")
    if isinstance(img, (bytes, bytearray)):
        img = Image.open(io.BytesIO(img))
    return img.convert("RGB")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset_from_minio(
    version: str,
    s3_client=None,
    config: TrainConfig | None = None,
) -> pd.DataFrame:
    """
    Load dataset.parquet for {version} as a DataFrame.

    Caches the parquet locally under DATA_CACHE_DIR (default /tmp/ocr-data-cache)
    so re-runs skip the MinIO download. The download streams to a .part file and
    is atomically renamed, so an interrupted download never leaves a corrupt cache.
    """
    cfg = config or TrainConfig()
    cache_dir = os.environ.get("DATA_CACHE_DIR", "/tmp/ocr-data-cache")
    local_path = os.path.join(cache_dir, version, "dataset.parquet")

    if os.path.exists(local_path):
        print(f"[train] using cached dataset {local_path}", flush=True)
        return pd.read_parquet(local_path)

    if s3_client is None:
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ["MINIO_ENDPOINT"],
            aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
            aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        )
    bucket = os.environ.get("MINIO_BUCKET_DATA", cfg.minio_bucket)
    resp = s3_client.get_object(Bucket=bucket, Key=f"{version}/dataset.parquet")

    total = int(resp.get("ContentLength", 0))
    body = resp["Body"]
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = local_path + ".part"
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="  downloading parquet"
    ) as pbar:
        for chunk in iter(lambda: body.read(1024 * 1024), b""):
            f.write(chunk)
            pbar.update(len(chunk))
    os.replace(tmp_path, local_path)  # atomic — only complete downloads cached
    return pd.read_parquet(local_path)


# ── LoRA config ────────────────────────────────────────────────────────────────

def get_lora_config(config: TrainConfig | None = None):
    """Return LoRA config built from TrainConfig."""
    from peft import LoraConfig, TaskType
    cfg = config or TrainConfig()
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )


# ── Training args ─────────────────────────────────────────────────────────────

def build_training_args(
    output_dir: str,
    config: TrainConfig | None = None,
    smoke_test: bool = False,
):
    """Build HF TrainingArguments from TrainConfig. smoke_test=True → max_steps=1."""
    from transformers import TrainingArguments
    cfg = config or TrainConfig()
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=True,
        logging_steps=10,
        # Per-epoch eval_loss on the held-out split. prediction_loss_only=True is
        # essential: it discards logits after computing loss, avoiding the ~8GB
        # full-sequence (text + thousands of image tokens) cross-entropy alloc
        # that OOMed earlier. CER (generate-based) still runs once at the end.
        eval_strategy="epoch" if cfg.val_split > 0 else "no",
        per_device_eval_batch_size=cfg.batch_size,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        report_to=["mlflow", "wandb"],
        max_steps=1 if smoke_test else -1,
        # Keep raw columns (image, output_text, ...) — the custom collator needs
        # them; Trainer would otherwise strip non-forward-signature columns.
        remove_unused_columns=False,
        # Trade compute for memory — long (6000-token) sequences with image
        # tokens blow up activation memory on a 24GB card without this.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


# ── Data collator ─────────────────────────────────────────────────────────────

class DataCollatorForOCR:
    """
    Formats each sample into the granite-docling chat template, tokenizes,
    then applies loss masking so only assistant (doctag) tokens contribute.
    """

    def __init__(self, processor, config: TrainConfig | None = None) -> None:
        self.processor = processor
        self.max_length = (config or TrainConfig()).max_length
        boundary_text = "<|start_of_role|>assistant<|end_of_role|>"
        self._boundary_tokens: list[int] = processor.tokenizer(
            boundary_text, add_special_tokens=False
        ).input_ids

    def __call__(self, samples: list[dict]) -> dict:
        import torch
        from src.training.collator import apply_label_mask, find_boundary_idx

        texts, images = [], []
        for sample in samples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _USER_PROMPT},
                    ],
                },
                {"role": "assistant", "content": sample["output_text"]},
            ]
            texts.append(
                self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            )
            images.append([_decode_image(sample["image"])])

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = []
        for ids, mask in zip(
            batch["input_ids"].tolist(), batch["attention_mask"].tolist()
        ):
            boundary_end = find_boundary_idx(ids, self._boundary_tokens)
            labels.append(apply_label_mask(ids, mask, boundary_end))

        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# ── CER evaluation (generate-based) ───────────────────────────────────────────

def evaluate_cer(
    model,
    processor,
    df: pd.DataFrame,
    config: TrainConfig | None = None,
    n_samples: int | None = None,
    max_new_tokens: int | None = None,
    batch_size: int | None = None,
) -> float:
    """
    Compute mean CER over `df` by batched generation, comparing predictions to
    ground-truth output_text (XML tags stripped on both sides).

    Batched generation (cer_batch_size rows per generate call) is much faster
    than one-at-a-time. Uses the KV cache (no full-sequence logits → no OOM).

    n_samples=None evaluates the WHOLE df (full benchmark); otherwise df.head(n).
    """
    import torch
    from src.training.evaluate import compute_batch_cer, strip_xml_tags

    cfg = config or TrainConfig()
    new_tokens = max_new_tokens or cfg.cer_max_new_tokens
    bs = batch_size or cfg.cer_batch_size
    eval_df = df if n_samples is None else df.head(n_samples)

    # generate() needs cache on, checkpointing off, and LEFT padding for batching
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    processor.tokenizer.padding_side = "left"

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"},
                                       {"type": "text", "text": _USER_PROMPT}]}],
        tokenize=False, add_generation_prompt=True,
    )

    rows = eval_df.to_dict("records")
    preds, gts = [], []
    for start in tqdm(range(0, len(rows), bs), desc="  CER eval", leave=False):
        batch = rows[start:start + bs]
        inputs = processor(
            text=[prompt] * len(batch),
            images=[[_decode_image(r["image"])] for r in batch],
            return_tensors="pt",
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
        # left padding → generated tokens start uniformly after the padded prompt
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for r, pred in zip(batch, decoded):
            preds.append(strip_xml_tags(pred))
            gts.append(strip_xml_tags(r["output_text"]))

    return compute_batch_cer(preds, gts)


# ── MLflow registration ───────────────────────────────────────────────────────

def register_adapter(
    run_id: str,
    model_name: str | None = None,
    mlflow_client=None,
    config: TrainConfig | None = None,
) -> str:
    """Register adapter in MLflow Model Registry and transition to Staging. Returns version string."""
    cfg = config or TrainConfig()
    name = model_name or cfg.model_name
    if mlflow_client is None:
        mlflow_client = mlflow.MlflowClient()
    version = mlflow_client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/adapter",
        run_id=run_id,
    )
    mlflow_client.transition_model_version_stage(
        name=name,
        version=version.version,
        stage="Staging",
    )
    return version.version


def _self_destruct(run_id: str) -> None:
    """
    Destroy the Vast.ai instance this job is running on, so a finished job
    doesn't leave an idle GPU burning money. Best-effort: needs VAST_API_KEY
    (forwarded into the instance .env) and the instance id (read from the run's
    vast_instance_id tag, set by the control plane after provisioning).
    Disable with TRAIN_AUTO_DESTROY=0 (e.g. when debugging on the box).
    """
    if os.environ.get("TRAIN_AUTO_DESTROY", "1") != "1":
        print("[train] auto-destroy disabled (TRAIN_AUTO_DESTROY=0)", flush=True)
        return
    api_key = os.environ.get("VAST_API_KEY")
    if not api_key:
        print("[train] no VAST_API_KEY — skipping self-destruct", flush=True)
        return
    try:
        import requests
        iid = mlflow.MlflowClient().get_run(run_id).data.tags.get("vast_instance_id")
        if not iid:
            print("[train] no vast_instance_id tag — skipping self-destruct", flush=True)
            return
        print(f"[train] job done → destroying Vast instance {iid}", flush=True)
        requests.delete(
            f"https://console.vast.ai/api/v0/instances/{iid}/",
            params={"api_key": api_key}, timeout=20,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the job over cleanup
        print(f"[train] self-destruct failed (destroy manually): {exc}", flush=True)


# ── Continual warm-start ──────────────────────────────────────────────────────

def resolve_warmstart_version(mlflow_client, model_name: str, init_version: str | None):
    """
    Pick the registered model version to warm-start (continual training) from.

    init_version given  → that exact version (raises if missing).
    init_version None   → the latest version (highest version number).
    No versions at all  → None (caller starts a fresh LoRA from base).
    """
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    if init_version:
        match = next((v for v in versions if str(v.version) == str(init_version)), None)
        if match is None:
            raise RuntimeError(f"Model '{model_name}' has no version {init_version}")
        return match
    return max(versions, key=lambda v: int(v.version))


# ── Main training pipeline ────────────────────────────────────────────────────

def train(
    data_version: str,
    output_dir: str = "/tmp/ocr-adapter",
    smoke_test: bool = False,
    config: TrainConfig | None = None,
    run_id: str | None = None,
    resume_from_checkpoint: bool = False,
    init_adapter_version: str | None = None,
) -> str:
    """
    Full LoRA fine-tune pipeline (continual):
      1. Pull dataset.parquet from MinIO
      2. Load base + warm-start LoRA from the prior model version (latest, or
         init_adapter_version); fresh LoRA only if no version exists yet
      3. Train with HF Trainer → log to MLflow + W&B
      4. mlflow.log_artifacts(output_dir, "adapter") → MinIO mlflow-artifacts
      5. register_adapter → Model Registry Staging
    Returns MLflow run_id.
    """
    import torch
    import wandb
    from datasets import Dataset as HFDataset
    from peft import PeftModel, get_peft_model
    from transformers import AutoProcessor, Trainer
    # transformers >= 4.49 renamed AutoModelForVision2Seq → AutoModelForImageTextToText
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    cfg = config or TrainConfig()

    # Map MinIO creds so MLflow's internal boto3 can reach artifact store
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("ocr-training")

    hf_token = os.environ.get("HF_TOKEN")

    print(f"[train] downloading dataset '{data_version}' from MinIO ...", flush=True)
    df = load_dataset_from_minio(data_version, config=cfg)
    hf_dataset = HFDataset.from_pandas(df)
    print(f"[train] loaded {len(hf_dataset):,} samples", flush=True)

    # Hold out a fraction for per-epoch eval_loss (overfit signal). Smoke test
    # and val_split<=0 keep the whole set for training (no eval).
    eval_dataset = None
    if not smoke_test and cfg.val_split > 0 and len(hf_dataset) > 1:
        split = hf_dataset.train_test_split(test_size=cfg.val_split, seed=42)
        hf_dataset, eval_dataset = split["train"], split["test"]
        print(f"[train] split → {len(hf_dataset):,} train / {len(eval_dataset):,} val "
              f"({cfg.val_split:.0%})", flush=True)

    print(f"[train] loading processor ({cfg.base_model}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(cfg.base_model, token=hf_token)
    print(f"[train] loading base model (downloads from HF on first run) ...", flush=True)
    model = AutoVLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    for param in model.parameters():
        param.requires_grad = False

    # Continual: warm-start from a prior adapter version (latest by default),
    # else fresh LoRA from base for the very first training.
    prev = resolve_warmstart_version(mlflow.MlflowClient(), cfg.model_name, init_adapter_version)
    if prev is not None:
        adapter_dst = os.path.join(output_dir, "_warmstart")
        os.makedirs(adapter_dst, exist_ok=True)
        local = mlflow.MlflowClient().download_artifacts(prev.run_id, "adapter", adapter_dst)
        print(f"[train] continual warm-start from model version {prev.version} "
              f"(run {prev.run_id[:8]})", flush=True)
        model = PeftModel.from_pretrained(model, local, is_trainable=True)
    else:
        print(f"[train] no prior adapter — fresh LoRA from base", flush=True)
        model = get_peft_model(model, get_lora_config(cfg))

    # Required for gradient checkpointing to flow grads through a frozen base
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print(f"[train] init W&B ...", flush=True)
    # entity=None → wandb uses the API key's default entity (avoids
    # "entity not found" when a hardcoded name doesn't match the account)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "chandra-ocr"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=f"train-{data_version}{'_smoke' if smoke_test else ''}",
    )

    with mlflow.start_run(run_id=run_id, run_name=f"train-{data_version}") as run:
        run_id = run.info.run_id
        mlflow.log_params({**cfg.as_mlflow_params(), "data_version": data_version, "smoke_test": smoke_test})
        mlflow.set_tag("wandb_url", wandb.run.get_url())

        # Durable checkpoints: upload each saved checkpoint's contents to MLflow
        # artifacts under "checkpoint/" (overwrites → only latest kept) + tag the
        # step, so a new instance can resume after the current one dies.
        from transformers import TrainerCallback

        class _CkptUploader(TrainerCallback):
            def on_save(self, args, state, control, **kw):
                ck = os.path.join(output_dir, f"checkpoint-{state.global_step}")
                if os.path.isdir(ck):
                    c = mlflow.MlflowClient()
                    c.log_artifacts(run_id, ck, artifact_path="checkpoint")
                    c.set_tag(run_id, "last_checkpoint_step", str(state.global_step))

        print(f"[train] starting training (smoke_test={smoke_test}, "
              f"resume={resume_from_checkpoint}) ...", flush=True)
        trainer = Trainer(
            model=model,
            args=build_training_args(output_dir, cfg, smoke_test=smoke_test),
            train_dataset=hf_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForOCR(processor, cfg),
            callbacks=[_CkptUploader()],
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        mlflow.set_tag("training_complete", "true")
        print(f"[train] training done, saving adapter ...", flush=True)

        if trainer.state.log_history:
            last = {k: v for d in trainer.state.log_history for k, v in d.items()}
            for key in ("eval_loss", "train_loss"):
                if key in last:
                    mlflow.log_metric(key, last[key])

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="adapter")

        # Generate-based CER on the fixed held-out benchmark → metric the CI
        # gate reads. Evaluating on a separate benchmark (not train data) avoids
        # leakage and makes cross-version regression comparison fair.
        # Smoke test stays on train df (1 sample) to keep it fast.
        if smoke_test:
            bench_df = df
        else:
            bench_version = os.environ.get("BENCHMARK_VERSION", "benchmark")
            try:
                bench_df = load_dataset_from_minio(bench_version, config=cfg)
                print(f"[train] CER benchmark: '{bench_version}' ({len(bench_df)} samples)", flush=True)
            except Exception as exc:
                bench_df = df
                print(f"[train] WARNING benchmark '{bench_version}' not found ({exc}); "
                      f"CER on TRAIN data (biased)", flush=True)

        print(f"[train] computing CER ...", flush=True)
        try:
            cer = evaluate_cer(
                model, processor, bench_df, cfg,
                n_samples=1 if smoke_test else None,   # None → full benchmark
                max_new_tokens=64 if smoke_test else None,
            )
            mlflow.log_metric("cer", cer)
            print(f"[train] CER = {cer:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 — don't lose the run if CER fails
            print(f"[train] CER eval failed (still registering): {exc}", flush=True)

    wandb.finish()
    register_adapter(run_id, config=cfg)
    if not smoke_test:
        _self_destruct(run_id)
    return run_id


# ── Recovery (stage-aware resume) ─────────────────────────────────────────────

def _has_artifact(client, run_id: str, path: str) -> bool:
    try:
        return len(client.list_artifacts(run_id, path)) > 0
    except Exception:
        return False


def resolve_recovery_action(client, cfg: TrainConfig, run_id: str) -> str:
    """
    Decide where a dead run should resume from, by inspecting MLflow state:
      DONE          — a model version is already registered for this run
      REGISTER_ONLY — cer logged but not registered yet → just register
      FINALIZE      — training finished (adapter artifact / training_complete tag)
                      but no cer → load adapter, compute CER, register (no retrain)
      RESUME_TRAIN  — died mid-training → resume from the durable checkpoint
    """
    versions = client.search_model_versions(f"name='{cfg.model_name}'")
    if any(v.run_id == run_id for v in versions):
        return "DONE"
    run = client.get_run(run_id)
    if "cer" in run.data.metrics:
        return "REGISTER_ONLY"
    if run.data.tags.get("training_complete") == "true" or _has_artifact(client, run_id, "adapter"):
        return "FINALIZE"
    return "RESUME_TRAIN"


def recover(run_id: str, output_dir: str = "/tmp/ocr-adapter", config: TrainConfig | None = None) -> str:
    """
    Resume a dead run from the right stage (see resolve_recovery_action).
    Reuses durable artifacts (adapter / checkpoint) saved under the SAME run_id.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoVLM

    cfg = config or TrainConfig()
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = mlflow.MlflowClient()

    action = resolve_recovery_action(client, cfg, run_id)
    print(f"[recover] run {run_id[:8]} → action = {action}", flush=True)

    if action == "DONE":
        print("[recover] already registered — nothing to do.", flush=True)
        _self_destruct(run_id)
        return run_id

    if action == "REGISTER_ONLY":
        register_adapter(run_id, config=cfg)
        client.set_terminated(run_id, status="FINISHED")  # leave RUNNING → watchdog stops
        print("[recover] registered to Staging.", flush=True)
        _self_destruct(run_id)
        return run_id

    if action == "FINALIZE":
        hf_token = os.environ.get("HF_TOKEN")
        processor = AutoProcessor.from_pretrained(cfg.base_model, token=hf_token)
        model = AutoVLM.from_pretrained(
            cfg.base_model, torch_dtype=torch.bfloat16, device_map="auto", token=hf_token,
        )
        adapter_dir = os.path.join(output_dir, "_recover_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        local = client.download_artifacts(run_id, "adapter", adapter_dir)
        model = PeftModel.from_pretrained(model, local)
        model.eval()
        print("[recover] FINALIZE: loaded adapter, computing CER on benchmark …", flush=True)
        try:
            bench_df = load_dataset_from_minio(os.environ.get("BENCHMARK_VERSION", "benchmark"), config=cfg)
            cer = evaluate_cer(model, processor, bench_df, cfg)
            client.log_metric(run_id, "cer", cer)
            print(f"[recover] CER = {cer:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[recover] CER failed (still registering): {exc}", flush=True)
        register_adapter(run_id, config=cfg)
        client.set_terminated(run_id, status="FINISHED")  # leave RUNNING → watchdog stops
        print("[recover] registered to Staging.", flush=True)
        _self_destruct(run_id)
        return run_id

    # RESUME_TRAIN — download the durable checkpoint and continue training, using
    # the SAME run_id and the original hyperparameters (read from run params).
    run = client.get_run(run_id)
    p = run.data.params
    data_version = p.get("data_version")
    if p.get("epochs"):     cfg.num_epochs = int(p["epochs"])
    if p.get("batch_size"): cfg.batch_size = int(p["batch_size"])
    if p.get("grad_accum"): cfg.grad_accum = int(p["grad_accum"])
    if p.get("lr"):         cfg.learning_rate = float(p["lr"])

    ckpt_dir = os.path.join(output_dir, "_resume_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    local = client.download_artifacts(run_id, "checkpoint", ckpt_dir)
    print(f"[recover] RESUME_TRAIN: resuming from checkpoint {local}", flush=True)
    return train(
        data_version, output_dir=output_dir, config=cfg,
        run_id=run_id, resume_from_checkpoint=local,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", default=None)
    parser.add_argument("--output-dir", default="/tmp/ocr-adapter")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", default=None, help="Resume an existing MLflow run")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint in --output-dir",
    )
    parser.add_argument(
        "--recover", action="store_true",
        help="Stage-aware recovery of a dead run (needs --run-id): resume train / "
             "finalize CER / register, decided from MLflow state.",
    )
    parser.add_argument(
        "--init-adapter-version", default=None,
        help="Continual: warm-start from this model version (default: latest)",
    )
    # Optional hyperparameter overrides (default → TrainConfig values)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum = args.grad_accum
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate

    if args.recover:
        if not args.run_id:
            parser.error("--recover requires --run-id")
        print(recover(args.run_id, output_dir=args.output_dir, config=cfg))
    else:
        if not args.data_version:
            parser.error("--data-version is required (unless --recover)")
        print(train(
            args.data_version,
            args.output_dir,
            args.smoke_test,
            config=cfg,
            run_id=args.run_id,
            resume_from_checkpoint=args.resume,
            init_adapter_version=args.init_adapter_version,
        ))
