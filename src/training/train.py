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
from tqdm import tqdm

from src.training.config import TrainConfig

# Load .env so MLFLOW_TRACKING_URI / MINIO_* / HF_TOKEN / WANDB_* are available
load_dotenv()

_USER_PROMPT = "Convert this page to docling format."


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset_from_minio(
    version: str,
    s3_client=None,
    config: TrainConfig | None = None,
) -> pd.DataFrame:
    """Download dataset.parquet from MinIO {bucket}/{version}/ and return as DataFrame."""
    cfg = config or TrainConfig()
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
    buf = io.BytesIO()
    with tqdm(
        total=total, unit="B", unit_scale=True, desc="  downloading parquet"
    ) as pbar:
        for chunk in iter(lambda: body.read(1024 * 1024), b""):
            buf.write(chunk)
            pbar.update(len(chunk))
    buf.seek(0)
    return pd.read_parquet(buf)


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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["mlflow", "wandb"],
        max_steps=1 if smoke_test else -1,
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
            images.append([sample["image"]])

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


# ── Main training pipeline ────────────────────────────────────────────────────

def train(
    data_version: str,
    output_dir: str = "/tmp/ocr-adapter",
    smoke_test: bool = False,
    config: TrainConfig | None = None,
    run_id: str | None = None,
) -> str:
    """
    Full LoRA fine-tune pipeline:
      1. Pull dataset.parquet from MinIO
      2. Load ibm-granite/granite-docling-258M + freeze base + apply LoRA
      3. Train with HF Trainer → log to MLflow + W&B
      4. mlflow.log_artifacts(output_dir, "adapter") → MinIO mlflow-artifacts
      5. register_adapter → Model Registry Staging
    Returns MLflow run_id.
    """
    import torch
    import wandb
    from datasets import Dataset as HFDataset
    from peft import get_peft_model
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

    print(f"[train] loading processor ({cfg.base_model}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(cfg.base_model, token=hf_token)
    print(f"[train] loading base model (downloads from HF on first run) ...", flush=True)
    model = AutoVLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    print(f"[train] freezing base + applying LoRA ...", flush=True)
    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, get_lora_config(cfg))
    model.print_trainable_parameters()

    print(f"[train] init W&B ...", flush=True)
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "chandra-ocr"),
        entity=os.environ.get("WANDB_ENTITY", "ducanhcttp"),
        name=f"train-{data_version}{'_smoke' if smoke_test else ''}",
    )

    with mlflow.start_run(run_id=run_id, run_name=f"train-{data_version}") as run:
        run_id = run.info.run_id
        mlflow.log_params({**cfg.as_mlflow_params(), "data_version": data_version, "smoke_test": smoke_test})
        mlflow.set_tag("wandb_url", wandb.run.get_url())

        print(f"[train] starting training (smoke_test={smoke_test}) ...", flush=True)
        trainer = Trainer(
            model=model,
            args=build_training_args(output_dir, cfg, smoke_test=smoke_test),
            train_dataset=hf_dataset,
            eval_dataset=hf_dataset,
            data_collator=DataCollatorForOCR(processor, cfg),
        )
        trainer.train()
        print(f"[train] training done, saving adapter ...", flush=True)

        if trainer.state.log_history:
            last = {k: v for d in trainer.state.log_history for k, v in d.items()}
            for key in ("eval_loss", "train_loss"):
                if key in last:
                    mlflow.log_metric(key, last[key])

        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="adapter")

    wandb.finish()
    register_adapter(run_id, config=cfg)
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--output-dir", default="/tmp/ocr-adapter")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", default=None, help="Resume an existing MLflow run")
    args = parser.parse_args()
    print(train(args.data_version, args.output_dir, args.smoke_test, run_id=args.run_id))
