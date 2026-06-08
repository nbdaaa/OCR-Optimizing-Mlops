"""
Training configuration for LoRA fine-tuning of granite-docling-258M.

All hyperparameters live here so they can be imported by train.py,
overridden in tests, and tracked as MLflow params without scattering
magic numbers across the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    max_length: int = 6000
    minio_bucket: str = "ocr-data"

    # ── Training hyperparameters ──────────────────────────────────────────────
    learning_rate: float = 5e-5
    batch_size: int = 2
    grad_accum: int = 8
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # ── Model / registry ──────────────────────────────────────────────────────
    base_model: str = "ibm-granite/granite-docling-258M"
    model_name: str = "granite-docling-adapter"

    def as_mlflow_params(self) -> dict:
        """Return a flat dict suitable for mlflow.log_params()."""
        return {
            "base_model":    self.base_model,
            "lr":            self.learning_rate,
            "batch_size":    self.batch_size,
            "grad_accum":    self.grad_accum,
            "epochs":        self.num_epochs,
            "lora_r":        self.lora_r,
            "lora_alpha":    self.lora_alpha,
            # named max_seq_length to avoid colliding with the transformers
            # MLflow callback, which logs generation_config's "max_length"
            "max_seq_length": self.max_length,
        }
