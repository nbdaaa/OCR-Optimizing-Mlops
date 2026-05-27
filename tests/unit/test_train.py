"""
Unit tests for src/training/train.py and src/training/config.py.

Tests that only need boto3/mlflow (always run):
  - TestTrainConfigDefaults, TestTrainConfigMlflowParams
  - TestLoadDatasetFromMinio, TestRegisterAdapter

Tests that need peft (skipped if not installed):
  - TestGetLoraConfig

Tests that need transformers (skipped if not installed):
  - TestBuildTrainingArgs
"""
import io
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.training.config import TrainConfig
from src.training.train import load_dataset_from_minio, register_adapter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return TrainConfig()


@pytest.fixture
def parquet_bytes():
    buf = io.BytesIO()
    pd.DataFrame([{
        "sample_id": "s1",
        "output_text": "<doctag><loc_1>test</loc_1></doctag>",
    }]).to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def mock_s3(parquet_bytes):
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(parquet_bytes)}
    return client


@pytest.fixture
def mock_mlflow_client():
    client = MagicMock()
    version_obj = MagicMock()
    version_obj.version = "3"
    client.create_model_version.return_value = version_obj
    return client


# ── TrainConfig defaults ──────────────────────────────────────────────────────

class TestTrainConfigDefaults:
    def test_learning_rate(self, cfg):
        assert cfg.learning_rate == pytest.approx(5e-5)

    def test_batch_size(self, cfg):
        assert cfg.batch_size == 2

    def test_grad_accum(self, cfg):
        assert cfg.grad_accum == 8

    def test_max_length(self, cfg):
        assert cfg.max_length == 6000

    def test_num_epochs(self, cfg):
        assert cfg.num_epochs == 10

    def test_warmup_ratio(self, cfg):
        assert cfg.warmup_ratio == pytest.approx(0.05)

    def test_weight_decay(self, cfg):
        assert cfg.weight_decay == pytest.approx(0.01)

    def test_lora_r(self, cfg):
        assert cfg.lora_r == 16

    def test_lora_alpha(self, cfg):
        assert cfg.lora_alpha == 32

    def test_lora_dropout(self, cfg):
        assert cfg.lora_dropout == pytest.approx(0.05)

    def test_lora_target_modules_attention(self, cfg):
        for m in ("q_proj", "v_proj", "k_proj", "o_proj"):
            assert m in cfg.lora_target_modules

    def test_lora_target_modules_mlp(self, cfg):
        for m in ("gate_proj", "up_proj", "down_proj"):
            assert m in cfg.lora_target_modules

    def test_base_model(self, cfg):
        assert cfg.base_model == "ibm-granite/granite-docling-258M"

    def test_model_name(self, cfg):
        assert cfg.model_name == "granite-docling-adapter"

    def test_minio_bucket(self, cfg):
        assert cfg.minio_bucket == "ocr-data"


class TestTrainConfigOverride:
    def test_can_override_learning_rate(self):
        cfg = TrainConfig(learning_rate=1e-4)
        assert cfg.learning_rate == pytest.approx(1e-4)

    def test_can_override_lora_r(self):
        cfg = TrainConfig(lora_r=8)
        assert cfg.lora_r == 8

    def test_can_override_num_epochs(self):
        cfg = TrainConfig(num_epochs=3)
        assert cfg.num_epochs == 3

    def test_lora_target_modules_are_independent_per_instance(self):
        cfg1 = TrainConfig()
        cfg2 = TrainConfig()
        cfg1.lora_target_modules.append("extra_proj")
        assert "extra_proj" not in cfg2.lora_target_modules


class TestTrainConfigMlflowParams:
    def test_returns_dict(self, cfg):
        assert isinstance(cfg.as_mlflow_params(), dict)

    def test_contains_lr(self, cfg):
        assert "lr" in cfg.as_mlflow_params()

    def test_contains_base_model(self, cfg):
        assert "base_model" in cfg.as_mlflow_params()

    def test_contains_lora_r(self, cfg):
        assert "lora_r" in cfg.as_mlflow_params()

    def test_lr_value_matches_config(self, cfg):
        assert cfg.as_mlflow_params()["lr"] == cfg.learning_rate

    def test_lora_r_value_matches_config(self, cfg):
        assert cfg.as_mlflow_params()["lora_r"] == cfg.lora_r


# ── load_dataset_from_minio ───────────────────────────────────────────────────

class TestLoadDatasetFromMinio:
    def test_calls_get_object_with_correct_bucket(self, mock_s3):
        load_dataset_from_minio("v1", s3_client=mock_s3)
        kwargs = mock_s3.get_object.call_args.kwargs
        assert kwargs["Bucket"] == "ocr-data"

    def test_calls_get_object_with_correct_key(self, mock_s3):
        load_dataset_from_minio("v1", s3_client=mock_s3)
        kwargs = mock_s3.get_object.call_args.kwargs
        assert kwargs["Key"] == "v1/dataset.parquet"

    def test_version_reflected_in_key(self, mock_s3):
        load_dataset_from_minio("v42", s3_client=mock_s3)
        kwargs = mock_s3.get_object.call_args.kwargs
        assert "v42" in kwargs["Key"]

    def test_returns_dataframe(self, mock_s3):
        result = load_dataset_from_minio("v1", s3_client=mock_s3)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_rows(self, mock_s3):
        result = load_dataset_from_minio("v1", s3_client=mock_s3)
        assert len(result) >= 1

    def test_env_bucket_override(self, mock_s3, monkeypatch):
        monkeypatch.setenv("MINIO_BUCKET_DATA", "custom-bucket")
        load_dataset_from_minio("v1", s3_client=mock_s3)
        kwargs = mock_s3.get_object.call_args.kwargs
        assert kwargs["Bucket"] == "custom-bucket"

    def test_propagates_s3_error(self, mock_s3):
        from botocore.exceptions import ClientError
        mock_s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
        )
        with pytest.raises(ClientError):
            load_dataset_from_minio("missing-version", s3_client=mock_s3)

    def test_config_bucket_used_when_no_env(self, mock_s3, monkeypatch):
        monkeypatch.delenv("MINIO_BUCKET_DATA", raising=False)
        cfg = TrainConfig(minio_bucket="cfg-bucket")
        load_dataset_from_minio("v1", s3_client=mock_s3, config=cfg)
        kwargs = mock_s3.get_object.call_args.kwargs
        assert kwargs["Bucket"] == "cfg-bucket"


# ── register_adapter ──────────────────────────────────────────────────────────

class TestRegisterAdapter:
    def test_creates_model_version(self, mock_mlflow_client):
        register_adapter("run-123", mlflow_client=mock_mlflow_client)
        mock_mlflow_client.create_model_version.assert_called_once()

    def test_model_name_is_default(self, mock_mlflow_client):
        register_adapter("run-123", mlflow_client=mock_mlflow_client)
        kwargs = mock_mlflow_client.create_model_version.call_args.kwargs
        assert kwargs["name"] == TrainConfig().model_name

    def test_source_contains_run_id(self, mock_mlflow_client):
        register_adapter("run-abc", mlflow_client=mock_mlflow_client)
        kwargs = mock_mlflow_client.create_model_version.call_args.kwargs
        assert "run-abc" in kwargs["source"]

    def test_source_points_to_adapter_artifact(self, mock_mlflow_client):
        register_adapter("run-abc", mlflow_client=mock_mlflow_client)
        kwargs = mock_mlflow_client.create_model_version.call_args.kwargs
        assert "adapter" in kwargs["source"]

    def test_transitions_to_staging(self, mock_mlflow_client):
        register_adapter("run-123", mlflow_client=mock_mlflow_client)
        kwargs = mock_mlflow_client.transition_model_version_stage.call_args.kwargs
        assert kwargs["stage"] == "Staging"

    def test_returns_version_number(self, mock_mlflow_client):
        result = register_adapter("run-123", mlflow_client=mock_mlflow_client)
        assert result == "3"

    def test_custom_model_name(self, mock_mlflow_client):
        register_adapter("run-123", model_name="custom-adapter", mlflow_client=mock_mlflow_client)
        kwargs = mock_mlflow_client.create_model_version.call_args.kwargs
        assert kwargs["name"] == "custom-adapter"


# ── get_lora_config ───────────────────────────────────────────────────────────

class TestGetLoraConfig:
    @pytest.fixture(autouse=True)
    def require_peft(self):
        pytest.importorskip("peft")

    def test_r_matches_config(self):
        from src.training.train import get_lora_config
        cfg = TrainConfig(lora_r=8)
        assert get_lora_config(cfg).r == 8

    def test_alpha_matches_config(self):
        from src.training.train import get_lora_config
        cfg = TrainConfig(lora_alpha=64)
        assert get_lora_config(cfg).lora_alpha == 64

    def test_dropout_matches_config(self):
        from src.training.train import get_lora_config
        cfg = TrainConfig(lora_dropout=0.1)
        assert get_lora_config(cfg).lora_dropout == pytest.approx(0.1)

    def test_target_modules_match_config(self):
        from src.training.train import get_lora_config
        cfg = TrainConfig()
        config = get_lora_config(cfg)
        for module in cfg.lora_target_modules:
            assert module in config.target_modules

    def test_task_type_is_causal_lm(self):
        from peft import TaskType
        from src.training.train import get_lora_config
        assert get_lora_config(TrainConfig()).task_type == TaskType.CAUSAL_LM

    def test_bias_is_none(self):
        from src.training.train import get_lora_config
        assert get_lora_config(TrainConfig()).bias == "none"


# ── build_training_args ───────────────────────────────────────────────────────

class TestBuildTrainingArgs:
    @pytest.fixture(autouse=True)
    def require_transformers(self):
        pytest.importorskip("transformers")

    def test_learning_rate_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(learning_rate=1e-4)
        assert build_training_args(str(tmp_path), cfg).learning_rate == pytest.approx(1e-4)

    def test_batch_size_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(batch_size=4)
        assert build_training_args(str(tmp_path), cfg).per_device_train_batch_size == 4

    def test_grad_accum_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(grad_accum=4)
        assert build_training_args(str(tmp_path), cfg).gradient_accumulation_steps == 4

    def test_num_epochs_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(num_epochs=3)
        assert build_training_args(str(tmp_path), cfg).num_train_epochs == 3

    def test_warmup_ratio_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(warmup_ratio=0.1)
        assert build_training_args(str(tmp_path), cfg).warmup_ratio == pytest.approx(0.1)

    def test_weight_decay_from_config(self, tmp_path):
        from src.training.train import build_training_args
        cfg = TrainConfig(weight_decay=0.05)
        assert build_training_args(str(tmp_path), cfg).weight_decay == pytest.approx(0.05)

    def test_output_dir(self, tmp_path):
        from src.training.train import build_training_args
        assert build_training_args(str(tmp_path), TrainConfig()).output_dir == str(tmp_path)

    def test_smoke_test_sets_max_steps_to_one(self, tmp_path):
        from src.training.train import build_training_args
        assert build_training_args(str(tmp_path), TrainConfig(), smoke_test=True).max_steps == 1

    def test_normal_mode_max_steps_is_minus_one(self, tmp_path):
        from src.training.train import build_training_args
        assert build_training_args(str(tmp_path), TrainConfig(), smoke_test=False).max_steps == -1
