import os
import json
import tempfile
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
from accumulation.dedup import filter_duplicates
from common.logging import get_logger
from dotenv import load_dotenv
load_dotenv()
log          = get_logger(__name__)
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")


def _list_manifest_files() -> list[str]:
    api   = HfApi(token=HF_TOKEN)
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    return [f for f in files if f.startswith("manifests/") and f.endswith(".jsonl")]


def _load_samples_from_manifests(manifest_files: list[str]) -> list[dict]:
    samples = []
    for mf in manifest_files:
        local = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=mf,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        with open(local) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    log.info(f"Loaded {len(samples)} samples from {len(manifest_files)} manifest(s)")
    return samples


def build_dataset() -> Dataset:
    manifest_files = _list_manifest_files()
    if not manifest_files:
        raise RuntimeError("No manifest files found in HF dataset repo.")

    samples = _load_samples_from_manifests(manifest_files)
    samples = filter_duplicates(samples)

    if not samples:
        raise RuntimeError("No unique samples after deduplication.")

    dataset = Dataset.from_list(samples)
    log.info(f"Built dataset with {len(dataset)} samples, columns: {dataset.column_names}")
    return dataset