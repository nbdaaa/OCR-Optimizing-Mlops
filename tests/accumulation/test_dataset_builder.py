import pytest
from unittest.mock import patch, MagicMock


def test_build_dataset_raises_when_no_manifests():
    with patch("accumulation.dataset_builder._list_manifest_files", return_value=[]):
        from accumulation.dataset_builder import build_dataset
        with pytest.raises(RuntimeError, match="No manifest files"):
            build_dataset()


def test_build_dataset_raises_when_all_duplicates():
    with patch("accumulation.dataset_builder._list_manifest_files", return_value=["manifests/a.jsonl"]), \
         patch("accumulation.dataset_builder._load_samples_from_manifests", return_value=[{"image_path": "a.png", "output_text": "hello"}]), \
         patch("accumulation.dataset_builder.filter_duplicates", return_value=[]):
        from accumulation.dataset_builder import build_dataset
        with pytest.raises(RuntimeError, match="No unique samples"):
            build_dataset()


def test_build_dataset_returns_dataset():
    samples = [
        {"image_path": "a.png", "output_text": "hello"},
        {"image_path": "b.png", "output_text": "world"},
    ]
    with patch("accumulation.dataset_builder._list_manifest_files", return_value=["manifests/a.jsonl"]), \
         patch("accumulation.dataset_builder._load_samples_from_manifests", return_value=samples), \
         patch("accumulation.dataset_builder.filter_duplicates", return_value=samples):
        from accumulation.dataset_builder import build_dataset
        ds = build_dataset()
        assert len(ds) == 2
        assert "image_path" in ds.column_names