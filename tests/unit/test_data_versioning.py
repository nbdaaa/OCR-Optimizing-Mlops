import pytest
from datetime import datetime

from src.data.data_versioning import is_valid, build_metadata


class TestIsValid:
    def test_rejects_none_image(self):
        sample = {
            "image": None,
            "output_text": "<doctag><text><loc_73><loc_31><loc_335><loc_43>text</text></doctag>",
        }
        assert not is_valid(sample)

    def test_rejects_empty_output_text(self):
        sample = {"image": b"fake_image", "output_text": ""}
        assert not is_valid(sample)

    def test_rejects_whitespace_only_output(self):
        sample = {"image": b"fake_image", "output_text": "   "}
        assert not is_valid(sample)

    def test_rejects_doctag_only_with_newline(self):
        sample = {"image": b"fake_image", "output_text": "<doctag>\n</doctag>"}
        assert not is_valid(sample)

    def test_rejects_doctag_only_empty(self):
        sample = {"image": b"fake_image", "output_text": "<doctag></doctag>"}
        assert not is_valid(sample)

    def test_rejects_missing_loc_tags(self):
        sample = {
            "image": b"fake_image",
            "output_text": "<doctag>some text without loc tags</doctag>",
        }
        assert not is_valid(sample)

    def test_rejects_missing_opening_doctag(self):
        sample = {
            "image": b"fake_image",
            "output_text": "<loc_10><loc_20><loc_30><loc_40>text</doctag>",
        }
        assert not is_valid(sample)

    def test_rejects_missing_closing_doctag(self):
        sample = {
            "image": b"fake_image",
            "output_text": "<doctag><loc_10><loc_20><loc_30><loc_40>text",
        }
        assert not is_valid(sample)

    def test_rejects_missing_output_text_key(self):
        sample = {"image": b"fake_image"}
        assert not is_valid(sample)

    def test_accepts_minimal_valid_sample(self):
        sample = {
            "image": b"fake_image_bytes",
            "output_text": "<doctag><text><loc_73><loc_31><loc_335><loc_43>Nội dung</text></doctag>",
        }
        assert is_valid(sample)

    def test_accepts_pil_image(self):
        from PIL import Image

        img = Image.new("RGB", (100, 100))
        sample = {
            "image": img,
            "output_text": "<doctag><text><loc_10><loc_20><loc_300><loc_40>text</text></doctag>",
        }
        assert is_valid(sample)

    def test_accepts_multiple_elements(self):
        sample = {
            "image": b"img",
            "output_text": (
                "<doctag>"
                "<text><loc_10><loc_20><loc_300><loc_40>First line</text>"
                "<section_header_level_1><loc_50><loc_60><loc_200><loc_70>Header</section_header_level_1>"
                "</doctag>"
            ),
        }
        assert is_valid(sample)

    def test_accepts_loc_values_at_extremes(self):
        sample = {
            "image": b"img",
            "output_text": "<doctag><text><loc_0><loc_0><loc_499><loc_499>edge</text></doctag>",
        }
        assert is_valid(sample)

    def test_accepts_sample_from_conftest(self, valid_sample):
        assert is_valid(valid_sample)

    def test_rejects_all_invalid_samples_from_conftest(self, invalid_samples):
        for sample in invalid_samples:
            assert not is_valid(sample), f"Expected invalid but passed: {sample}"


class TestBuildMetadata:
    def test_has_all_required_fields(self):
        meta = build_metadata(
            version="v1",
            count=1000,
            hf_repo="nbdaaa/all-ocr-data",
            filter_stats={"total": 1200, "rejected": 200},
            split="train",
        )
        required = {"version", "count", "created_at", "hf_repo", "filter_stats", "split"}
        assert required.issubset(set(meta.keys()))

    def test_version_stored_correctly(self):
        meta = build_metadata(version="v3", count=5000, hf_repo="repo", filter_stats={}, split="train")
        assert meta["version"] == "v3"

    def test_count_stored_correctly(self):
        meta = build_metadata(version="v1", count=42000, hf_repo="repo", filter_stats={}, split="train")
        assert meta["count"] == 42000

    def test_hf_repo_stored_correctly(self):
        meta = build_metadata(
            version="v1", count=100, hf_repo="nbdaaa/all-ocr-data", filter_stats={}, split="train"
        )
        assert meta["hf_repo"] == "nbdaaa/all-ocr-data"

    def test_created_at_is_parseable_iso_datetime(self):
        meta = build_metadata(version="v1", count=100, hf_repo="repo", filter_stats={}, split="train")
        datetime.fromisoformat(meta["created_at"])  # raises if invalid

    def test_filter_stats_stored_correctly(self):
        stats = {"total": 1200, "rejected": 200, "rejection_rate": 0.167}
        meta = build_metadata(version="v1", count=1000, hf_repo="repo", filter_stats=stats, split="train")
        assert meta["filter_stats"] == stats

    def test_split_stored_correctly(self):
        meta = build_metadata(version="v1", count=100, hf_repo="repo", filter_stats={}, split="val")
        assert meta["split"] == "val"
