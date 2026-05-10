import pytest
from datetime import datetime
from PIL import Image

from src.data.data_versioning import (
    is_valid,
    build_metadata,
    compute_image_hash,
    compute_phash,
    exact_dedup,
    phash_dedup,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_OUTPUT = "<doctag><text><loc_10><loc_20><loc_300><loc_40>text</text></doctag>"


def make_sample(image, output_text=VALID_OUTPUT, img_w=100, img_h=100):
    return {"image": image, "output_text": output_text, "img_w": img_w, "img_h": img_h}


def solid_pil(color="white", size=(64, 64)):
    return Image.new("RGB", size, color=color)


def solid_bytes(color="white", size=(64, 64)):
    import io
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# compute_image_hash
# ---------------------------------------------------------------------------

class TestComputeImageHash:
    def test_same_bytes_same_hash(self):
        data = solid_bytes("white")
        assert compute_image_hash(data) == compute_image_hash(data)

    def test_same_pil_image_same_hash(self):
        img = solid_pil("red")
        assert compute_image_hash(img) == compute_image_hash(img)

    def test_different_images_different_hash(self):
        assert compute_image_hash(solid_bytes("white")) != compute_image_hash(solid_bytes("black"))

    def test_returns_string(self):
        assert isinstance(compute_image_hash(solid_bytes("blue")), str)

    def test_handles_pil_image_input(self):
        result = compute_image_hash(solid_pil("green"))
        assert isinstance(result, str) and len(result) > 0

    def test_handles_bytes_input(self):
        result = compute_image_hash(solid_bytes("red"))
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# compute_phash
# ---------------------------------------------------------------------------

class TestComputePHash:
    def test_same_image_zero_hamming_distance(self):
        import imagehash
        img = solid_pil("white")
        h1 = compute_phash(img)
        h2 = compute_phash(img)
        assert (h1 - h2) == 0

    def test_resized_image_low_hamming_distance(self):
        # Same content, different resolution → pHash should be close
        img_small = solid_pil("white", size=(32, 32))
        img_large = solid_pil("white", size=(128, 128))
        h1 = compute_phash(img_small)
        h2 = compute_phash(img_large)
        assert (h1 - h2) <= 8

    def test_different_images_high_hamming_distance(self):
        # Completely different content → pHash should differ significantly
        import numpy as np
        rng = np.random.default_rng(42)
        arr1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        arr2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        h1 = compute_phash(Image.fromarray(arr1))
        h2 = compute_phash(Image.fromarray(arr2))
        assert (h1 - h2) > 8

    def test_accepts_pil_image(self):
        result = compute_phash(solid_pil("blue"))
        assert result is not None

    def test_accepts_bytes_image(self):
        result = compute_phash(solid_bytes("red"))
        assert result is not None

    def test_supports_hamming_subtraction(self):
        # Result must support the - operator (imagehash type)
        h = compute_phash(solid_pil("white"))
        assert isinstance(h - h, int)


# ---------------------------------------------------------------------------
# exact_dedup
# ---------------------------------------------------------------------------

class TestExactDedup:
    def test_no_duplicates_returns_all(self):
        samples = [
            make_sample(solid_bytes("red")),
            make_sample(solid_bytes("green")),
            make_sample(solid_bytes("blue")),
        ]
        result, stats = exact_dedup(samples)
        assert len(result) == 3

    def test_removes_exact_duplicate(self):
        img = solid_bytes("white")
        samples = [make_sample(img), make_sample(img)]
        result, stats = exact_dedup(samples)
        assert len(result) == 1

    def test_keeps_one_per_duplicate_group(self):
        img = solid_bytes("white")
        samples = [make_sample(img), make_sample(img), make_sample(img)]
        result, stats = exact_dedup(samples)
        assert len(result) == 1

    def test_prefers_sample_with_longer_output_text(self):
        img = solid_bytes("white")
        short_out = "<doctag><text><loc_10><loc_20><loc_30><loc_40>hi</text></doctag>"
        long_out = "<doctag><text><loc_10><loc_20><loc_300><loc_40>much longer content here</text></doctag>"
        samples = [make_sample(img, output_text=short_out), make_sample(img, output_text=long_out)]
        result, _ = exact_dedup(samples)
        assert result[0]["output_text"] == long_out

    def test_all_unique_no_removal(self):
        samples = [make_sample(solid_bytes(c)) for c in ["red", "green", "blue", "yellow"]]
        result, stats = exact_dedup(samples)
        assert len(result) == 4
        assert stats["exact_removed"] == 0

    def test_stats_exact_removed_count(self):
        img = solid_bytes("white")
        samples = [make_sample(img), make_sample(img), make_sample(solid_bytes("black"))]
        _, stats = exact_dedup(samples)
        assert stats["exact_removed"] == 1

    def test_stats_exact_groups_count(self):
        img_a = solid_bytes("white")
        img_b = solid_bytes("red")
        samples = [
            make_sample(img_a), make_sample(img_a),   # group A: 2 copies
            make_sample(img_b), make_sample(img_b),   # group B: 2 copies
        ]
        _, stats = exact_dedup(samples)
        assert stats["exact_groups"] == 2

    def test_returns_stats_dict(self):
        samples = [make_sample(solid_bytes("white"))]
        _, stats = exact_dedup(samples)
        assert isinstance(stats, dict)
        assert "exact_removed" in stats
        assert "exact_groups" in stats

    def test_empty_input_returns_empty(self):
        result, stats = exact_dedup([])
        assert result == []
        assert stats["exact_removed"] == 0


# ---------------------------------------------------------------------------
# phash_dedup
# ---------------------------------------------------------------------------

class TestPHashDedup:
    def test_no_near_duplicates_returns_all(self):
        import numpy as np
        rng = np.random.default_rng(0)
        samples = []
        for i in range(3):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            samples.append(make_sample(Image.fromarray(arr)))
        result, _ = phash_dedup(samples, threshold=8)
        assert len(result) == 3

    def test_removes_near_duplicate_within_threshold(self):
        # Two resized versions of the same image → near-duplicate
        base = solid_pil("white", size=(128, 128))
        small = base.resize((32, 32)).resize((64, 64))
        samples = [make_sample(base), make_sample(small)]
        result, stats = phash_dedup(samples, threshold=8)
        assert len(result) == 1
        assert stats["phash_removed"] == 1

    def test_keeps_sample_above_threshold(self):
        import numpy as np
        rng = np.random.default_rng(7)
        arr1 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        arr2 = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        samples = [make_sample(Image.fromarray(arr1)), make_sample(Image.fromarray(arr2))]
        result, _ = phash_dedup(samples, threshold=8)
        assert len(result) == 2

    def test_prefers_higher_resolution_when_near_dup(self):
        base = solid_pil("white", size=(128, 128))
        small = base.resize((32, 32)).resize((64, 64))
        high_res = make_sample(base, img_w=128, img_h=128)
        low_res = make_sample(small, img_w=32, img_h=32)
        result, _ = phash_dedup([low_res, high_res], threshold=8)
        assert result[0]["img_w"] == 128
        assert result[0]["img_h"] == 128

    def test_threshold_zero_only_removes_identical_phash(self):
        img = solid_pil("white", size=(64, 64))
        samples = [make_sample(img), make_sample(img)]
        result, _ = phash_dedup(samples, threshold=0)
        assert len(result) == 1

    def test_stats_phash_removed_count(self):
        base = solid_pil("white", size=(128, 128))
        small = base.resize((32, 32)).resize((64, 64))
        samples = [make_sample(base), make_sample(small), make_sample(solid_pil("black"))]
        _, stats = phash_dedup(samples, threshold=8)
        assert stats["phash_removed"] == 1

    def test_stats_threshold_recorded(self):
        samples = [make_sample(solid_pil("white"))]
        _, stats = phash_dedup(samples, threshold=10)
        assert stats["threshold"] == 10

    def test_returns_stats_dict(self):
        samples = [make_sample(solid_pil("white"))]
        _, stats = phash_dedup(samples, threshold=8)
        assert isinstance(stats, dict)
        assert "phash_removed" in stats
        assert "phash_groups" in stats
        assert "threshold" in stats

    def test_empty_input_returns_empty(self):
        result, stats = phash_dedup([], threshold=8)
        assert result == []
        assert stats["phash_removed"] == 0

    def test_default_threshold_is_8(self):
        # Calling without threshold arg should use default=8 and not raise
        samples = [make_sample(solid_pil("white"))]
        result, stats = phash_dedup(samples)
        assert stats["threshold"] == 8
