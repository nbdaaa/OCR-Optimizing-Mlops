import pytest

from src.training.evaluate import (
    compute_batch_cer,
    compute_cer,
    compute_loc_mae,
    extract_locs,
    strip_xml_tags,
)


class TestExtractLocs:
    def test_extracts_in_order(self):
        s = "<text><loc_10><loc_20><loc_30><loc_40>hi</text>"
        assert extract_locs(s) == [10, 20, 30, 40]

    def test_empty_when_no_locs(self):
        assert extract_locs("plain text no tags") == []


class TestComputeLocMAE:
    def test_perfect_match_zero_mae(self):
        gt = "<text><loc_10><loc_20><loc_30><loc_40>a</text>"
        mae, cov = compute_loc_mae([gt], [gt])
        assert mae == pytest.approx(0.0)
        assert cov == pytest.approx(1.0)

    def test_constant_offset(self):
        gt   = "<text><loc_10><loc_20><loc_30><loc_40>a</text>"
        pred = "<text><loc_13><loc_23><loc_33><loc_43>a</text>"  # +3 each
        mae, cov = compute_loc_mae([pred], [gt])
        assert mae == pytest.approx(3.0)
        assert cov == pytest.approx(1.0)

    def test_coverage_when_pred_has_fewer_locs(self):
        gt   = "<a><loc_0><loc_0><loc_0><loc_0></a><b><loc_5><loc_5><loc_5><loc_5></b>"
        pred = "<a><loc_0><loc_0><loc_0><loc_0></a>"  # only 4 of 8 locs
        mae, cov = compute_loc_mae([pred], [gt])
        assert mae == pytest.approx(0.0)   # the 4 compared are exact
        assert cov == pytest.approx(0.5)   # 4/8

    def test_nan_when_no_comparable_locs(self):
        mae, cov = compute_loc_mae(["no locs"], ["also none"])
        assert mae != mae   # NaN


class TestComputeCER:
    def test_perfect_match_returns_zero(self):
        assert compute_cer("hello world", "hello world") == pytest.approx(0.0)

    def test_both_empty_returns_zero(self):
        assert compute_cer("", "") == pytest.approx(0.0)

    def test_empty_prediction_full_ground_truth(self):
        # edit_distance("", "abc") = 3, len(gt) = 3 → CER = 1.0
        assert compute_cer("", "abc") == pytest.approx(1.0)

    def test_single_char_deletion(self):
        # "helo" vs "hello": 1 insertion needed, len(gt)=5 → 0.2
        assert compute_cer("helo", "hello") == pytest.approx(0.2)

    def test_single_char_substitution(self):
        # "bello" vs "hello": 1 substitution, len(gt)=5 → 0.2
        assert compute_cer("bello", "hello") == pytest.approx(0.2)

    def test_single_char_insertion(self):
        # "helloo" vs "hello": 1 deletion needed, len(gt)=5 → 0.2
        assert compute_cer("helloo", "hello") == pytest.approx(0.2)

    def test_cer_can_exceed_one(self):
        # prediction much longer than GT
        cer = compute_cer("hello world extra", "hello")
        assert cer > 1.0

    def test_cer_non_negative(self):
        cer = compute_cer("anything", "something")
        assert cer >= 0.0

    def test_vietnamese_unicode_perfect_match(self):
        assert compute_cer("Nội dung văn bản", "Nội dung văn bản") == pytest.approx(0.0)

    def test_vietnamese_unicode_partial_error(self):
        cer = compute_cer("Nội dung", "Nội dung văn bản")
        assert 0.0 < cer <= 1.0

    def test_returns_float(self):
        assert isinstance(compute_cer("abc", "abc"), float)

    def test_case_sensitive(self):
        # "Hello" vs "hello" differ in first char → CER > 0
        assert compute_cer("Hello", "hello") > 0.0


class TestStripXmlTags:
    def test_strips_doctag_wrapper(self):
        text = "<doctag><text><loc_10><loc_20><loc_30><loc_40>content</text></doctag>"
        result = strip_xml_tags(text)
        assert "<doctag>" not in result
        assert "</doctag>" not in result

    def test_strips_loc_tokens(self):
        text = "<loc_73><loc_31><loc_335><loc_43>content"
        result = strip_xml_tags(text)
        assert "<loc_" not in result

    def test_strips_element_tags(self):
        text = "<text>content</text>"
        result = strip_xml_tags(text)
        assert "<text>" not in result
        assert "</text>" not in result

    def test_preserves_text_content(self):
        text = "<doctag><text><loc_10><loc_20><loc_30><loc_40>Nội dung văn bản</text></doctag>"
        result = strip_xml_tags(text)
        assert "Nội dung văn bản" in result

    def test_empty_string_returns_empty(self):
        assert strip_xml_tags("") == ""

    def test_plain_text_unchanged(self):
        text = "plain text without any tags"
        assert strip_xml_tags(text) == text

    def test_strips_section_header_tag(self):
        text = "<section_header_level_1><loc_10><loc_20><loc_30><loc_40>Header</section_header_level_1>"
        result = strip_xml_tags(text)
        assert "<section_header_level_1>" not in result
        assert "Header" in result


class TestComputeBatchCER:
    def test_perfect_batch_returns_zero(self):
        preds = ["hello", "world", "test"]
        gts = ["hello", "world", "test"]
        assert compute_batch_cer(preds, gts) == pytest.approx(0.0)

    def test_returns_average_over_batch(self):
        # sample 1: CER=0.0, sample 2: 1 edit / 5 chars = 0.2 → avg = 0.1
        preds = ["hello", "helo"]
        gts = ["hello", "hello"]
        assert compute_batch_cer(preds, gts) == pytest.approx(0.1)

    def test_mismatched_lengths_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_batch_cer(["a", "b"], ["a"])

    def test_empty_batch_raises_value_error(self):
        with pytest.raises(ValueError):
            compute_batch_cer([], [])

    def test_returns_float(self):
        assert isinstance(compute_batch_cer(["abc"], ["abc"]), float)
