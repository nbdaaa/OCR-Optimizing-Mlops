import pytest

from src.training.evaluate import compute_cer, compute_batch_cer, strip_xml_tags


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
