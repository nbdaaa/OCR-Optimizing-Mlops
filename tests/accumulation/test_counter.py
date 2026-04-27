import pytest
from unittest.mock import patch, MagicMock


def test_get_count_returns_zero_when_file_missing():
    with patch("accumulation.counter.download_json", side_effect=Exception("not found")):
        from accumulation.counter import get_count
        assert get_count() == 0


def test_get_count_returns_value():
    with patch("accumulation.counter.download_json", return_value={"count": 42}):
        from accumulation.counter import get_count
        assert get_count() == 42


def test_increment_uploads_incremented_value():
    with patch("accumulation.counter.download_json", return_value={"count": 10}), \
         patch("accumulation.counter.upload_json") as mock_upload:
        from accumulation.counter import increment
        result = increment(1)
        assert result == 11
        mock_upload.assert_called_once_with({"count": 11}, "count.json")


def test_reset_uploads_zero():
    with patch("accumulation.counter.upload_json") as mock_upload:
        from accumulation.counter import reset
        reset()
        mock_upload.assert_called_once_with({"count": 0}, "count.json")


def test_is_ready_false_below_threshold():
    with patch("accumulation.counter.download_json", return_value={"count": 100}), \
         patch.dict("os.environ", {"ACCUMULATION_THRESHOLD": "5000"}):
        from accumulation.counter import is_ready
        assert is_ready() is False


def test_is_ready_true_at_threshold():
    with patch("accumulation.counter.download_json", return_value={"count": 5000}), \
         patch.dict("os.environ", {"ACCUMULATION_THRESHOLD": "5000"}):
        from accumulation.counter import is_ready
        assert is_ready() is True