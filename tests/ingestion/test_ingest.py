import pytest
from unittest.mock import patch
from ingestion.ingest import ingest_file


def test_ingest_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        ingest_file("nonexistent.pdf")


def test_ingest_raises_on_unsupported_extension(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_file(str(f))


def test_ingest_single_image_success(tmp_path):
    img = tmp_path / "scan.png"
    img.write_bytes(b"fake_image_bytes")

    with patch("ingestion.ingest.call_chandra", return_value="<div>html</div>"), \
         patch("ingestion.ingest.save_sample", return_value=1) as mock_save:
        ingest_file(str(img))
        mock_save.assert_called_once_with(str(img), "<div>html</div>")


def test_ingest_pdf_splits_and_processes(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake_pdf")
    fake_pages = ["/tmp/page_0.png", "/tmp/page_1.png"]

    with patch("ingestion.ingest.pdf_to_images", return_value=fake_pages), \
         patch("ingestion.ingest.call_chandra", return_value="<div>html</div>"), \
         patch("ingestion.ingest.save_sample", return_value=2) as mock_save:
        ingest_file(str(pdf))
        assert mock_save.call_count == 2


def test_ingest_continues_on_partial_failure(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"fake_pdf")
    fake_pages = ["/tmp/page_0.png", "/tmp/page_1.png"]

    with patch("ingestion.ingest.pdf_to_images", return_value=fake_pages), \
         patch("ingestion.ingest.call_chandra",
               side_effect=["<div>html</div>", Exception("timeout")]), \
         patch("ingestion.ingest.save_sample", return_value=1) as mock_save:
        ingest_file(str(pdf))
        assert mock_save.call_count == 1
