import pytest


@pytest.fixture
def valid_sample():
    return {
        "image": b"fake_image_bytes",
        "output_text": (
            "<doctag>"
            "<text><loc_73><loc_31><loc_335><loc_43>Nội dung văn bản tiếng Việt</text>"
            "<section_header_level_1><loc_93><loc_44><loc_252><loc_50>Tiêu đề</section_header_level_1>"
            "</doctag>"
        ),
        "sample_id": "sample_001",
        "source": "test",
        "img_w": 595,
        "img_h": 842,
    }


@pytest.fixture
def invalid_samples():
    return [
        {"image": None, "output_text": "<doctag><loc_10><loc_20><loc_30><loc_40>text</doctag>"},
        {"image": b"img", "output_text": ""},
        {"image": b"img", "output_text": "<doctag>\n</doctag>"},
        {"image": b"img", "output_text": "<doctag>no loc tags</doctag>"},
        {"image": b"img", "output_text": "<loc_10><loc_20><loc_30><loc_40>text"},
    ]
