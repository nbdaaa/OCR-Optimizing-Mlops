import pytest
from unittest.mock import patch

@pytest.fixture
def mock_hf_api():
    with patch("common.storage._api") as mock:
        yield mock

@pytest.fixture
def mock_vastai():
    with patch("requests.put") as mock_put, \
         patch("requests.get") as mock_get, \
         patch("requests.delete") as mock_del:
        yield {"put": mock_put, "get": mock_get, "delete": mock_del}

@pytest.fixture
def mock_prometheus():
    with patch("common.metrics.push_to_gateway") as mock:
        yield mock
