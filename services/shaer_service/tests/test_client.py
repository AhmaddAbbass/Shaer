from services.shaer_service.app.client import ShaerRunpodClient
from services.shaer_service.app.schemas import BaytGenerationRequest
from services.shaer_service.app.config import Settings


def _dummy_client() -> ShaerRunpodClient:
    settings = Settings(
        runpod_base_url="http://example.com",
        shaer_endpoint_id="endpoint",
        runpod_api_key="key",
        max_new_tokens=16,
        temperature=0.7,
        top_p=0.9,
        request_timeout=5.0,
    )
    return ShaerRunpodClient(settings)


def test_extract_text_simple_choice():
    client = _dummy_client()
    resp = {"output": [{"choices": [{"message": {"content": "generated bayt"}}]}]}
    assert client._extract_text(resp) == "generated bayt"


def test_extract_text_direct_output_text():
    client = _dummy_client()
    resp = {"output": [{"output_text": "another line"}]}
    assert client._extract_text(resp) == "another line"


def test_extract_text_tokens_list():
    client = _dummy_client()
    resp = {"output": [{"choices": [{"tokens": ["tok1", "tok2"]}]}]}
    assert client._extract_text(resp) == "tok1tok2"
