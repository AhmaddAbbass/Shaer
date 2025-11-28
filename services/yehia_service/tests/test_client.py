from services.common_schemas.schemas import PoemSpec, YehiaFeedback
from services.yehia_service.app.client import YehiaRunpodClient
from services.yehia_service.app.config import Settings


def _dummy_client() -> YehiaRunpodClient:
    settings = Settings(
        runpod_base_url="http://example.com",
        yehia_endpoint_id="endpoint",
        runpod_api_key="key",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        request_timeout=5.0,
    )
    return YehiaRunpodClient(settings)


def test_extract_text_from_choices():
    client = _dummy_client()
    payload = {"output": [{"choices": [{"message": {"content": "hello world"}}]}]}
    assert client._extract_text(payload) == "hello world"


def test_parse_spec_text_with_defaults():
    client = _dummy_client()
    text = '{"poem_meter": "meter", "poem_description": "desc", "num_verses": 3}'
    spec = client._parse_spec_text(text, fallback_query="fallback")
    assert isinstance(spec, PoemSpec)
    assert spec.poem_meter == "meter"
    assert spec.num_verses == 3
    assert spec.poem_description == "desc"


def test_parse_feedback_text_minimal():
    client = _dummy_client()
    fb = client._parse_feedback_text('{"ok": true, "score": 90, "feedback": "good"}')
    assert isinstance(fb, YehiaFeedback)
    assert fb.ok is True
    assert fb.score == 90
    assert "good" in fb.feedback
