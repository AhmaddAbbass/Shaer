from fastapi.testclient import TestClient

from services.common_schemas.schemas import YehiaFeedback, PoemSpec
from services.yehia_service.app.api import get_client
from services.yehia_service.app.main import create_app


class DummyClient:
    async def feedback(self, request):
        return YehiaFeedback(ok=True, score=90, feedback="good")


def test_feedback_endpoint_returns_feedback():
    app = create_app()
    app.dependency_overrides[get_client] = lambda: DummyClient()
    client = TestClient(app)
    resp = client.post(
        "/feedback",
        json={
            "verse_text": "bayt",
            "spec": {"poem_meter": "meter", "poem_description": "desc", "num_verses": 4},
        },
    )
    assert resp.status_code == 200
    fb = resp.json()["feedback"]
    assert fb["ok"] is True
    assert fb["score"] == 90
    assert "good" in fb["feedback"]
