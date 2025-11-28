from fastapi.testclient import TestClient

from services.yehia_service.app.api import get_client
from services.yehia_service.app.main import create_app


class DummyClient:
    async def chat(self, request):
        return "hello"


def test_chat_endpoint_returns_text():
    app = create_app()
    app.dependency_overrides[get_client] = lambda: DummyClient()
    client = TestClient(app)
    resp = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["text"] == "hello"
