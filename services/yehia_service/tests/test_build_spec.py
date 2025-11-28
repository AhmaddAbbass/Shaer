from fastapi.testclient import TestClient

from services.common_schemas.schemas import PoemSpec
from services.yehia_service.app.api import get_client
from services.yehia_service.app.main import create_app


class DummyClient:
    async def build_spec(self, request):
        return PoemSpec(
            poem_meter="meter",
            poem_description="desc",
            num_verses=4,
            poem_theme=None,
            poem_era=None,
            poet_name=None,
            poem_title=None,
        )


def test_build_spec_endpoint_returns_spec():
    app = create_app()
    app.dependency_overrides[get_client] = lambda: DummyClient()
    client = TestClient(app)
    resp = client.post(
        "/build-spec",
        json={"user_query": "قصيدة عن الشوق", "rag_hits": []},
    )
    assert resp.status_code == 200
    data = resp.json()["spec"]
    assert data["poem_meter"] == "meter"
    assert data["poem_description"] == "desc"
    assert data["num_verses"] == 4
