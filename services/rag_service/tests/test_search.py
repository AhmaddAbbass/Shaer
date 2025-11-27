
from fastapi.testclient import TestClient

from services.common_schemas.schemas import RagSearchHit, RagSearchResponse
from services.rag_service.app.api import get_chroma_client, get_neo4j_client
from services.rag_service.app.main import create_app


class DummyChroma:
    def __init__(self):
        self.seen = []

    def search(self, query_text: str, top_k: int = 5) -> RagSearchResponse:
        self.seen.append((query_text, top_k))
        return RagSearchResponse(
            hits=[
                RagSearchHit(
                    poem_id="1",
                    poem_title="",
                    poet_name="Dummy Poet",
                    poem_description="desc",
                    poem_meter=None,
                    poem_era=None,
                    poem_theme=None,
                    has_bad_description=False,
                )
            ]
        )

    def ping(self) -> bool:
        return True


class DummyNeo4j:
    def ping(self) -> bool:
        return True


def test_search_endpoint_returns_hits():
    app = create_app()

    chroma = DummyChroma()
    neo4j = DummyNeo4j()

    app.dependency_overrides[get_chroma_client] = lambda: chroma
    app.dependency_overrides[get_neo4j_client] = lambda: neo4j

    client = TestClient(app)
    resp = client.get("/search", params={"text": "حب", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "hits" in data
    assert len(data["hits"]) == 1
    assert chroma.seen == [("حب", 2)]


def test_health_uses_pings():
    app = create_app()
    chroma = DummyChroma()
    neo4j = DummyNeo4j()
    app.dependency_overrides[get_chroma_client] = lambda: chroma
    app.dependency_overrides[get_neo4j_client] = lambda: neo4j

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["chroma"] == "ok"
