
from fastapi.testclient import TestClient

from services.common_schemas.schemas import RagPoemRecord
from services.rag_service.app.api import get_chroma_client, get_neo4j_client
from services.rag_service.app.main import create_app


class DummyChroma:
    def ping(self) -> bool:
        return True


class DummyNeo4j:
    def __init__(self, poem: RagPoemRecord | None):
        self.poem = poem

    def get_poem(self, poem_id: str):
        return self.poem

    def ping(self) -> bool:
        return True

    def filter_poems(self, **kwargs):
        return None


def test_get_poem_happy_path():
    poem = RagPoemRecord(
        poem_id="42",
        poem_title="Title",
        poet_name="Poet",
        poem_meter="meter",
        poem_era="era",
        poem_theme="theme",
        num_verses=2,
        verses=["v1", "v2"],
        source_url=None,
    )

    app = create_app()
    app.dependency_overrides[get_chroma_client] = lambda: DummyChroma()
    app.dependency_overrides[get_neo4j_client] = lambda: DummyNeo4j(poem)

    client = TestClient(app)
    resp = client.get("/poem/42")
    assert resp.status_code == 200
    body = resp.json()
    assert body["poem_id"] == "42"
    assert body["verses"] == ["v1", "v2"]


def test_get_poem_not_found():
    app = create_app()
    app.dependency_overrides[get_chroma_client] = lambda: DummyChroma()
    app.dependency_overrides[get_neo4j_client] = lambda: DummyNeo4j(None)

    client = TestClient(app)
    resp = client.get("/poem/999")
    assert resp.status_code == 404
