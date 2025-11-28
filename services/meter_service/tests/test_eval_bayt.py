from fastapi.testclient import TestClient

from services.common_schemas.schemas import BaytMeterEval
from services.meter_service.app.api import get_scansion_service
from services.meter_service.app.main import create_app


class DummyScansion:
    def __init__(self, *, score: int, on_meter: bool, notes: str):
        self.assets_loaded = True
        self._result = BaytMeterEval(
            target_meter="bahar",
            meter_score=score,
            on_meter=on_meter,
            notes=notes,
        )

    def evaluate_bayt(self, verse_text: str, target_meter: str) -> BaytMeterEval:
        return self._result


def test_health_reports_assets_loaded():
    app = create_app()
    app.dependency_overrides[get_scansion_service] = lambda: DummyScansion(
        score=90, on_meter=True, notes="ok"
    )
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["assets_loaded"] is True


def test_eval_bayt_pass():
    app = create_app()
    app.dependency_overrides[get_scansion_service] = lambda: DummyScansion(
        score=90, on_meter=True, notes="تم الاجتياز"
    )
    client = TestClient(app)
    resp = client.post(
        "/eval-bayt",
        json={"verse_text": "bayt", "target_meter": "bahar"},
    )
    assert resp.status_code == 200
    body = resp.json()["result"]
    assert body["meter_score"] == 90
    assert body["on_meter"] is True
    assert "تم" in body["notes"]


def test_eval_bayt_fail():
    app = create_app()
    app.dependency_overrides[get_scansion_service] = lambda: DummyScansion(
        score=40, on_meter=False, notes="لم يجتز"
    )
    client = TestClient(app)
    resp = client.post(
        "/eval-bayt",
        json={"verse_text": "bad", "target_meter": "unknown"},
    )
    assert resp.status_code == 200
    body = resp.json()["result"]
    assert body["meter_score"] == 40
    assert body["on_meter"] is False
    assert "لم" in body["notes"]
