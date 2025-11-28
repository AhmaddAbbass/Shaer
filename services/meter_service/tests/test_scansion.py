import types

import pytest

from services.common_schemas.schemas import BaytMeterEval
from services.meter_service.app.scansion import ScansionService
from services.meter_service.app.config import Settings


class DummyModel:
    def __init__(self, probs):
        self.probs = probs

    def predict(self, X, verbose=0):
        return [self.probs]


class DummyEncoder:
    def __init__(self, classes_):
        self.classes_ = classes_


@pytest.fixture
def fake_scansion(monkeypatch):
    fake_tf = types.SimpleNamespace(convert_to_tensor=lambda arr, dtype=None: arr, int32=None)
    monkeypatch.setattr("services.meter_service.app.scansion.tf", fake_tf, raising=False)

    settings = Settings(
        meter_model_path="dummy.h5",
        meter_label_encoder_path="dummy.joblib",
        meter_vocab_config_path="dummy.json",
        meter_score_threshold=80,
    )

    svc = ScansionService(settings)
    svc.assets_loaded = True
    svc.stoi = {"a": 1, " ": 0}
    svc.max_len = 4
    svc.label_encoder = DummyEncoder(classes_=["bahar1", "bahar2"])
    svc.model = DummyModel([0.9, 0.1])

    yield svc


def test_evaluate_bayt_pass(fake_scansion):
    res: BaytMeterEval = fake_scansion.evaluate_bayt("aa", "bahar1")
    assert res.on_meter is True
    assert res.meter_score == 90
    assert "bahar1" in res.notes or "bahar" in res.notes


def test_evaluate_bayt_fail(fake_scansion):
    fake_scansion.model = DummyModel([0.1, 0.9])
    res: BaytMeterEval = fake_scansion.evaluate_bayt("aa", "bahar1")
    assert res.on_meter is False
    assert res.meter_score == 10
    assert "bahar" in res.notes


def test_unknown_target_meter(fake_scansion):
    res = fake_scansion.evaluate_bayt("aa", "unknown-bahar")
    assert res.on_meter is False
    assert res.meter_score == 0
    assert "أقرب بحر متوقع" in res.notes or "bahar" in res.notes
