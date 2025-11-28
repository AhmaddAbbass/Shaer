import pytest

from services.ashaar_meter_service.app import ashaar as ashaar_module
from services.ashaar_meter_service.app.ashaar import AshaarMeterService
from services.ashaar_meter_service.app.config import Settings


class DummyConfig:
    def __init__(self, ashaar_weight: float):
        self.ashaar_weight = ashaar_weight


def fake_reward(text: str, config: DummyConfig):
    return {"reward": 0.42 * config.ashaar_weight, "ashaar_score": 0.84}


@pytest.fixture
def patched_assets(monkeypatch):
    monkeypatch.setattr(ashaar_module, "AshaarRewardConfig", DummyConfig, raising=False)
    monkeypatch.setattr(ashaar_module, "ashaar_reward", fake_reward, raising=False)
    yield
    monkeypatch.setattr(ashaar_module, "AshaarRewardConfig", None, raising=False)
    monkeypatch.setattr(ashaar_module, "ashaar_reward", None, raising=False)


def test_evaluate_bayt_returns_scores(patched_assets):
    settings = Settings(ashaar_weight=0.5)
    service = AshaarMeterService(settings)
    assert service.assets_loaded is True

    result = service.evaluate_bayt("بيت اختباري")
    assert result.ashaar_score == 0.84
    assert result.reward == pytest.approx(0.21)
    assert "درجة التشابه" in result.notes


def test_default_when_assets_missing(monkeypatch):
    monkeypatch.setattr(ashaar_module, "AshaarRewardConfig", None, raising=False)
    monkeypatch.setattr(ashaar_module, "ashaar_reward", None, raising=False)
    settings = Settings()
    service = AshaarMeterService(settings)
    assert service.assets_loaded is False

    result = service.evaluate_bayt("أي بيت")
    assert result.ashaar_score == 0.0
    assert result.reward == 0.0
    assert "لا يمكن حساب" in result.notes
