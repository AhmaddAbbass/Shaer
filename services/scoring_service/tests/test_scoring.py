import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from services.aspect_evaluator_service.app.schemas import (
    AspectEvaluation,
    AspectJudgeResult,
    EvaluateBaytResponse,
)
from services.common_schemas.schemas import BaytMeterEval, PoemSpec, YehiaFeedback
from services.scoring_service.app.aggregator import ScoringAggregator
from services.scoring_service.app.api import get_aggregator
from services.scoring_service.app.main import create_app
from services.scoring_service.app.schemas import ScoreBaytRequest, ScoreResponse
from services.scoring_service.app.scoring import ScoreCalculator, ScoreComponents
from services.scoring_service.app.config import Settings


class FakeMeterClient:
    async def eval_bayt(self, verse_text: str, target_meter: str) -> BaytMeterEval:
        return BaytMeterEval(target_meter=target_meter, meter_score=80, on_meter=True, notes="ok")


def _aspect_eval(score: float) -> AspectEvaluation:
    return AspectEvaluation(
        yehia_feedback=YehiaFeedback(ok=True, score=None, feedback="جيد"),
        judge=AspectJudgeResult(score_0_1=score, notes="ملاحظة"),
    )


class FakeAspectClient:
    def __init__(self, score: float = 0.8) -> None:
        self.score = score

    async def evaluate_bayt(self, verse_text, spec, previous_verses, use_openai_scoring):  # type: ignore[override]
        eval_obj = _aspect_eval(self.score)
        return EvaluateBaytResponse(
            meaning=eval_obj,
            cohesion=eval_obj,
            fluency=eval_obj,
            poeticness=eval_obj,
        )


@pytest.mark.asyncio
async def test_scoring_aggregator_combines_scores():
    settings = Settings(pass_threshold=0.7)
    aggregator = ScoringAggregator(FakeMeterClient(), FakeAspectClient(), ScoreCalculator(settings), settings)
    request = ScoreBaytRequest(
        verse_text="يا قلب",
        spec=PoemSpec(poem_meter="الكامل", poem_description="وصف", num_verses=4),
    )
    response = await aggregator.score(request)
    assert response.meter_eval.meter_score == 80
    assert response.aspects.meaning.judge.score_0_1 == pytest.approx(0.8)
    assert 0.7 < response.final_score <= 1.0
    assert response.passed is True


def test_score_calculator_equation():
    settings = Settings(pass_threshold=0.7)
    calculator = ScoreCalculator(settings)
    components = ScoreComponents(
        meter_score=0.8,
        meaning=0.6,
        fluency=0.7,
        poeticness=0.5,
        cohesion=0.4,
    )
    result = calculator.compute(components)
    expected = 0.5 * 0.8 + 0.15 * 0.6 + 0.15 * 0.7 + 0.1 * 0.5 + 0.1 * 0.4
    assert abs(result.final_score - expected) < 1e-6
    assert result.passed is (expected >= settings.pass_threshold)


def test_score_endpoint_returns_composite_response():
    app = create_app()
    fake_response = ScoreResponse(
        meter_eval=BaytMeterEval(target_meter="الكامل", meter_score=90, on_meter=True, notes=""),
        aspects=EvaluateBaytResponse(
            meaning=_aspect_eval(0.8),
            cohesion=_aspect_eval(0.7),
            fluency=_aspect_eval(0.9),
            poeticness=_aspect_eval(0.6),
        ),
        final_score=0.82,
        passed=True,
        breakdown={"meter_score": 0.45},
    )

    class FakeAggregator:
        async def score(self, request):  # type: ignore[override]
            return fake_response

    app.dependency_overrides[get_aggregator] = lambda: FakeAggregator()
    client = TestClient(app)
    payload = {
        "verse_text": "يا قلب",
        "spec": {"poem_meter": "الكامل", "poem_description": "وصف", "num_verses": 4},
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["final_score"] == 0.82
    assert "meter_eval" in data
    assert "aspects" in data
