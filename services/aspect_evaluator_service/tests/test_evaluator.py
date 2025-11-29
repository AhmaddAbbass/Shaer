import pytest

from services.common_schemas.schemas import PoemSpec, YehiaFeedback
from services.aspect_evaluator_service.app.evaluator import AspectEvaluator
from services.aspect_evaluator_service.app.openai_judge import JudgeResult
from services.aspect_evaluator_service.app.prompts import ASPECTS
from services.aspect_evaluator_service.app.schemas import AspectEvaluation


class FakeYehiaClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def feedback(self, verse_text, spec, aspect, previous_verses=None):  # type: ignore[override]
        self.calls.append(aspect)
        return YehiaFeedback(ok=True, score=None, feedback=f"{aspect}-feedback")


class FakeJudge:
    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[str] = []

    async def score(self, aspect: str, user_prompt: str) -> JudgeResult:  # type: ignore[override]
        self.calls.append(aspect)
        return JudgeResult(score_0_1=0.5, notes=f"{aspect}-notes")


@pytest.mark.asyncio
async def test_evaluate_all_invokes_all_aspects():
    evaluator = AspectEvaluator(FakeYehiaClient(), FakeJudge())
    spec = PoemSpec(poem_meter="الكامل", poem_description="وصف", num_verses=4)
    results = await evaluator.evaluate_all("بيت ما", spec, [], use_openai=True)
    assert set(results.keys()) == set(ASPECTS)
    assert all(isinstance(val, AspectEvaluation) for val in results.values())


@pytest.mark.asyncio
async def test_evaluate_all_scoring_disabled():
    yehia = FakeYehiaClient()
    judge = FakeJudge()
    judge.enabled = False
    evaluator = AspectEvaluator(yehia, judge)
    spec = PoemSpec(poem_meter="الكامل", poem_description="وصف", num_verses=4)
    results = await evaluator.evaluate_all("بيت", spec, [], use_openai=False)
    assert results[ASPECTS[0]].judge.score_0_1 is None
