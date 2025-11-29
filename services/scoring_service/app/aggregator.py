from __future__ import annotations

from services.aspect_evaluator_service.app.schemas import EvaluateBaytResponse
from services.common_schemas.schemas import BaytMeterEval

from .clients import AspectEvaluatorClient, MeterServiceClient
from .config import Settings
from .schemas import ScoreBaytRequest, ScoreResponse
from .scoring import ScoreCalculator, ScoreComponents


class ScoringAggregator:
    def __init__(
        self,
        meter_client: MeterServiceClient,
        aspect_client: AspectEvaluatorClient,
        calculator: ScoreCalculator,
        settings: Settings,
    ) -> None:
        self.meter_client = meter_client
        self.aspect_client = aspect_client
        self.calculator = calculator
        self.settings = settings

    async def score(self, request: ScoreBaytRequest) -> ScoreResponse:
        meter_eval = await self.meter_client.eval_bayt(request.verse_text, request.spec.poem_meter)
        aspects = await self.aspect_client.evaluate_bayt(
            verse_text=request.verse_text,
            spec=request.spec,
            previous_verses=request.previous_verses,
            use_openai_scoring=request.use_openai_scoring,
        )
        components = self._build_components(meter_eval, aspects)
        calc = self.calculator.compute(components)
        return ScoreResponse(
            meter_eval=meter_eval,
            aspects=aspects,
            final_score=calc.final_score,
            passed=calc.passed,
            breakdown=calc.breakdown,
        )

    def _build_components(
        self,
        meter_eval: BaytMeterEval,
        aspects: EvaluateBaytResponse,
    ) -> ScoreComponents:
        def _score(value: float | None) -> float:
            if value is None:
                return 0.0
            return max(0.0, min(1.0, value))

        return ScoreComponents(
            meter_score=max(0.0, min(1.0, (meter_eval.meter_score or 0) / 100.0)),
            meaning=_score(aspects.meaning.judge.score_0_1),
            cohesion=_score(aspects.cohesion.judge.score_0_1),
            fluency=_score(aspects.fluency.judge.score_0_1),
            poeticness=_score(aspects.poeticness.judge.score_0_1),
        )
