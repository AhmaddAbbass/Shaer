from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from services.common_schemas.schemas import PoemSpec

from .clients import YehiaServiceClient
from .openai_judge import AspectJudge, JudgeResult
from .prompts import ASPECTS, ASPECT_TITLES
from .schemas import AspectEvaluation, AspectJudgeResult


@dataclass
class AspectContext:
    aspect: str
    verse_text: str
    spec: PoemSpec
    previous_verses: list[str]


class AspectEvaluator:
    def __init__(self, yehia_client: YehiaServiceClient, judge: AspectJudge):
        self.yehia_client = yehia_client
        self.judge = judge

    async def evaluate_all(
        self,
        verse_text: str,
        spec: PoemSpec,
        previous_verses: list[str],
        use_openai: bool,
    ) -> Dict[str, AspectEvaluation]:
        results: Dict[str, AspectEvaluation] = {}
        for aspect in ASPECTS:
            ctx = AspectContext(aspect, verse_text, spec, previous_verses)
            results[aspect] = await self._evaluate_single(ctx, use_openai)
        return results

    async def evaluate_one(
        self,
        aspect: str,
        verse_text: str,
        spec: PoemSpec,
        previous_verses: list[str],
        use_openai: bool,
    ) -> AspectEvaluation:
        ctx = AspectContext(aspect, verse_text, spec, previous_verses)
        return await self._evaluate_single(ctx, use_openai)

    async def _evaluate_single(self, ctx: AspectContext, use_openai: bool) -> AspectEvaluation:
        feedback = await self.yehia_client.feedback(ctx.verse_text, ctx.spec, ctx.aspect, ctx.previous_verses)
        judge_result = await self._score_with_openai(ctx, feedback.feedback, use_openai)
        return AspectEvaluation(
            yehia_feedback=feedback,
            judge=AspectJudgeResult(score_0_1=judge_result.score_0_1, notes=judge_result.notes),
        )

    async def _score_with_openai(
        self,
        ctx: AspectContext,
        feedback_text: str,
        use_openai: bool,
    ) -> JudgeResult:
        if not use_openai or not self.judge.enabled:
            return JudgeResult(score_0_1=None, notes="تم تعطيل التقييم العددي لهذا الطلب.")
        previous_block = "\n".join(ctx.previous_verses) if ctx.previous_verses else "لا توجد أبيات سابقة."
        spec = ctx.spec
        user_prompt = (
            f"البحر: {spec.poem_meter}\n"
            f"الوصف: {spec.poem_description}\n"
            f"العصر/الأسلوب: {spec.poem_era or 'غير محدد'} | {spec.poet_name or 'أسلوب عام'}\n"
            f"الأبيات السابقة:\n{previous_block}\n\n"
            f"البيت المراد تقييمه:\n{ctx.verse_text}\n\n"
            f"ملاحظات الناقد يحيى حول {ASPECT_TITLES.get(ctx.aspect, ctx.aspect)}:\n{feedback_text}\n\n"
            "أعد درجة بين 0 و1 مع ملاحظة موجزة."
        )
        return await self.judge.score(ctx.aspect, user_prompt)
