from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from services.common_schemas.schemas import BaytMeterEval, YehiaFeedback
from services.shaer_service.app.schemas import BaytGenerationRequest

from .config import Settings

METER_KEYWORDS = ("وزن", "عروض", "بحر", "كسر")
SEMANTIC_KEYWORDS = ("معنى", "فكرة", "صورة", "انسجام", "احساس", "عاطفة")
REPETITION_KEYWORDS = ("تكرار", "كرر")


@dataclass
class EnhancementPlan:
    request: BaytGenerationRequest
    applied_changes: List[str]


class EnhancementPlanner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_plan(
        self,
        request: BaytGenerationRequest,
        meter_eval: Optional[BaytMeterEval],
        feedback: Optional[YehiaFeedback],
        feedback_summary: Sequence[str],
    ) -> EnhancementPlan:
        updated_request = request.model_copy(deep=True)
        applied: List[str] = []
        guidance = list(updated_request.extra_guidance or [])

        if self._needs_meter_focus(meter_eval, feedback_summary, feedback):
            applied.extend(
                self._append_guidance(
                    guidance,
                    "ركّز على التزام الوزن بدقة وتجنّب الكسور.",
                    label="added_meter_guidance",
                )
            )

        if self._needs_semantic_focus(feedback, feedback_summary):
            if self.settings.enable_description_tightening:
                tightened = self._tighten_description(updated_request.poem_description, feedback, feedback_summary)
                if tightened != updated_request.poem_description:
                    updated_request = updated_request.model_copy(update={"poem_description": tightened})
                    applied.append("tightened_description")
            applied.extend(
                self._append_guidance(
                    guidance,
                    "حافظ على المعنى والجو الوارد في الوصف بعناية.",
                    label="added_semantic_guidance",
                )
            )

        if self._needs_repetition_warning(feedback_summary, feedback):
            applied.extend(
                self._append_guidance(
                    guidance,
                    "تجنّب تكرار الأبيات أو العبارات السابقة.",
                    label="added_repetition_guidance",
                )
            )

        if len(guidance) > self.settings.max_extra_guidance:
            guidance = guidance[: self.settings.max_extra_guidance]

        updated_request = updated_request.model_copy(update={"extra_guidance": guidance})

        return EnhancementPlan(request=updated_request, applied_changes=_dedupe(applied))

    def _needs_meter_focus(
        self,
        meter_eval: Optional[BaytMeterEval],
        feedback_summary: Sequence[str],
        feedback: Optional[YehiaFeedback],
    ) -> bool:
        if meter_eval and meter_eval.meter_score < self.settings.meter_focus_threshold:
            return True
        return _summary_contains(feedback_summary, METER_KEYWORDS) or (
            feedback is not None and _text_contains(feedback.feedback, METER_KEYWORDS)
        )

    def _needs_semantic_focus(
        self,
        feedback: Optional[YehiaFeedback],
        feedback_summary: Sequence[str],
    ) -> bool:
        if feedback and not feedback.ok:
            return True
        return _summary_contains(feedback_summary, SEMANTIC_KEYWORDS) or (
            feedback is not None and _text_contains(feedback.feedback, SEMANTIC_KEYWORDS)
        )

    def _needs_repetition_warning(
        self,
        feedback_summary: Sequence[str],
        feedback: Optional[YehiaFeedback],
    ) -> bool:
        return _summary_contains(feedback_summary, REPETITION_KEYWORDS) or (
            feedback is not None and _text_contains(feedback.feedback, REPETITION_KEYWORDS)
        )

    def _append_guidance(self, guidance: List[str], line: str, label: str) -> List[str]:
        clean = line.strip()
        if not clean or clean in guidance or len(guidance) >= self.settings.max_extra_guidance:
            return []
        guidance.append(clean)
        return [label]

    def _tighten_description(
        self,
        description: str,
        feedback: Optional[YehiaFeedback],
        feedback_summary: Sequence[str],
    ) -> str:
        addition_source = _first_summary_line(feedback_summary)
        if not addition_source and feedback and feedback.feedback:
            addition_source = feedback.feedback.split("\n")[0]
        if not addition_source:
            return description
        addition = addition_source.strip().replace("\n", " ")
        addition = addition[:120]
        if not addition:
            return description
        if addition in description:
            return description
        if description.strip():
            return f"{description.strip()} التركيز على {addition}"
        return addition


def normalize_candidate_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _summary_contains(summary: Sequence[str], keywords: Tuple[str, ...]) -> bool:
    return any(_text_contains(line, keywords) for line in summary)


def _text_contains(text: Optional[str], keywords: Tuple[str, ...]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for val in values:
        if val and val not in seen:
            seen.add(val)
            ordered.append(val)
    return ordered


def _first_summary_line(summary: Sequence[str]) -> str:
    for line in summary:
        if line.strip():
            return line.strip()
    return ""
