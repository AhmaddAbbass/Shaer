import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from services.enhancer_service.app.config import Settings
from services.enhancer_service.app.policy import EnhancementPlanner
from services.common_schemas.schemas import BaytMeterEval, YehiaFeedback
from services.shaer_service.app.schemas import BaytGenerationRequest


def _base_request() -> BaytGenerationRequest:
    return BaytGenerationRequest(
        poem_meter="الكامل",
        poem_description="قصيدة عن أمل المسافر",
        num_verses=4,
        sequence_number=2,
        previous_verses=["يا ليلُ طوّل فإن الشوق يسألني"],
    )


def test_planner_adds_meter_guidance_when_score_low():
    planner = EnhancementPlanner(Settings(max_extra_guidance=2))
    meter_eval = BaytMeterEval(target_meter="الكامل", meter_score=40, on_meter=False, notes="الوزن مكسور")
    plan = planner.build_plan(
        _base_request(),
        meter_eval,
        feedback=None,
        feedback_summary=[],
    )
    assert any("الوزن" in tip for tip in plan.request.extra_guidance)
    assert "added_meter_guidance" in plan.applied_changes


def test_planner_tightens_description_and_adds_semantic_guidance():
    planner = EnhancementPlanner(Settings(max_extra_guidance=2))
    feedback = YehiaFeedback(ok=False, score=40, feedback="المعنى غير واضح ويحتاج إلى مزيد من التركيز")
    plan = planner.build_plan(
        _base_request(),
        meter_eval=None,
        feedback=feedback,
        feedback_summary=["المعنى ضعيف"],
    )
    assert plan.request.poem_description != "قصيدة عن أمل المسافر"
    assert any("المعنى" in tip for tip in plan.request.extra_guidance)
    assert "tightened_description" in plan.applied_changes
