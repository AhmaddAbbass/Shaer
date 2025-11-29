from __future__ import annotations

from typing import List

from services.scoring_service.app.schemas import ScoreResponse

LOW_SCORE_THRESHOLD = 0.7


def summarize_feedback(scoring: ScoreResponse) -> List[str]:
    bullets: List[str] = []
    meter_eval = scoring.meter_eval
    if not meter_eval.on_meter:
        bullets.append("ركّز على الالتزام بالوزن وتصحيح الكسور.")
    for aspect_name, title in (
        ("meaning", "المعنى"),
        ("cohesion", "الترابط"),
        ("fluency", "الفصاحة"),
        ("poeticness", "الشعرية"),
    ):
        aspect = getattr(scoring.aspects, aspect_name)
        score = aspect.judge.score_0_1
        if score is not None and score < LOW_SCORE_THRESHOLD:
            bullets.append(f"حسّن {title} بالالتزام بتوجيه يحيى: {aspect.yehia_feedback.feedback[:90]}...")
    if not bullets:
        bullets.append("حافظ على انسجام البيت مع الوصف العام وزد التدفق الشعري.")
    return bullets[:3]
