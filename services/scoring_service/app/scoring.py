from __future__ import annotations

from dataclasses import dataclass

from .config import Settings

WEIGHTS = {
    "meter_score": 0.50,
    "meaning": 0.15,
    "fluency": 0.15,
    "poeticness": 0.10,
    "cohesion": 0.10,
}


@dataclass
class ScoreComponents:
    meter_score: float
    meaning: float
    fluency: float
    poeticness: float
    cohesion: float


@dataclass
class ScoreResult:
    final_score: float
    passed: bool
    breakdown: dict[str, float]


class ScoreCalculator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def compute(self, components: ScoreComponents) -> ScoreResult:
        breakdown: dict[str, float] = {}
        final_score = 0.0
        for key, weight in WEIGHTS.items():
            value = getattr(components, key)
            contribution = float(weight * value)
            breakdown[key] = contribution
            final_score += contribution
        final_score = min(max(final_score, 0.0), 1.0)
        passed = final_score >= self.settings.pass_threshold
        breakdown["final_score"] = final_score
        return ScoreResult(final_score=final_score, passed=passed, breakdown=breakdown)
