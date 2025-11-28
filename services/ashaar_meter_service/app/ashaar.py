from __future__ import annotations

from typing import Any, Dict

from .config import Settings
from .logging import get_logger
from .schemas import AshaarScore

logger = get_logger("ashaar_meter_service.ashaar")

try:
    from Models.Ashaar_runtime.ashaar_only.reward import (
        AshaarRewardConfig,
        ashaar_reward,
    )
except ImportError:  # pragma: no cover - Ashaar package missing
    AshaarRewardConfig = None  # type: ignore[assignment]
    ashaar_reward = None  # type: ignore[assignment]


class AshaarMeterService:
    """
    Lightweight wrapper around Ashaar's structural scoring utilities.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assets_loaded = self._check_assets()

    def _check_assets(self) -> bool:
        if AshaarRewardConfig is None or ashaar_reward is None:
            logger.warning(
                "Ashaar reward utilities not available; install Models/Ashaar_runtime dependencies."
            )
            return False
        return True

    def evaluate_bayt(self, verse_text: str) -> AshaarScore:
        if not self.assets_loaded or AshaarRewardConfig is None or ashaar_reward is None:
            return self._default_score()

        cfg = AshaarRewardConfig(ashaar_weight=self.settings.ashaar_weight)
        try:
            result: Dict[str, Any] = ashaar_reward(verse_text, config=cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Ashaar reward computation failed: %s", exc)
            return self._default_score()

        reward = self._clamp(float(result.get("reward", 0.0)))
        raw_score = self._clamp(float(result.get("ashaar_score", 0.0)))
        notes = f"درجة التشابه العروضي: {raw_score:.2f}"
        return AshaarScore(reward=reward, ashaar_score=raw_score, notes=notes)

    def _default_score(self) -> AshaarScore:
        """Fallback result when Ashaar assets are unavailable."""
        return AshaarScore(
            reward=0.0,
            ashaar_score=0.0,
            notes="لا يمكن حساب الوزن العروضي حالياً؛ يرجى التحقق من إعدادات الخدمة.",
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
