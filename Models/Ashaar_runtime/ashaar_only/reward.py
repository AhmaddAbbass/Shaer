"""
Ashaar-only meter reward utilities for Arabic poetry generation.

Exposes:
- ashaar_meter_pattern_score(text) -> float in [0, 1]
- ashaar_reward(text, config) -> {"reward", "ashaar_score"}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _lazy_import_bait_analysis():
    """
    Lazily import Ashaar BaitAnalysis and the empty_analysis sentinel.
    """
    try:
        from Ashaar.bait_analysis import BaitAnalysis, empty_analysis  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative error
        raise ImportError(
            "Ashaar.bait_analysis is required for Ashaar-based rewards. "
            "Make sure Models/Ashaar_runtime/ashaar_only is on PYTHONPATH."
        ) from exc
    return BaitAnalysis, empty_analysis


_BAIT_ANALYSIS: Any = None
_EMPTY_ANALYSIS: Any = None


def _get_bait_analysis():
    """
    Return a cached Ashaar `BaitAnalysis` instance and empty sentinel.
    """
    global _BAIT_ANALYSIS, _EMPTY_ANALYSIS

    if _BAIT_ANALYSIS is None or _EMPTY_ANALYSIS is None:
        BaitAnalysis, empty_analysis = _lazy_import_bait_analysis()
        _BAIT_ANALYSIS = BaitAnalysis()
        _EMPTY_ANALYSIS = empty_analysis

    return _BAIT_ANALYSIS, _EMPTY_ANALYSIS


def _normalize_text_for_ashaar(text: str) -> str:
    """
    Convert a bayt in `[sep]` format into the format expected by Ashaar.
    """
    if "[sep]" in text:
        parts = [p.strip() for p in text.split("[sep]")]
        if len(parts) == 2:
            return " # ".join(parts)
    return text


def ashaar_meter_pattern_score(text: str) -> float:
    """
    Compute a structural meter similarity score using Ashaar's BaitAnalysis.

    Returns a mean ratio in [0, 1].
    """
    analysis_obj, empty_analysis = _get_bait_analysis()

    bait = _normalize_text_for_ashaar(text)

    analysis = analysis_obj.analyze(
        baits=[bait],
        short_qafiyah=False,
        override_tashkeel=False,
        highlight_output=False,
        predict_era=False,
        predict_theme=False,
        predict_closest=True,  # needed for closest_patterns ratios
    )

    if analysis is empty_analysis:
        return 0.0

    closest_patterns = analysis.get("closest_patterns")
    if not closest_patterns:
        return 0.0

    ratios = [float(t[1]) for t in closest_patterns if len(t) >= 2]
    if not ratios:
        return 0.0

    mean_ratio = sum(ratios) / len(ratios)
    return max(0.0, min(1.0, mean_ratio))


@dataclass
class AshaarRewardConfig:
    """
    Configuration for Ashaar structural rewards.

    ashaar_weight:
        Optional scaling applied to the Ashaar score before clamping to [0, 1].
    """

    ashaar_weight: float = 1.0


def ashaar_reward(
    text: str,
    config: Optional[AshaarRewardConfig] = None,
) -> Dict[str, float]:
    """
    Compute an Ashaar-only reward for a generated bayt.
    """
    if config is None:
        config = AshaarRewardConfig()

    ashaar_score = ashaar_meter_pattern_score(text)
    weighted = max(0.0, min(1.0, ashaar_score * max(config.ashaar_weight, 0.0)))

    return {
        "reward": float(weighted),
        "ashaar_score": float(ashaar_score),
    }


__all__ = [
    "AshaarRewardConfig",
    "ashaar_meter_pattern_score",
    "ashaar_reward",
]
