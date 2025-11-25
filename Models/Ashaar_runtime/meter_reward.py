"""
Meter-level reward utilities for Arabic poetry generation (GRPO, RL, etc.).

This module combines:
1) A Hugging Face MARBERT meter classifier
2) The Ashaar `BaitAnalysis` structural meter analysis

Given a generated bayt (two hemistichs separated by ``[sep]``), it produces:
- A MARBERT probability distribution over meters
- A structural similarity score from Ashaar (0–1)
- A combined scalar reward suitable for GRPO

Example
-------
from meter_reward import meter_reward, marbert_meter_scores, ashaar_meter_pattern_score

text = "ويوم نلتقي فيه قصير[sep]يطول اليوم لا ألقاك فيه"

scores = marbert_meter_scores(text)
ashaar_score = ashaar_meter_pattern_score(text)
reward = meter_reward(text)["reward"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


def _lazy_import_transformers_pipeline():
    """Lazily import the HF pipeline to avoid mandatory dependency at import time."""
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is required for MARBERT meter rewards. "
            "Install it (see requirements.txt) before using MARBERT-based scoring."
        ) from exc

    return pipeline


def _lazy_import_bait_analysis():
    """Lazily import Ashaar BaitAnalysis to avoid side effects on import."""
    try:
        from Ashaar.bait_analysis import BaitAnalysis, empty_analysis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Ashaar.bait_analysis is required for Ashaar-based meter rewards. "
            "Make sure you are running from `Models/Ashaar_runtime` or that this "
            "directory is on PYTHONPATH."
        ) from exc

    return BaitAnalysis, empty_analysis


# ---------------------------------------------------------------------------
# Global singletons (created lazily so they are shared across calls)
# ---------------------------------------------------------------------------

_MARBERT_CLASSIFIER = None
_BAIT_ANALYSIS = None


def _get_marbert_classifier(
    model_name: str = "Ammar-alhaj-ali/arabic-MARBERT-poetry-classification",
    device: Optional[int] = None,
):
    """
    Return a cached Hugging Face pipeline for MARBERT meter classification.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.
    device:
        Optional device index for GPU; if None, uses the transformers default.
    """
    global _MARBERT_CLASSIFIER

    if _MARBERT_CLASSIFIER is None:
        pipeline = _lazy_import_transformers_pipeline()
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        _MARBERT_CLASSIFIER = pipeline(
            "text-classification",
            model=model_name,
            **kwargs,
        )
    return _MARBERT_CLASSIFIER


def _get_bait_analysis():
    """
    Return a cached Ashaar `BaitAnalysis` instance.

    Uses the default configuration, which expects:
    - `test.yml` in the Ashaar_runtime root
    - `deep-learning-models` folder under `Ashaar_runtime/Ashaar`
    """
    global _BAIT_ANALYSIS

    if _BAIT_ANALYSIS is None:
        BaitAnalysis, _ = _lazy_import_bait_analysis()
        # Use default `abs_path="."` and package-relative paths from Ashaar.
        _BAIT_ANALYSIS = BaitAnalysis()
    return _BAIT_ANALYSIS


def _normalize_text_for_ashaar(text: str) -> str:
    """
    Convert a bayt in `[sep]` format into the format expected by Ashaar.

    Ashaar's `BaitAnalysis.analyze` expects a string where shatrain are
    separated by `#`, e.g.: "شطر أول # شطر ثان".
    """
    if "[sep]" in text:
        parts = [p.strip() for p in text.split("[sep]")]
        # BaitAnalysis ignores baits that don't have exactly two shatrain.
        if len(parts) == 2:
            return " # ".join(parts)
    return text


# ---------------------------------------------------------------------------
# Public configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class MeterRewardConfig:
    """
    Configuration for combining MARBERT and Ashaar scores into a single reward.

    Attributes
    ----------
    marbert_weight:
        Relative weight for the MARBERT confidence score.
    ashaar_weight:
        Relative weight for the Ashaar structural similarity score.
    marbert_target_label:
        If provided, use the MARBERT probability assigned to this label
        (exact string match on `classifier(text)[i]['label']`).
        If None, the MARBERT component uses the maximum probability.
    """

    marbert_weight: float = 0.5
    ashaar_weight: float = 0.5
    marbert_target_label: Optional[str] = None


# ---------------------------------------------------------------------------
# MARBERT-based scoring
# ---------------------------------------------------------------------------


def marbert_meter_scores(
    text: str,
    model_name: str = "Ammar-alhaj-ali/arabic-MARBERT-poetry-classification",
    device: Optional[int] = None,
) -> Dict[str, float]:
    """
    Return MARBERT's probability distribution over poetic meters for a bayt.

    Parameters
    ----------
    text:
        The bayt text. For two-hemistich verses pass it as:
        "الشطر الأول[sep]الشطر الثاني".
    model_name:
        Hugging Face model name or path.
    device:
        Optional device index for GPU; if None, transformers chooses.

    Returns
    -------
    dict
        Mapping from MARBERT `label` → `score` (float in [0, 1]).
    """
    classifier = _get_marbert_classifier(model_name=model_name, device=device)

    # The pipeline returns a list of dicts. By default this is top-1; using
    # `top_k=None` gives full distribution for newer transformers, but to keep
    # compatibility we just handle the returned list as-is.
    results = classifier(text)

    # Some transformers versions return a single dict instead of list; normalize.
    if isinstance(results, dict):
        results = [results]

    return {str(res["label"]): float(res["score"]) for res in results}


# ---------------------------------------------------------------------------
# Ashaar-based structural similarity scoring
# ---------------------------------------------------------------------------


def ashaar_meter_pattern_score(text: str) -> float:
    """
    Compute a structural meter similarity score using Ashaar's BaitAnalysis.

    The score is the mean over shatrain of the `ratio` values returned from
    `closest_patterns` in `BaitAnalysis.analyze`, i.e. SequenceMatcher-based
    similarity between the predicted pattern and the closest canonical pattern
    for the inferred meter (0 = completely different, 1 = perfect match).

    This uses the auto-diacritization and meter model from Ashaar.

    Parameters
    ----------
    text:
        Bayt text, with optional `[sep]` delimiter between shatrain.

    Returns
    -------
    float
        Mean structural similarity score in [0, 1]. Returns 0.0 if analysis
        fails or if no valid shatrain are produced.
    """
    BaitAnalysis, empty_analysis = _lazy_import_bait_analysis()
    analysis_obj = _get_bait_analysis()

    bait = _normalize_text_for_ashaar(text)
    analysis = analysis_obj.analyze(
        baits=[bait],
        short_qafiyah=False,
        override_tashkeel=False,
        highlight_output=False,
        predict_era=False,
        predict_theme=False,
        predict_closest=False,
    )

    # If for some reason analysis returned the sentinel empty dict, give 0 reward.
    if analysis is empty_analysis or not analysis.get("closest_patterns"):
        return 0.0

    ratios = [float(t[1]) for t in analysis["closest_patterns"] if len(t) >= 2]
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


# ---------------------------------------------------------------------------
# Combined meter reward
# ---------------------------------------------------------------------------


def meter_reward(
    text: str,
    config: Optional[MeterRewardConfig] = None,
    *,
    model_name: str = "Ammar-alhaj-ali/arabic-MARBERT-poetry-classification",
    device: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute a combined meter reward for a generated bayt.

    The reward is a weighted combination of:
    - MARBERT confidence (either max probability, or probability for a target
      label if `config.marbert_target_label` is provided)
    - Ashaar structural similarity score (pattern-level similarity)

    Parameters
    ----------
    text:
        Generated bayt, with shatrain separated by `[sep]` if applicable.
    config:
        `MeterRewardConfig` controlling component weights and MARBERT target label.
        If None, uses the default weights (0.5 / 0.5) and max-probability MARBERT.
    model_name:
        Hugging Face MARBERT model identifier.
    device:
        Optional device index for GPU for MARBERT.

    Returns
    -------
    dict
        {
            "reward": float,           # combined scalar reward in [0, 1]
            "marbert_score": float,    # MARBERT component (0–1)
            "ashaar_score": float,     # Ashaar structural component (0–1)
            "marbert_distribution": {label: prob, ...},
        }
    """
    if config is None:
        config = MeterRewardConfig()

    marbert_dist: Dict[str, float] = {}
    marbert_score = 0.0

    if config.marbert_weight > 0.0:
        marbert_dist = marbert_meter_scores(
            text=text,
            model_name=model_name,
            device=device,
        )
        if marbert_dist:
            if config.marbert_target_label is None:
                # Use the highest confidence across meters.
                marbert_score = max(marbert_dist.values())
            else:
                # Use the confidence assigned to the target meter label.
                marbert_score = float(marbert_dist.get(config.marbert_target_label, 0.0))

    ashaar_score = 0.0
    if config.ashaar_weight > 0.0:
        ashaar_score = ashaar_meter_pattern_score(text)

    total_weight = max(config.marbert_weight + config.ashaar_weight, 1e-8)
    combined = (config.marbert_weight * marbert_score + config.ashaar_weight * ashaar_score) / total_weight

    return {
        "reward": float(combined),
        "marbert_score": float(marbert_score),
        "ashaar_score": float(ashaar_score),
        "marbert_distribution": marbert_dist,
    }


__all__ = [
    "MeterRewardConfig",
    "marbert_meter_scores",
    "ashaar_meter_pattern_score",
    "meter_reward",
]

