"""
Meter-level reward utilities for Arabic poetry generation (GRPO, RL, etc.).

This module combines:
1) An AraPoemBERT-based meter classifier
   (default: `faisalq/bert-base-arapoembert`
    assumed to be used with a meter classification head)
2) The Ashaar `BaitAnalysis` structural meter analysis

Given a generated bayt (two hemistichs optionally separated by `[sep]`), it produces:
- A classifier probability distribution over meters
- A structural similarity score from Ashaar (0–1)
- A combined scalar reward suitable for GRPO

Example
-------
from meter_reward import meter_reward, classifier_meter_scores, ashaar_meter_pattern_score, MeterRewardConfig

text = "ويوم نلتقي فيه قصير[sep]يطول اليوم لا ألقاك فيه"
target_meter = "البسيط"

cfg = MeterRewardConfig(
    classifier_weight=0.5,
    ashaar_weight=0.5,
    classifier_target_label=target_meter,
)

res = meter_reward(text, config=cfg)
print(res["reward"], res["classifier_score"], res["ashaar_score"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _lazy_import_transformers_pipeline():
    """Lazily import the HF pipeline to avoid mandatory dependency at import time."""
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers is required for AraPoemBERT-based meter rewards. "
            "Install it (see requirements.txt) before using classifier-based scoring."
        ) from exc
    return pipeline


def _lazy_import_bait_analysis():
    """
    Lazily import Ashaar BaitAnalysis and the empty_analysis sentinel.

    This expects that:
    - The `Ashaar_runtime` repo is available on PYTHONPATH.
    - `Ashaar.bait_analysis` can be imported.
    """
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

_DEFAULT_METER_CLASSIFIER_MODEL = "faisalq/bert-base-arapoembert"

_METER_CLASSIFIER = None  # type: ignore[var-annotated]
_BAIT_ANALYSIS: Any = None
_EMPTY_ANALYSIS: Any = None


def _get_meter_classifier(
    model_name: str = _DEFAULT_METER_CLASSIFIER_MODEL,
    device: Optional[int] = None,
):
    """
    Return a cached Hugging Face pipeline for meter classification.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier. By default:
        `faisalq/bert-base-arapoembert` (AraPoemBERT base, assumed fine-tuned
        or wrapped for meter classification).
    device:
        Optional device index for GPU; if None, uses the transformers default.
    """
    global _METER_CLASSIFIER

    if _METER_CLASSIFIER is None:
        pipeline = _lazy_import_transformers_pipeline()
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        _METER_CLASSIFIER = pipeline(
            "text-classification",
            model=model_name,
            **kwargs,
        )
    return _METER_CLASSIFIER


def _get_bait_analysis():
    """
    Return a cached Ashaar `BaitAnalysis` instance and ensure the empty sentinel is set.

    Uses the default configuration, which expects:
    - `test.yml` in the Ashaar_runtime root
    - `deep-learning-models` folder under `Ashaar_runtime/Ashaar`
    """
    global _BAIT_ANALYSIS, _EMPTY_ANALYSIS

    if _BAIT_ANALYSIS is None or _EMPTY_ANALYSIS is None:
        BaitAnalysis, empty_analysis = _lazy_import_bait_analysis()
        _BAIT_ANALYSIS = BaitAnalysis()
        _EMPTY_ANALYSIS = empty_analysis

    return _BAIT_ANALYSIS


def _get_empty_analysis_sentinel():
    """Return the global empty_analysis sentinel from Ashaar, if initialized."""
    global _EMPTY_ANALYSIS
    if _EMPTY_ANALYSIS is None:
        _get_bait_analysis()
    return _EMPTY_ANALYSIS


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
    Configuration for combining classifier and Ashaar scores into a single reward.

    Attributes
    ----------
    classifier_weight:
        Relative weight for the classifier confidence score.
    ashaar_weight:
        Relative weight for the Ashaar structural similarity score.
    classifier_target_label:
        If provided, use the classifier probability assigned to this label
        (exact string match on `classifier(text)[i]['label']`).
        If None, the classifier component uses the maximum probability.
    """

    classifier_weight: float = 0.5
    ashaar_weight: float = 0.5
    classifier_target_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Classifier-based scoring (AraPoemBERT-based meter classifier)
# ---------------------------------------------------------------------------


def classifier_meter_scores(
    text: str,
    model_name: str = _DEFAULT_METER_CLASSIFIER_MODEL,
    device: Optional[int] = None,
) -> Dict[str, float]:
    """
    Return the classifier's probability distribution over poetic meters for a bayt.

    Parameters
    ----------
    text:
        The bayt text. For two-hemistich verses pass it as:
        "الشطر الأول[sep]الشطر الثاني".
    model_name:
        Hugging Face model name or path. By default uses the AraPoemBERT base
        model: `faisalq/bert-base-arapoembert` (assumed to have a classification head).
    device:
        Optional device index for GPU; if None, transformers chooses.

    Returns
    -------
    dict
        Mapping from classifier `label` → `score` (float in [0, 1]).

    Notes
    -----
    Depending on transformers version, the pipeline may return:
    - a single dict, or
    - a list of dicts (top-k results).
    This function normalizes both to `Dict[label, score]`.
    """
    classifier = _get_meter_classifier(model_name=model_name, device=device)

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
    analysis_obj = _get_bait_analysis()
    empty_analysis = _get_empty_analysis_sentinel()

    bait = _normalize_text_for_ashaar(text)

    # Important: predict_closest=True to actually populate `closest_patterns`
    analysis = analysis_obj.analyze(
        baits=[bait],
        short_qafiyah=False,
        override_tashkeel=False,
        highlight_output=False,
        predict_era=False,
        predict_theme=False,
        predict_closest=True,
    )

    # Some Ashaar versions return `empty_analysis` sentinel on failure.
    if analysis is empty_analysis:
        return 0.0

    closest_patterns = analysis.get("closest_patterns")
    if not closest_patterns:
        return 0.0

    # closest_patterns is expected to be a list of tuples:
    # (pattern_str, ratio, taf3ilat_str, ...)
    ratios = [float(t[1]) for t in closest_patterns if len(t) >= 2]
    if not ratios:
        return 0.0

    # Clamp to [0, 1] just in case
    mean_ratio = sum(ratios) / len(ratios)
    mean_ratio = max(0.0, min(1.0, mean_ratio))
    return mean_ratio


# ---------------------------------------------------------------------------
# Combined meter reward
# ---------------------------------------------------------------------------


def meter_reward(
    text: str,
    config: Optional[MeterRewardConfig] = None,
    *,
    model_name: str = _DEFAULT_METER_CLASSIFIER_MODEL,
    device: Optional[int] = None,
) -> Dict[str, object]:
    """
    Compute a combined meter reward for a generated bayt.

    The reward is a weighted combination of:
    - classifier confidence (either max probability, or probability for a target
      label if `config.classifier_target_label` is provided)
    - Ashaar structural similarity score (pattern-level similarity)

    Parameters
    ----------
    text:
        Generated bayt, with shatrain separated by `[sep]` if applicable.
    config:
        `MeterRewardConfig` controlling component weights and classifier target label.
        If None, uses the default weights (0.5 / 0.5) and max-probability classifier.
    model_name:
        Hugging Face classifier model identifier. Defaults to
        `faisalq/bert-base-arapoembert`.
    device:
        Optional device index for GPU for the classifier.

    Returns
    -------
    dict
        {
            "reward": float,             # combined scalar reward in [0, 1]
            "classifier_score": float,   # classifier component (0–1)
            "ashaar_score": float,       # Ashaar structural component (0–1)
            "classifier_distribution": {label: prob, ...},
        }
    """
    if config is None:
        config = MeterRewardConfig()

    # -------------------------
    # Classifier component
    # -------------------------
    classifier_dist: Dict[str, float] = {}
    classifier_score = 0.0

    if config.classifier_weight > 0.0:
        classifier_dist = classifier_meter_scores(
            text=text,
            model_name=model_name,
            device=device,
        )
        if classifier_dist:
            if config.classifier_target_label is None:
                # Use the highest confidence across meters.
                classifier_score = max(classifier_dist.values())
            else:
                # Use the confidence assigned to the target meter label.
                classifier_score = float(
                    classifier_dist.get(config.classifier_target_label, 0.0)
                )

    # Ensure score is in [0, 1]
    classifier_score = max(0.0, min(1.0, classifier_score))

    # -------------------------
    # Ashaar component
    # -------------------------
    ashaar_score = 0.0
    if config.ashaar_weight > 0.0:
        ashaar_score = ashaar_meter_pattern_score(text)
        ashaar_score = max(0.0, min(1.0, ashaar_score))

    # -------------------------
    # Combine
    # -------------------------
    total_weight = config.classifier_weight + config.ashaar_weight
    if total_weight <= 0.0:
        combined = 0.0
    else:
        combined = (
            config.classifier_weight * classifier_score
            + config.ashaar_weight * ashaar_score
        ) / total_weight

    combined = max(0.0, min(1.0, combined))

    return {
        "reward": float(combined),
        "classifier_score": float(classifier_score),
        "ashaar_score": float(ashaar_score),
        "classifier_distribution": classifier_dist,
    }


__all__ = [
    "MeterRewardConfig",
    "classifier_meter_scores",
    "ashaar_meter_pattern_score",
    "meter_reward",
]
