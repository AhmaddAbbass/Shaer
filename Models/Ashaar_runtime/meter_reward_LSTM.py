"""
Meter-level reward utilities for Arabic poetry generation (GRPO, RL, etc.).

This version uses:
1) Your char-level BiLSTM meter classifier:
   - bilstm_model/training/poem_meter_bilstm.keras
   - bilstm_model/training/meter_label_encoder.joblib
   - bilstm_model/training/bilstm_vocab_config.json
2) Ashaar `BaitAnalysis` for structural meter similarity.

API:
- classifier_meter_scores(text) -> {label: prob}
- ashaar_meter_pattern_score(text) -> float in [0, 1]
- meter_reward(text, config) -> dict with:
    "reward", "classifier_score", "ashaar_score", "classifier_distribution"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path
import json

import numpy as np

# ---------------------------------------------------------------------------
# Paths to your BiLSTM assets
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = ROOT_DIR / "bilstm_model" / "training"

_DEFAULT_BILSTM_MODEL_PATH = TRAINING_DIR / "poem_meter_bilstm.keras"
_DEFAULT_LABEL_ENCODER_PATH = TRAINING_DIR / "meter_label_encoder.joblib"
_DEFAULT_VOCAB_CONFIG_PATH = TRAINING_DIR / "bilstm_vocab_config.json"


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _lazy_import_tf():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tensorflow is required for BiLSTM meter rewards. "
            "Install it in your environment before using this module."
        ) from exc
    return tf


def _lazy_import_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load the meter_label_encoder.joblib. "
            "Install it before using this module."
        ) from exc
    return joblib


def _lazy_import_bait_analysis():
    """
    Lazily import Ashaar BaitAnalysis and the empty_analysis sentinel.
    """
    try:
        from Ashaar.bait_analysis import BaitAnalysis, empty_analysis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Ashaar.bait_analysis is required for Ashaar-based meter rewards. "
            "Make sure the Ashaar_runtime repo is on PYTHONPATH."
        ) from exc
    return BaitAnalysis, empty_analysis


# ---------------------------------------------------------------------------
# Global singletons (created lazily so they are shared across calls)
# ---------------------------------------------------------------------------

_BILSTM_MODEL = None      # type: ignore[var-annotated]
_LABEL_ENCODER = None
_STOI: Dict[str, int] = {}
_MAX_LEN: int = 0

_BAIT_ANALYSIS: Any = None
_EMPTY_ANALYSIS: Any = None


def _load_bilstm_assets(
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
):
    """
    Load (once) the BiLSTM keras model, label encoder, and char vocab config.
    """
    global _BILSTM_MODEL, _LABEL_ENCODER, _STOI, _MAX_LEN

    if _BILSTM_MODEL is None or _LABEL_ENCODER is None or not _STOI:
        tf = _lazy_import_tf()
        joblib = _lazy_import_joblib()

        # Load keras model
        _BILSTM_MODEL = tf.keras.models.load_model(str(model_path))

        # Load label encoder
        _LABEL_ENCODER = joblib.load(str(label_encoder_path))

        # Load stoi + max_len
        with open(vocab_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _STOI = {str(k): int(v) for k, v in cfg["stoi"].items()}
        _MAX_LEN = int(cfg["max_len"])


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

    Ashaar expects shatrain separated by '#', e.g. "شطر أول # شطر ثان".
    """
    if "[sep]" in text:
        parts = [p.strip() for p in text.split("[sep]")]
        if len(parts) == 2:
            return " # ".join(parts)
    return text


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MeterRewardConfig:
    """
    Configuration for combining BiLSTM classifier and Ashaar scores into a reward.

    classifier_weight:
        Weight for BiLSTM meter confidence.
    ashaar_weight:
        Weight for Ashaar structural similarity.
    classifier_target_label:
        If provided, use probability assigned to this label; otherwise use max prob.
    """

    classifier_weight: float = 0.5
    ashaar_weight: float = 0.5
    classifier_target_label: Optional[str] = None


# ---------------------------------------------------------------------------
# BiLSTM classifier-based scoring
# ---------------------------------------------------------------------------


def _encode_text_to_ints(text: str) -> np.ndarray:
    """
    Encode a single verse string into shape (1, max_len) int32 using the saved stoi.

    `[sep]` is replaced with a space before encoding.
    """
    if not _STOI or _MAX_LEN <= 0:
        _load_bilstm_assets()  # ensures _STOI and _MAX_LEN are set

    # Replace [sep] with a space so the BiLSTM doesn't see '[' 's' 'e' 'p'.
    clean = text.replace("[sep]", " ")
    seq = np.zeros((_MAX_LEN,), dtype="int32")
    for t, ch in enumerate(clean[:_MAX_LEN]):
        seq[t] = _STOI.get(ch, 0)
    return seq[None, :]  # shape (1, max_len)


def classifier_meter_scores(
    text: str,
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, float]:
    """
    Return BiLSTM classifier probability distribution over meters for a bayt.

    Returns
    -------
    dict
        label (meter name) -> probability in [0, 1]
    """
    _load_bilstm_assets(
        model_path=model_path,
        label_encoder_path=label_encoder_path,
        vocab_config_path=vocab_config_path,
    )

    X = _encode_text_to_ints(text)
    probs = _BILSTM_MODEL.predict(X, verbose=0)[0]  # (num_classes,)

    labels = list(_LABEL_ENCODER.classes_)
    return {str(lbl): float(probs[i]) for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# Ashaar-based structural similarity scoring
# ---------------------------------------------------------------------------


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
        predict_closest=True,  # IMPORTANT: needed for closest_patterns
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


# ---------------------------------------------------------------------------
# Combined meter reward
# ---------------------------------------------------------------------------


def meter_reward(
    text: str,
    config: Optional[MeterRewardConfig] = None,
    *,
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, object]:
    """
    Compute a combined meter reward for a generated bayt.

    Returns
    -------
    dict
        {
            "reward": float,               # combined scalar in [0, 1]
            "classifier_score": float,     # BiLSTM component
            "ashaar_score": float,         # Ashaar component
            "classifier_distribution": {label: prob, ...},
        }
    """
    if config is None:
        config = MeterRewardConfig()

    # ----- BiLSTM classifier component -----
    classifier_dist: Dict[str, float] = {}
    classifier_score = 0.0

    if config.classifier_weight > 0.0:
        classifier_dist = classifier_meter_scores(
            text=text,
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            vocab_config_path=vocab_config_path,
        )
        if classifier_dist:
            if config.classifier_target_label is None:
                classifier_score = max(classifier_dist.values())
            else:
                classifier_score = float(
                    classifier_dist.get(config.classifier_target_label, 0.0)
                )

    classifier_score = max(0.0, min(1.0, classifier_score))

    # ----- Ashaar component -----
    ashaar_score = 0.0
    if config.ashaar_weight > 0.0:
        ashaar_score = ashaar_meter_pattern_score(text)
        ashaar_score = max(0.0, min(1.0, ashaar_score))

    # ----- Combine -----
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
