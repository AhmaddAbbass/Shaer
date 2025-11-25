"""
Meter-level reward utilities used during GRPO training.

This adapts the BiLSTM-based meter reward from
`Models/Ashaar_runtime/meter_reward_LSTM.py` and exposes a GRPO-friendly API:

- `meter_reward(completions, prompts, poem_meter=None, ...)` -> list of floats
- `meter_reward_single(text, ...)` -> detailed dict for one bayt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json
import sys

import numpy as np

from .form_reward import _extract_text_from_completion

# ---------------------------------------------------------------------------
# Paths to BiLSTM assets (anchored to Models/Ashaar_runtime)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASHAAR_RUNTIME_DIR = PROJECT_ROOT / "Models" / "Ashaar_runtime"
TRAINING_DIR = ASHAAR_RUNTIME_DIR / "bilstm_model" / "training"

# Ensure Ashaar_runtime is on sys.path so `import Ashaar...` works.
if str(ASHAAR_RUNTIME_DIR) not in sys.path:
    sys.path.append(str(ASHAAR_RUNTIME_DIR))

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
    try:
        from Ashaar.bait_analysis import BaitAnalysis, empty_analysis  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Ashaar.bait_analysis is required for Ashaar-based meter rewards. "
            "Ensure Models/Ashaar_runtime is on PYTHONPATH."
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
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
):
    """
    Load (once) the BiLSTM keras model, label encoder, and char vocab config.
    """
    global _BILSTM_MODEL, _LABEL_ENCODER, _STOI, _MAX_LEN

    if _BILSTM_MODEL is None or _LABEL_ENCODER is None or not _STOI:
        tf = _lazy_import_tf()
        joblib = _lazy_import_joblib()

        model_path = Path(model_path)
        label_encoder_path = Path(label_encoder_path)
        vocab_config_path = Path(vocab_config_path)

        _BILSTM_MODEL = tf.keras.models.load_model(str(model_path))
        _LABEL_ENCODER = joblib.load(str(label_encoder_path))

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
    """
    if not _STOI or _MAX_LEN <= 0:
        _load_bilstm_assets()  # ensures _STOI and _MAX_LEN are set

    clean = text.replace("[sep]", " ")
    seq = np.zeros((_MAX_LEN,), dtype="int32")
    for t, ch in enumerate(clean[:_MAX_LEN]):
        seq[t] = _STOI.get(ch, 0)
    return seq[None, :]  # shape (1, max_len)


def classifier_meter_scores(
    text: str,
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, float]:
    """
    Return BiLSTM classifier probability distribution over meters for a bayt.
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
        predict_closest=True,  # needed for closest_patterns
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
# Combined meter reward (single text)
# ---------------------------------------------------------------------------


def meter_reward_single(
    text: str,
    config: Optional[MeterRewardConfig] = None,
    *,
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, object]:
    """
    Compute a combined meter reward for a generated bayt.
    """
    if config is None:
        config = MeterRewardConfig()

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

    ashaar_score = 0.0
    if config.ashaar_weight > 0.0:
        ashaar_score = ashaar_meter_pattern_score(text)
        ashaar_score = max(0.0, min(1.0, ashaar_score))

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


# ---------------------------------------------------------------------------
# GRPO-friendly batch wrapper
# ---------------------------------------------------------------------------


def meter_reward(
    completions: Iterable[Any],
    prompts: Iterable[Any] | None = None,
    poem_meter: Optional[Iterable[str] | str] = None,
    *,
    classifier_weight: float = 0.5,
    ashaar_weight: float = 0.5,
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
) -> list[float]:
    """
    Batch meter reward for GRPO: returns one scalar per completion.

    Parameters
    ----------
    completions:
        Model outputs to score (same shapes as other reward functions).
    poem_meter:
        Optional target meter(s); can be a single string or iterable aligned
        with `completions`. If provided, the classifier probability for that
        label is used instead of max probability.
    classifier_weight / ashaar_weight:
        Relative weights for combining classifier and Ashaar components.
    """
    completions_list = list(completions or [])
    n = len(completions_list)

    if poem_meter is None:
        target_meters = [None] * n
    elif isinstance(poem_meter, str):
        target_meters = [poem_meter] * n
    else:
        target_meters = list(poem_meter)
        if len(target_meters) < n:
            target_meters += [None] * (n - len(target_meters))
        elif len(target_meters) > n:
            target_meters = target_meters[:n]

    rewards: list[float] = []

    for i, completion in enumerate(completions_list):
        text = _extract_text_from_completion(completion)
        if not isinstance(text, str):
            text = str(text or "")

        cfg = MeterRewardConfig(
            classifier_weight=classifier_weight,
            ashaar_weight=ashaar_weight,
            classifier_target_label=target_meters[i],
        )

        res = meter_reward_single(
            text=text,
            config=cfg,
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            vocab_config_path=vocab_config_path,
        )
        rewards.append(float(res["reward"]))

    return rewards


__all__ = [
    "MeterRewardConfig",
    "classifier_meter_scores",
    "ashaar_meter_pattern_score",
    "meter_reward_single",
    "meter_reward",
]
