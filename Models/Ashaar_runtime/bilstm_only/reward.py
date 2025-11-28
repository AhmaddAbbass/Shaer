"""
BiLSTM-only meter reward utilities for Arabic poetry generation.

Exposes:
- classifier_meter_scores(text) -> Dict[meter_label, probability]
- bilstm_reward(text, config) -> {"reward", "classifier_score", "classifier_distribution"}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np

# ---------------------------------------------------------------------------
# Paths to the BiLSTM assets
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
    except ImportError as exc:  # pragma: no cover - informative error
        raise ImportError(
            "tensorflow is required for BiLSTM meter rewards. "
            "Install it in your environment before using this module."
        ) from exc
    return tf


def _lazy_import_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:  # pragma: no cover - informative error
        raise ImportError(
            "joblib is required to load meter_label_encoder.joblib. "
            "Install it in your environment before using this module."
        ) from exc
    return joblib


# ---------------------------------------------------------------------------
# Global singletons (shared across calls)
# ---------------------------------------------------------------------------

_BILSTM_MODEL = None  # type: ignore[var-annotated]
_LABEL_ENCODER = None
_STOI: Dict[str, int] = {}
_MAX_LEN: int = 0


def _load_bilstm_assets(
    *,
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
) -> None:
    """Load (once) the BiLSTM keras model, label encoder, and char vocab config."""
    global _BILSTM_MODEL, _LABEL_ENCODER, _STOI, _MAX_LEN

    if _BILSTM_MODEL is not None and _LABEL_ENCODER is not None and _STOI and _MAX_LEN > 0:
        return

    tf = _lazy_import_tf()
    joblib = _lazy_import_joblib()

    _BILSTM_MODEL = tf.keras.models.load_model(str(model_path))
    _LABEL_ENCODER = joblib.load(str(label_encoder_path))

    with open(vocab_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _STOI = {str(k): int(v) for k, v in cfg["stoi"].items()}
    _MAX_LEN = int(cfg["max_len"])


def _encode_text_to_ints(text: str) -> np.ndarray:
    """
    Encode a single verse string into shape (1, max_len) int32 using the saved stoi.

    `[sep]` is replaced with a space before encoding.
    """
    if not _STOI or _MAX_LEN <= 0:
        _load_bilstm_assets()

    clean = text.replace("[sep]", " ")
    seq = np.zeros((_MAX_LEN,), dtype="int32")
    for t, ch in enumerate(clean[:_MAX_LEN]):
        seq[t] = _STOI.get(ch, 0)
    return seq[None, :]


def classifier_meter_scores(
    text: str,
    *,
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
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
    probs = _BILSTM_MODEL.predict(X, verbose=0)[0]  # type: ignore[operator]

    labels = list(_LABEL_ENCODER.classes_)  # type: ignore[call-arg]
    return {str(lbl): float(probs[i]) for i, lbl in enumerate(labels)}


def _resolve_label(target_label: str, labels: Dict[str, float]) -> Optional[str]:
    """
    Match target_label against classifier labels ignoring whitespace/case.
    """
    target_norm = (target_label or "").strip().lower().replace(" ", "")
    if not target_norm:
        return None

    for lbl in labels.keys():
        norm = str(lbl).strip().lower().replace(" ", "")
        if norm == target_norm:
            return str(lbl)
    return None


@dataclass
class BilstmRewardConfig:
    """
    Configuration for BiLSTM-only rewards.

    classifier_target_label:
        If provided, reward uses the probability assigned to this label (case-insensitive match).
        Otherwise, reward uses the maximum probability across labels.
    """

    classifier_target_label: Optional[str] = None


def bilstm_reward(
    text: str,
    config: Optional[BilstmRewardConfig] = None,
    *,
    model_path: Path = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, object]:
    """
    Compute a BiLSTM-only meter reward for a bayt.
    """
    if config is None:
        config = BilstmRewardConfig()

    classifier_dist = classifier_meter_scores(
        text=text,
        model_path=model_path,
        label_encoder_path=label_encoder_path,
        vocab_config_path=vocab_config_path,
    )

    classifier_score = 0.0
    if classifier_dist:
        if config.classifier_target_label is None:
            classifier_score = max(classifier_dist.values())
        else:
            resolved = _resolve_label(config.classifier_target_label, classifier_dist)
            if resolved:
                classifier_score = float(classifier_dist.get(resolved, 0.0))

    classifier_score = max(0.0, min(1.0, classifier_score))

    return {
        "reward": float(classifier_score),
        "classifier_score": float(classifier_score),
        "classifier_distribution": classifier_dist,
    }


__all__ = [
    "BilstmRewardConfig",
    "classifier_meter_scores",
    "bilstm_reward",
]
