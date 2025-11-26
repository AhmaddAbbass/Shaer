"""
BiLSTM meter classifier utilities (GRPO-ready, classifier-only).

Exports:
- classifier_meter_scores(text) -> Dict[meter_label, prob]
- meter_reward(completions, poem_meter=None, ...) -> list[float]   (batch helper)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json
import sys

import numpy as np

from .form_reward import _extract_text_from_completion

# ---------------------------------------------------------------------------
# Paths (anchored to Models/Ashaar_runtime)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASHAAR_RUNTIME_DIR = PROJECT_ROOT / "Models" / "Ashaar_runtime"
TRAINING_DIR = ASHAAR_RUNTIME_DIR / "bilstm_model" / "training"

_DEFAULT_BILSTM_MODEL_PATH = TRAINING_DIR / "poem_meter_bilstm.keras"
_DEFAULT_LABEL_ENCODER_PATH = TRAINING_DIR / "meter_label_encoder.joblib"
_DEFAULT_VOCAB_CONFIG_PATH = TRAINING_DIR / "bilstm_vocab_config.json"


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _lazy_import_tf():
    """
    Import TensorFlow (tf + tf_keras backend).

    We keep this separate so that the rest of the file can be imported
    even if TF is not installed; the error will only surface when
    meter_reward is actually used.
    """
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "tensorflow is required for the BiLSTM meter classifier. "
            "Install it in your environment before using meter_reward."
        ) from exc
    return tf


def _lazy_import_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load the meter_label_encoder.joblib. "
            "Install it before using the BiLSTM meter classifier."
        ) from exc
    return joblib


# ---------------------------------------------------------------------------
# Global singletons (cached across calls)
# ---------------------------------------------------------------------------

_BILSTM_MODEL = None      # type: ignore[var-annotated]
_LABEL_ENCODER = None     # type: ignore[var-annotated]
_STOI: Dict[str, int] = {}
_MAX_LEN: int = 0


def _load_bilstm_assets(
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
) -> None:
    """
    Load (once) the BiLSTM keras model, label encoder, and char vocab.

    This includes a compatibility shim for older keras configs that use
    `batch_shape` in the InputLayer.
    """
    global _BILSTM_MODEL, _LABEL_ENCODER, _STOI, _MAX_LEN

    if _BILSTM_MODEL is not None and _LABEL_ENCODER is not None and _STOI and _MAX_LEN > 0:
        return  # already loaded

    tf = _lazy_import_tf()
    joblib = _lazy_import_joblib()

    model_path = Path(model_path)
    label_encoder_path = Path(label_encoder_path)
    vocab_config_path = Path(vocab_config_path)

    # --- Keras compatibility shim for InputLayer(batch_shape=...) ----------
    class LegacyInputLayer(tf.keras.layers.InputLayer):
        """
        Wrapper that maps old `batch_shape` kwarg to `batch_input_shape`
        so that models saved with older Keras still deserialize cleanly.
        """
        def __init__(self, *args, **kwargs):
            batch_shape = kwargs.pop("batch_shape", None)
            # If an old config has batch_shape, map it to batch_input_shape.
            if batch_shape is not None and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = batch_shape
            super().__init__(*args, **kwargs)

    try:
        # Try to load with custom InputLayer and safe_mode=False so that
        # our custom_objects override actually takes effect.
        _BILSTM_MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"InputLayer": LegacyInputLayer},
            compile=False,
            safe_mode=False,  # IMPORTANT for custom_objects in newer tf_keras
        )
    except TypeError:
        # Older TF / tf_keras may not accept safe_mode kwarg.
        # Retry without it.
        _BILSTM_MODEL = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"InputLayer": LegacyInputLayer},
            compile=False,
        )

    # Label encoder
    _LABEL_ENCODER = joblib.load(str(label_encoder_path))

    # Vocab / max_len config
    with open(vocab_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _STOI = {str(k): int(v) for k, v in cfg["stoi"].items()}
    _MAX_LEN = int(cfg["max_len"])


def _encode_text_to_ints(text: str) -> np.ndarray:
    """
    Encode a single verse string into shape (1, max_len) int32 using saved stoi.
    """
    if not _STOI or _MAX_LEN <= 0:
        _load_bilstm_assets()

    clean = text.replace("[sep]", " ")
    seq = np.zeros((_MAX_LEN,), dtype="int32")
    for t, ch in enumerate(clean[:_MAX_LEN]):
        seq[t] = _STOI.get(ch, 0)
    return seq[None, :]  # (1, max_len)


def classifier_meter_scores(
    text: str,
    *,
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
) -> Dict[str, float]:
    """
    Return BiLSTM classifier probability distribution over meters for one bayt.

    If anything goes wrong with TF / Keras deserialization, we catch the error
    and return {} so that GRPO can continue (reward becomes 0 for this head).
    """
    global _BILSTM_MODEL, _LABEL_ENCODER

    try:
        _load_bilstm_assets(
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            vocab_config_path=vocab_config_path,
        )
    except Exception as e:
        # Fail soft: log to stderr and return empty distribution.
        print(f"[meter_reward] ⚠️ Failed to load BiLSTM assets: {e}", file=sys.stderr)
        _BILSTM_MODEL = None
        _LABEL_ENCODER = None
        return {}

    if _BILSTM_MODEL is None or _LABEL_ENCODER is None:
        return {}

    try:
        X = _encode_text_to_ints(text)
        probs = _BILSTM_MODEL.predict(X, verbose=0)[0]  # (num_classes,)
    except Exception as e:
        print(f"[meter_reward] ⚠️ BiLSTM predict failed: {e}", file=sys.stderr)
        return {}

    labels = list(_LABEL_ENCODER.classes_)
    return {str(lbl): float(probs[i]) for i, lbl in enumerate(labels)}


# ---------------------------------------------------------------------------
# GRPO-friendly batch wrapper (classifier-only reward)
# ---------------------------------------------------------------------------

def meter_reward(
    completions: Iterable[Any],
    prompts: Iterable[Any] | None = None,  # unused, kept for signature-compat
    poem_meter: Optional[Iterable[str] | str] = None,
    *,
    model_path: Path | str = _DEFAULT_BILSTM_MODEL_PATH,
    label_encoder_path: Path | str = _DEFAULT_LABEL_ENCODER_PATH,
    vocab_config_path: Path | str = _DEFAULT_VOCAB_CONFIG_PATH,
    trainer_state=None,
    **kwargs,
) -> list[float]:
    """
    Classifier-only reward for GRPO: one scalar per completion in [0, 1].

    - If poem_meter is None: uses max probability over meters (classifier confidence).
    - If poem_meter is provided: uses probability of that target meter label.
    """
    completions_list = list(completions or [])
    n = len(completions_list)

    if poem_meter is None:
        targets = [None] * n
    elif isinstance(poem_meter, str):
        targets = [poem_meter] * n
    else:
        targets = list(poem_meter)
        if len(targets) < n:
            targets += [None] * (n - len(targets))
        elif len(targets) > n:
            targets = targets[:n]

    rewards: list[float] = []

    for i, completion in enumerate(completions_list):
        text = _extract_text_from_completion(completion)
        if not isinstance(text, str):
            text = str(text or "")

        dist = classifier_meter_scores(
            text,
            model_path=model_path,
            label_encoder_path=label_encoder_path,
            vocab_config_path=vocab_config_path,
        )

        if not dist:
            # If classifier couldn't run, reward = 0 for this head.
            rewards.append(0.0)
            continue

        target = targets[i]
        if target is None:
            score = max(dist.values())
        else:
            score = float(dist.get(target, 0.0))

        # Clamp to [0, 1] just in case.
        rewards.append(max(0.0, min(1.0, score)))

    return rewards


__all__ = ["classifier_meter_scores", "meter_reward"]
