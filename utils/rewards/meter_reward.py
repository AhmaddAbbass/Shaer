"""
BiLSTM meter classifier utilities (GRPO-ready, classifier-only).

Exports:
- classifier_meter_scores(text) -> Dict[meter_label, prob]
- meter_reward(completions, poem_meter=None, ...) -> list[float]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import json
import os
import warnings
from contextlib import contextmanager

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

# Silence sklearn pickle version warning from the saved label encoder.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from .form_reward import _extract_text_from_completion

# ---------------------------------------------------------------------------
# Paths (anchored to Models/Ashaar_runtime)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Use lowercase paths to match repo layout.
ASHAAR_RUNTIME_DIR = PROJECT_ROOT / "models" / "Ashaar_runtime"
TRAINING_DIR = ASHAAR_RUNTIME_DIR / "bilstm_model" / "training"

_DEFAULT_BILSTM_MODEL_PATH = ASHAAR_RUNTIME_DIR / "bilstm_model" / "poem_meter_bilstm_v2.h5"
_DEFAULT_LABEL_ENCODER_PATH = TRAINING_DIR / "meter_label_encoder.joblib"
_DEFAULT_VOCAB_CONFIG_PATH = TRAINING_DIR / "bilstm_vocab_config.json"

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _lazy_import_tf():
    # Silence verbose TF logs and force CPU to avoid cuDNN mismatch with torch.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf

    # Ensure we stay on CPU (model is tiny) and keep dtype policy simple.
    try:
        # Keep TF on CPU; the BiLSTM is tiny and we want to avoid GPU/cuDNN issues.
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    return tf


@contextmanager
def _patched_policy():
    """
    Keras 3 + legacy saved model dtype can break when policy is a raw string.
    This forces get_policy to always return a Policy object.
    """
    try:
        from tensorflow.keras.mixed_precision import policy as mp_policy
        from tensorflow.keras import mixed_precision
    except Exception:
        yield
        return

    real_get_policy = mp_policy.get_policy

    def _safe_get_policy(identifier):
        try:
            return real_get_policy(identifier)
        except Exception:
            return mp_policy.Policy("float32")

    mp_policy.get_policy = _safe_get_policy  # type: ignore[attr-defined]
    try:
        yield
    finally:
        mp_policy.get_policy = real_get_policy  # type: ignore[attr-defined]


def _lazy_import_joblib():
    try:
        import joblib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load meter_label_encoder.joblib. Install it first."
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
    """Load (once) the BiLSTM keras model, label encoder, and char vocab."""
    global _BILSTM_MODEL, _LABEL_ENCODER, _STOI, _MAX_LEN

    if _BILSTM_MODEL is not None and _LABEL_ENCODER is not None and _STOI and _MAX_LEN > 0:
        return

    tf = _lazy_import_tf()
    joblib = _lazy_import_joblib()

    # --- Compatibility shim for old InputLayer with 'batch_shape' ----------
    class LegacyInputLayer(tf.keras.layers.InputLayer):  # type: ignore[attr-defined]
        def __init__(self, *args, **kwargs):
            # Old saved config passes batch_shape; new InputLayer doesn't accept it.
            kwargs.pop("batch_shape", None)
            super().__init__(*args, **kwargs)

    # Monkey-patch globally so that any deserialization that wants an InputLayer
    # actually gets our LegacyInputLayer instead.
    tf.keras.layers.InputLayer = LegacyInputLayer  # type: ignore[assignment]

    model_path = Path(model_path)
    label_encoder_path = Path(label_encoder_path)
    vocab_config_path = Path(vocab_config_path)

    custom_objects = {"InputLayer": LegacyInputLayer}
    # Handle dtype policy from newer Keras exports if present
    try:
        from tensorflow.keras.mixed_precision import policy as mp_policy  # type: ignore
        custom_objects["DTypePolicy"] = mp_policy.Policy  # type: ignore[attr-defined]
    except Exception:
        pass

    _BILSTM_MODEL = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False,  # allow loading legacy layers
    )
    _LABEL_ENCODER = joblib.load(str(label_encoder_path))

    with open(vocab_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _STOI = {str(k): int(v) for k, v in cfg["stoi"].items()}
    _MAX_LEN = int(cfg["max_len"])


def _encode_text_to_ints(text: str) -> np.ndarray:
    """Encode a single verse into shape (1, max_len) int32 using saved stoi."""
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
    Classifier-only reward for GRPO: one scalar per completion.

    - If poem_meter is None: uses max probability over meters (confidence).
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
            rewards.append(0.0)
            continue

        target = targets[i]
        score = max(dist.values()) if target is None else float(dist.get(target, 0.0))
        rewards.append(max(0.0, min(1.0, score)))

    return rewards


__all__ = ["classifier_meter_scores", "meter_reward"]
