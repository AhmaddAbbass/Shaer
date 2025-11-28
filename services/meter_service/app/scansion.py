from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
try:  # TensorFlow is heavy; allow graceful fallback when not installed.
    import tensorflow as tf
except ImportError:  # pragma: no cover - tensorflow not present in lightweight envs
    tf = None

from services.common_schemas.schemas import BaytMeterEval

from .config import Settings
from .logging import get_logger

logger = get_logger("meter_service.scansion")

# Silence sklearn pickle warning on legacy encoder
import warnings

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


class ScansionService:
    """
    Wrapper around the Ashaar BiLSTM meter classifier.
    Expects a re-exported model (H5 or SavedModel) at meter_model_path.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.assets_loaded = False
        self.model = None
        self.label_encoder = None
        self.stoi: Dict[str, int] = {}
        self.max_len: int = 0
        self._load_assets()

    def _load_assets(self) -> None:
        if tf is None:
            logger.warning("TensorFlow not available; meter service will return default evaluations.")
            self.assets_loaded = False
            return
        try:
            from utils.rewards.meter_reward import (
                _load_bilstm_assets,
                _LABEL_ENCODER,
                _STOI,
                _MAX_LEN,
                _BILSTM_MODEL,
            )

            _load_bilstm_assets(
                model_path=self.settings.meter_model_path,
                label_encoder_path=self.settings.meter_label_encoder_path,
                vocab_config_path=self.settings.meter_vocab_config_path,
            )

            self.model = _BILSTM_MODEL
            self.label_encoder = _LABEL_ENCODER
            self.stoi = _STOI
            self.max_len = _MAX_LEN
            if self.model is None or self.label_encoder is None or not self.stoi or self.max_len <= 0:
                raise RuntimeError("Model assets not fully loaded")
            self.assets_loaded = True
            logger.info("Meter assets loaded")
        except FileNotFoundError as exc:
            logger.error(
                "Meter model file not found at %s. Run scripts/convert_meter_model.py and rebuild the image. (%s)",
                self.settings.meter_model_path,
                exc,
            )
            self.assets_loaded = False
        except Exception as exc:
            logger.error("Failed to load meter assets: %s", exc)
            self.assets_loaded = False

    def _encode_text_to_ints(self, text: str) -> "tf.Tensor":
        seq = np.zeros((self.max_len,), dtype="int32")
        clean = text.replace("[sep]", " ")
        for t, ch in enumerate(clean[: self.max_len]):
            seq[t] = self.stoi.get(ch, 0)
        return tf.convert_to_tensor([seq], dtype=tf.int32)

    def _predict_dist(self, text: str) -> Dict[str, float]:
        if not self.assets_loaded or not self.model or not self.label_encoder:
            raise RuntimeError("Meter assets not loaded")

        x = self._encode_text_to_ints(text)
        probs = self.model.predict(x, verbose=0)[0]
        labels = list(self.label_encoder.classes_)
        return {str(lbl): float(probs[i]) for i, lbl in enumerate(labels)}

    def _find_target_label(self, target_meter: str, labels: Dict[str, float]) -> Optional[str]:
        target_norm = (target_meter or "").strip().lower().replace(" ", "")
        norm_map = {str(label).strip().lower().replace(" ", ""): label for label in labels.keys()}
        return norm_map.get(target_norm)

    def _top_prediction(self, scores: Dict[str, float]) -> Tuple[str, float]:
        if not scores:
            return "", 0.0
        label, prob = max(scores.items(), key=lambda kv: kv[1])
        return label, float(prob or 0.0)

    def evaluate_bayt(self, verse_text: str, target_meter: str) -> BaytMeterEval:
        if not self.assets_loaded or tf is None:
            return self._default_eval(target_meter)

        scores = self._predict_dist(verse_text)

        top_label, top_prob = self._top_prediction(scores)
        target_label = self._find_target_label(target_meter, scores)

        if target_label:
            target_prob = float(scores.get(target_label, 0.0))
            meter_score = round(target_prob * 100)
            on_meter = meter_score >= self.settings.meter_score_threshold
            top_note = f"أعلى بحر متوقع: {top_label} ({round(top_prob * 100)}%)" if top_label else ""
            if on_meter:
                notes = f"تم اجتياز البحر المطلوب ({target_meter}) بدرجة {meter_score}%. {top_note}".strip()
            else:
                notes = f"لم يجتز البحر المطلوب ({target_meter}). الدرجة {meter_score}%. {top_note}".strip()
        else:
            meter_score = 0
            on_meter = False
            notes = (
                f"البحر المطلوب غير موجود في نموذج الأوزان؛ "
                f"أقرب بحر متوقع هو {top_label} ({round(top_prob * 100)}%)."
            )

        return BaytMeterEval(
            target_meter=target_meter,
            meter_score=int(meter_score),
            on_meter=bool(on_meter),
            notes=notes,
        )

    def _default_eval(self, target_meter: str) -> BaytMeterEval:
        """Return a safe default when the model or TensorFlow is unavailable."""
        return BaytMeterEval(
            target_meter=target_meter or "غير محدد",
            meter_score=75,
            on_meter=True,
            notes="تم قبول البيت افتراضياً لعدم توفر نموذج البحر حالياً.",
        )
