"""
Diacritization block for the Arabic poetry pipeline using Fine-Tashkeel (ByT5).

Responsibilities:
- Normalize any verse by stripping all existing diacritics.
- Call the Fine-Tashkeel model to re-diacritize the verse.
- Expose a clean JSON-in / JSON-out API for the rest of the pipeline.
"""

from typing import Dict, Any

from camel_tools.utils.dediac import dediac_ar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────
# 0. Load model/tokenizer ONCE at import-time
# ─────────────────────────────────────────────

# Public Fine-Tashkeel model on Hugging Face
MODEL_NAME = "basharalrfooh/Fine-Tashkeel"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


# ─────────────────────────────────────────────
# 1. Normalization utilities
# ─────────────────────────────────────────────

def strip_diacritics(text: str) -> str:
    """
    Remove all Arabic diacritics from the text using CAMeL Tools.

    Converts any partially or fully diacritized Arabic into clean undiacritized form.
    """
    if not text:
        return ""
    return dediac_ar(text)


# ─────────────────────────────────────────────
# 2. Diacritization with Fine-Tashkeel
# ─────────────────────────────────────────────

def _run_fine_tashkeel(text: str) -> str:
    """
    Diacritize the input Arabic text using Fine-Tashkeel (ByT5 model).
    """
    if not text.strip():
        return text

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=128)
    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded


# ─────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────

def diacritize_verse(verse: str) -> str:
    """
    Full diacritization pipeline for a single verse:

      1. Strip ALL existing diacritics (neutralization).
      2. Run Fine-Tashkeel on the cleaned verse to obtain a fresh,
         consistent diacritization.
    """
    verse = (verse or "").strip()
    if not verse:
        return verse

    # Step 1: neutralize
    neutral = strip_diacritics(verse)
    if not neutral:
        return neutral

    # Step 2: diacritize with Fine-Tashkeel
    return _run_fine_tashkeel(neutral)


def add_diacritized_verse(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a generation JSON like:

        {
          "verse": "...",
          "meter_requested": "...",
          "theme": "...",
          "description": "...",
          "previous_verses": [...],
          "poet_era": "...",
          "poet_name": "...",
          ...
        }

    and return a NEW dict with an extra key:

        "verse_diacritized": "<fully diacritized verse>"
    """
    verse = payload.get("verse", "") or ""
    dia_verse = diacritize_verse(verse)

    new_payload = payload.copy()
    new_payload["verse_diacritized"] = dia_verse

    return new_payload
