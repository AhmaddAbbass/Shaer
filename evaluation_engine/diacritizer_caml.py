"""
Diacritization block for the Arabic poetry pipeline.

Responsibilities:
- Normalize any verse by stripping all existing diacritics.
- Call the CAMeL `camel_diac` CLI to re-diacritize the verse.
- Expose a clean JSON-in / JSON-out API for the rest of the pipeline.
"""

import subprocess
from typing import Dict, Any

from camel_tools.utils.dediac import dediac_ar


# ─────────────────────────────────────────────
# 1. Normalization utilities
# ─────────────────────────────────────────────

def strip_diacritics(text: str) -> str:
    """
    Remove all Arabic diacritics from the text using CAMeL Tools.

    This turns:
      - fully diacritized
      - partially diacritized
      - mixed text

    into a "neutral" undiacritized Arabic string.
    """
    if not text:
        return ""
    return dediac_ar(text)


# ─────────────────────────────────────────────
# 2. Thin wrapper around `camel_diac` CLI
# ─────────────────────────────────────────────

def _run_camel_diac(text: str, db: str = "calima-msa-r13") -> str:
    """
    Call the external `camel_diac` command to diacritize `text`.

    - Sends `text` via stdin.
    - Reads diacritized text from stdout.
    - Uses the specified morphology database (default: calima-msa-r13).

    On any failure, it returns the *input* text unchanged so that the
    rest of the pipeline does not crash.
    """
    text = (text or "").strip()
    if not text:
        return text

    try:
        # `camel_diac` reads input from stdin and writes diacritized text to stdout.
        proc = subprocess.run(
            ["camel_diac", "-d", db],
            input=text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
    except FileNotFoundError:
        # CLI not on PATH (e.g. venv not active)
        print("[diacritizer] ERROR: `camel_diac` command not found. "
              "Is the virtualenv active and camel-tools installed?")
        return text
    except Exception as e:
        print("[diacritizer] ERROR: Unexpected exception while calling camel_diac:", repr(e))
        return text

    if proc.returncode != 0:
        print("[diacritizer] ERROR: camel_diac exited with code", proc.returncode)
        if proc.stderr:
            print("[diacritizer] STDERR:", proc.stderr.strip())
        return text

    # For a single verse we expect one line (or a few) back.
    # We strip outer whitespace but leave inner whitespace and punctuation as-is.
    return proc.stdout.strip()


# ─────────────────────────────────────────────
# 3. Public API
# ─────────────────────────────────────────────

def diacritize_verse(verse: str) -> str:
    """
    Full diacritization pipeline for a single verse:

      1. Strip ALL existing diacritics (neutralization).
      2. Run `camel_diac` on the cleaned verse to obtain a fresh,
         consistent diacritization.

    This way we avoid mixing partial old tashkeel with new ones.
    """
    verse = (verse or "").strip()
    if not verse:
        return verse

    # Step 1: neutralize
    neutral = strip_diacritics(verse)
    if not neutral:
        return neutral

    # Step 2: diacritize with CAMeL
    diacritized = _run_camel_diac(neutral)
    return diacritized


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
          "poet_name": "..."
        }

    and return a NEW dict with an extra key:

        "verse_diacritized": "<fully diacritized verse>"

    The original payload is not modified in-place, so this can be safely
    used in a functional pipeline.
    """
    verse = payload.get("verse", "") or ""
    dia_verse = diacritize_verse(verse)

    new_payload = payload.copy()
    new_payload["verse_diacritized"] = dia_verse

    # Optional: we could also add the neutral version if you want later:
    # new_payload["verse_neutral"] = strip_diacritics(verse)

    return new_payload
