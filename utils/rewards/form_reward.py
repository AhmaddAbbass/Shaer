"""
Form-level reward for GRPO (Arabic bayt shape).

Goal:
- Work with the CURRENT SFT behaviour: the model outputs a single bayt
  as plain text, typically "صدر البيت ثم العجز" on one line.
- Optionally handle "[sep]" if it appears, but NEVER require it.

What we reward:
- Text is non-empty and mostly Arabic.
- We can reasonably split it into EXACTLY TWO hemistichs.
- No obvious extra junk (3+ clauses, multiple newlines).
- Hemistich lengths are reasonably balanced.
- Overall bayt length is in a realistic range (~3–8 words).

Returns:
- form_reward(completions, prompts, ...) -> list[float] in [0, 1]
"""

from __future__ import annotations

from typing import Any, Iterable, List, Tuple
import math
import re

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARABIC_LETTERS_RE = re.compile(r"[\u0600-\u06FF]+")


def _extract_text_from_completion(completion: Any) -> str:
    """
    Best-effort extraction of raw text from a GRPO completion.

    Handles:
    - plain strings
    - dicts from some generation APIs: {"text": "..."} or {"generated_text": "..."}
    - fall back to str(completion)
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        if "text" in completion and isinstance(completion["text"], str):
            return completion["text"]
        if "generated_text" in completion and isinstance(
            completion["generated_text"], str
        ):
            return completion["generated_text"]

    return str(completion or "")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def _split_lines(text: str) -> List[str]:
    text = _normalize_whitespace(text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines


def _split_hemistichs_from_line(line: str) -> Tuple[List[str], str]:
    """
    Try to split ONE line into hemistichs.

    Strategy (in order):
    1) If "[sep]" present -> split there (you may never use this; it's just a bonus).
    2) If Arabic separators like "،" or "؛" or "/" exist, try splitting on them.
    3) As a last resort, split by WORDS near the middle (heuristic).

    Returns:
      (segments_list, strategy_name)
    """
    line = line.strip()

    # 1) Explicit "[sep]"
    if "[sep]" in line:
        parts = [p.strip() for p in line.split("[sep]") if p.strip()]
        return parts, "[sep]"

    # 2) Known separators that *often* split hemistichs
    #    We try them one by one; if we get exactly 2 non-empty chunks, we keep that.
    sep_candidates = [" / ", " ، ", "،", " ؛ ", "؛", " - "]
    for sep in sep_candidates:
        if sep in line:
            parts = [p.strip() for p in line.split(sep) if p.strip()]
            if len(parts) == 2:
                return parts, f"sep:{sep}"
            elif len(parts) > 2:
                # still return them, but caller will see >2 and penalize
                return parts, f"sep:{sep}(>2)"

    # 3) WORD-based heuristic: split words near the middle.
    words = line.split()
    if len(words) < 2:
        # too short to sensibly have 2 hemistichs
        return [line], "short"

    mid = len(words) // 2
    sadr_words = words[:mid]
    ajz_words = words[mid:]

    first = " ".join(sadr_words).strip()
    second = " ".join(ajz_words).strip()
    parts = [p for p in [first, second] if p]

    return parts, "word-mid"


def _analyze_bayt_shape(text: str) -> dict:
    """
    Analyze structural shape of a generated bayt.

    Returns a dict with:
      - n_lines
      - n_segments (after splitting main line into chunks)
      - has_two_hemistichs: bool
      - extra_segments: int (segments beyond 2)
      - len1, len2  (char lengths of two hemistichs)
      - balance_score in [0, 1] (1 = perfectly balanced)
      - arabic_density in [0, 1]
      - word_count (total words in the bayt)
    """
    raw = text or ""
    norm = _normalize_whitespace(raw)

    # Lines
    lines = _split_lines(norm)
    n_lines = len(lines)

    tokens = norm.split()
    word_count = len(tokens)

    if not lines:
        return {
            "n_lines": 0,
            "n_segments": 0,
            "has_two_hemistichs": False,
            "extra_segments": 0,
            "len1": 0,
            "len2": 0,
            "balance_score": 0.0,
            "arabic_density": 0.0,
            "word_count": 0,
        }

    # For poetry in this setup, we *want* essentially ONE line.
    # If there are multiple lines, merge them or treat as extra structure.
    main_line = " ".join(lines)
    segments, strategy = _split_hemistichs_from_line(main_line)
    n_segments = len(segments)

    has_two = n_segments >= 2
    extra_segments = max(0, n_segments - 2)

    # Length balance
    len1 = len(segments[0]) if n_segments >= 1 else 0
    len2 = len(segments[1]) if n_segments >= 2 else 0

    if len1 > 0 and len2 > 0:
        balance_score = 1.0 - abs(len1 - len2) / max(len1, len2)
    else:
        balance_score = 0.0

    balance_score = max(0.0, min(1.0, balance_score))

    # Arabic density: fraction of characters that look Arabic letters
    total_chars = len(norm)
    arabic_chars = sum(len(m.group(0)) for m in ARABIC_LETTERS_RE.finditer(norm))
    arabic_density = arabic_chars / total_chars if total_chars > 0 else 0.0
    arabic_density = max(0.0, min(1.0, arabic_density))

    return {
        "n_lines": n_lines,
        "n_segments": n_segments,
        "has_two_hemistichs": bool(has_two),
        "extra_segments": int(extra_segments),
        "len1": len1,
        "len2": len2,
        "balance_score": float(balance_score),
        "arabic_density": float(arabic_density),
        "word_count": int(word_count),
    }


def _length_score_from_word_count(wc: int) -> float:
    """
    Map bayt word count to a soft score in [0, 1].

    Based on dataset stats:
    - typical bayt word-length: mean ≈ 4.9, p90 ≈ 6
    -> we treat 3–8 as the "sweet spot".
    """
    if wc <= 0:
        return 0.0
    if 3 <= wc <= 8:
        return 1.0
    if wc == 2 or 9 <= wc <= 10:
        return 0.7
    if wc == 1 or 11 <= wc <= 12:
        return 0.4
    # way too short or too long → strongly penalize
    return 0.2


def _score_form(text: str) -> float:
    """
    Turn bayt shape stats into a scalar form score in [0, 1].
    """
    stats = _analyze_bayt_shape(text)

    # If no Arabic at all, it's junk for our purposes.
    if stats["arabic_density"] < 0.3:
        return 0.0

    # Base score from two-hemistich structure
    if not stats["has_two_hemistichs"]:
        base = 0.2  # at least it might be some Arabic text
    else:
        # Good if exactly 2 segments, penalize if there are extras.
        if stats["n_segments"] == 2:
            base = 0.8
        else:
            # e.g. 3+ clauses → still something, but we dislike it
            base = 0.5 / (1 + stats["extra_segments"])

    # Penalize for many lines (we want mostly single-line bayt)
    if stats["n_lines"] > 2:
        base *= 0.7
    elif stats["n_lines"] == 2:
        base *= 0.85

    # Mix in balance (how close in length صدر vs. عجز)
    balance = stats["balance_score"]
    arabic_density = stats["arabic_density"]
    length_score = _length_score_from_word_count(stats["word_count"])

    # Weighted combination (weights sum to 1.0)
    score = (
        0.45 * base
        + 0.25 * balance
        + 0.15 * arabic_density
        + 0.15 * length_score
    )

    # Clip
    score = max(0.0, min(1.0, score))
    return float(score)


# ---------------------------------------------------------------------------
# Public GRPO reward API
# ---------------------------------------------------------------------------


def form_reward(
    completions: Iterable[Any],
    prompts: Iterable[Any] | None = None,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    """
    Form reward for GRPO.

    Input:
      - completions: iterable of model outputs (strings or dicts)
      - prompts: (unused but kept for signature compatibility)
      - trainer_state: (unused, for future logging if needed)

    Output:
      - list[float] in [0, 1], one per completion
    """
    scores: List[float] = []

    for completion in completions or []:
        text = _extract_text_from_completion(completion)
        score = _score_form(text)
        scores.append(score)

    return scores


__all__ = ["form_reward", "_extract_text_from_completion"]
