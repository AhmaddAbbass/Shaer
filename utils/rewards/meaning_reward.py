# utils/rewards/meaning_reward.py

import os
import re
from typing import List, Any, Iterable, Optional

import requests

from .form_reward import _extract_text_from_completion


def _normalize_arabic(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    t = s.replace("ـ", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def meaning_reward(
    completions: Iterable[Any],
    prompts: Optional[Iterable[Any]] = None,
    poem_description: Optional[Iterable[Any]] = None,
    previous_verses_clean: Optional[Iterable[Any]] = None,
    trainer_state=None,
    **kwargs,
) -> List[float]:
    """
    Meaning / semantic alignment reward.

    Delegates to a remote Yehia judge server (vLLM on GPU 1)
    exposed via HTTP.

    For each sample:
      - description        = poem_description[i]
      - previous_verses    = previous_verses_clean[i] (list[str] or str, optional)
      - verse              = extracted from completions[i]
      - Server returns a float in [0, 10].

    Returns:
      - list[float] in [0, 10]
    """
    judge_url = os.environ.get("MEANING_JUDGE_URL", "http://localhost:8009/score_batch")

    completions = list(completions)
    n = len(completions)

    # --- normalize descriptions length ---
    if poem_description is None:
        descs: List[Optional[str]] = [None] * n
    else:
        tmp = list(poem_description)
        if len(tmp) < n:
            tmp = tmp + [None] * (n - len(tmp))
        elif len(tmp) > n:
            tmp = tmp[:n]
        descs = tmp

    # --- normalize previous verses length ---
    if previous_verses_clean is None:
        prev_all: List[Optional[Any]] = [None] * n
    else:
        tmp_prev = list(previous_verses_clean)
        if len(tmp_prev) < n:
            tmp_prev = tmp_prev + [None] * (n - len(tmp_prev))
        elif len(tmp_prev) > n:
            tmp_prev = tmp_prev[:n]
        prev_all = tmp_prev

    items: List[dict] = []

    for i in range(n):
        verse_raw = _extract_text_from_completion(completions[i])
        verse = _normalize_arabic(verse_raw)

        desc = descs[i]
        desc_norm = _normalize_arabic(desc) if desc is not None else ""

        prev = prev_all[i]
        if isinstance(prev, list):
            prev_text = " | ".join(_normalize_arabic(p) for p in prev if p)
        else:
            prev_text = _normalize_arabic(prev) if prev else ""

        items.append(
            {
                "verse": verse,
                "description": desc_norm,
                "previous_verses": prev_text,
            }
        )

    payload = {"items": items}

    try:
        resp = requests.post(judge_url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        scores = data.get("scores", [])
        if not isinstance(scores, list):
            raise ValueError("Invalid judge response format: 'scores' is not a list")

        # Ensure correct length and range
        if len(scores) != n:
            print(
                f"[meaning_reward] Judge returned {len(scores)} scores for {n} items. "
                "Falling back to zeros."
            )
            return [0.0] * n

        cleaned: List[float] = []
        for s in scores:
            try:
                v = float(s)
            except Exception:
                v = 0.0
            v = max(0.0, min(10.0, v))
            cleaned.append(v)

        return cleaned

    except Exception as e:
        print("[meaning_reward] Remote judge error:", e)
        return [0.0] * n
