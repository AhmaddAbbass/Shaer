# utils/rewards/meaning_reward.py

import os
import re
from typing import List

import requests

from .form_reward import _extract_text_from_completion


def _normalize_arabic(s):
    if not isinstance(s, str):
        return ""
    t = s.replace("ـ", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def meaning_reward(completions, prompts, poem_description=None, trainer_state=None, **kwargs):
    """
    Meaning / semantic alignment reward.

    Now delegates to a remote Yehia judge server (vLLM on GPU 1)
    exposed via HTTP, instead of instantiating vLLM locally.

    For each sample:
      - description = poem_description[i]
      - verse       = extracted from completions[i]
      - Server returns a float in [0, 10].

    Returns:
      - list[float] in [0, 10]
    """
    judge_url = os.environ.get("MEANING_JUDGE_URL", "http://localhost:8009/score_batch")

    n = len(completions)
    if poem_description is None:
        descs = [None] * n
    else:
        descs = list(poem_description)
        if len(descs) < n:
            descs = descs + [None] * (n - len(descs))
        elif len(descs) > n:
            descs = descs[:n]

    items: List[dict] = []

    for i in range(n):
        verse_raw = _extract_text_from_completion(completions[i])
        verse = _normalize_arabic(verse_raw)

        desc = descs[i]
        desc_norm = _normalize_arabic(desc) if desc is not None else ""

        items.append(
            {
                "verse": verse,
                "description": desc_norm,
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

        cleaned = []
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
