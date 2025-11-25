# utils/rewards/form_reward.py

import re

WHITESPACE_RE = re.compile(r"\s+")


def _extract_text_from_completion(completion):
    """
    Normalize whatever GRPO gives us into a plain text string.
    Can be:
      - str
      - {"content": "..."} or {"content": [{"text": "..."}]}
      - list of {"role": ..., "content": "..."} (chat-style)
    """
    if completion is None:
        return ""

    # plain string
    if isinstance(completion, str):
        return completion

    # dict: try common shapes
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # vLLM / chat-style: [{"type":"text","text":"..."}] or similar
            pieces = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text") or part.get("content") or ""
                    if isinstance(txt, str):
                        pieces.append(txt)
                elif isinstance(part, str):
                    pieces.append(part)
            return " ".join(pieces)

        # whole dict is maybe a message
        txt = completion.get("text") or completion.get("value") or ""
        if isinstance(txt, str):
            return txt
        return str(completion)

    # list of messages / segments
    if isinstance(completion, list):
        pieces = []
        for item in completion:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                txt = item.get("content") or item.get("text") or ""
                if isinstance(txt, str):
                    pieces.append(txt)
        if pieces:
            return " ".join(pieces)

    # fallback
    return str(completion)


def _normalize_text(text):
    if not isinstance(text, str):
        return ""
    t = text.replace("ـ", "")          # strip tatweel if present
    t = WHITESPACE_RE.sub(" ", t)
    return t.strip()


def form_reward(completions, prompts, **kwargs):
    """
    Reward ≈ 1 if the answer looks like *exactly one bayt* (one logical line).
    For now we keep it super simple:
      - empty → 0
      - multiple non-empty lines → 0
      - single non-empty line → 1
    """
    rewards = []

    for completion in completions:
        raw = _extract_text_from_completion(completion)
        text = _normalize_text(raw)

        if not text:
            rewards.append(0.0)
            continue

        # split by lines and filter empties
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        if len(lines) == 0:
            rewards.append(0.0)
        elif len(lines) == 1:
            rewards.append(1.0)
        else:
            # more than one line → probably multiple abyaat / explanation
            rewards.append(0.0)

    return rewards
