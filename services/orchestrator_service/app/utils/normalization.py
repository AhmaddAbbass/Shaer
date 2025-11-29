from __future__ import annotations

from typing import Iterable, List


def normalize_single_bayt(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def normalize_previous_verses(verses: Iterable[str]) -> List[str]:
    return [line.strip() for line in verses if line and line.strip()]
