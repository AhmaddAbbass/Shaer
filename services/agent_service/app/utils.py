from __future__ import annotations

from typing import Iterable, List


def extract_candidate_verses(text: str) -> List[str]:
    """Extract bayt-like lines from a raw user message.

    We treat any non-empty line as a candidate verse and strip punctuation artifacts.
    """

    verses: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Skip obvious prose sentences to reduce noise.
        if len(line) < 15:
            continue
        if len(line.split()) < 3:
            continue
        verses.append(line)
    return verses


def format_poem_snippet(verses: Iterable[str], max_lines: int = 4) -> str:
    selected = []
    for verse in verses:
        if len(selected) >= max_lines:
            break
        if verse.strip():
            selected.append(verse.strip())
    return "\n".join(selected)
