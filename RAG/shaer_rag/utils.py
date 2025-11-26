from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TypeVar
from urllib.parse import urlparse

T = TypeVar("T")


def batched(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    """Yield fixed-size batches from a sequence."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    for i in range(0, len(seq), size):
        yield list(itertools.islice(seq, i, i + size))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def domain_from_url(url: str | None) -> str:
    if not url:
        return "unknown"
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    host = host.lower()
    host = host.lstrip("www.")
    return host or "unknown"


ARABIC_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics to make fuzzy matching more forgiving."""
    return ARABIC_DIACRITICS_RE.sub("", text)


def collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

