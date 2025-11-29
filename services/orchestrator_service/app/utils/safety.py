from __future__ import annotations

import re
from fastapi import HTTPException

from ..settings import Settings

_ARABIC_RE = re.compile(r"[ء-ي]")
_POETRY_KEYWORDS = (
    "شعر",
    "قصيدة",
    "بيت",
    "أبيات",
    "الشاعر",
    "بحر",
    "قافية",
    "قصائد",
)


def guard_poetry_topic(text: str, settings: Settings) -> None:
    """
    Soft guardrail: if ALLOW_ONLY_POETRY_TOPICS is enabled, require either
    some Arabic letters or a poetry keyword. Otherwise, raise 400.
    """
    if not settings.allow_only_poetry_topics:
        return
    lowered = (text or "").lower()
    has_arabic = bool(_ARABIC_RE.search(lowered))
    has_keyword = any(kw in lowered for kw in _POETRY_KEYWORDS)
    if not (has_arabic or has_keyword):
        raise HTTPException(
            status_code=400,
            detail="الطلب خارج نطاق الشعر العربي. يرجى تحديد طلب يتعلق بالقصائد أو الأبيات أو الشعراء.",
        )
