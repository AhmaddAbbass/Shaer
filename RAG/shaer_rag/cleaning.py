from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

from rapidfuzz import fuzz

from .settings import Settings
from .utils import collapse_spaces, domain_from_url, strip_diacritics

# Common placeholder / "I don't know" Arabic patterns we want to exclude.
PLACEHOLDER_PATTERNS = [
    "لا اعرف",
    "لا أعرف",
    "لا ادري",
    "لا أدري",
    "غير معروف",
    "غير متوفر",
    "غير متاح",
]

# Boilerplate prefixes to drop before embedding.
BOILERPLATE_PREFIXES = [
    "هذه القصيدة",  # "this poem ..."
    "تتحدث القصيدة",  # "the poem talks about ..."
    "القصيدة تتحدث",  # same phrasing flipped
    "قصيدة تتحدث",    # generic starter
]

ARABIC_LETTER_RE = re.compile(r"[ء-ي]")
NON_LETTER_RE = re.compile(r"[^ء-ي0-9\s]")


@dataclass
class CleanedPoem:
    poem_id: int
    poem_title: Optional[str]
    poem_meter: str
    poem_theme: Optional[str]
    poem_url: Optional[str]
    poet_name: Optional[str]
    poet_url: Optional[str]
    poet_description: Optional[str]
    poet_era: Optional[str]
    poet_location: Optional[str]
    poem_language_type: Optional[str]
    poem_verses: List[str]
    num_verses: int
    description_raw: str
    description_clean: Optional[str]
    has_bad_description: bool
    needs_resummarization: bool
    issues: List[str]
    source: str
    verse_preview: str
    poet_key: Optional[str]

    def as_dict(self) -> dict:
        return asdict(self)


def arabic_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    arabic_letters = [ch for ch in letters if ARABIC_LETTER_RE.match(ch)]
    return len(arabic_letters) / len(letters)


def normalize_for_similarity(text: str) -> str:
    """Normalize text to compare poem description vs poem content."""
    text = text.lower()
    text = strip_diacritics(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = NON_LETTER_RE.sub(" ", text)
    return collapse_spaces(text)


def is_placeholder(desc: str) -> bool:
    lowered = desc.lower().strip()
    return any(p in lowered for p in PLACEHOLDER_PATTERNS)


def drop_boilerplate_prefix(desc: str) -> str:
    lowered = desc.lower().lstrip()
    for prefix in BOILERPLATE_PREFIXES:
        if lowered.startswith(prefix):
            trimmed = desc[len(prefix):].lstrip(" -،,:")
            return trimmed
    return desc


def description_matches_poem(
    description: str, poem_verses: Iterable[str], *, similarity_threshold: float
) -> bool:
    joined_poem = " ".join(poem_verses)
    if not joined_poem.strip():
        return False
    desc_norm = normalize_for_similarity(description)
    poem_norm = normalize_for_similarity(joined_poem)
    if not desc_norm or not poem_norm:
        return False

    # Limit comparison length to avoid pathological slowdowns on very long poems.
    max_len = max(len(desc_norm) * 4, 1200)
    poem_norm = poem_norm[:max_len]
    score = fuzz.ratio(desc_norm, poem_norm) / 100.0
    return score >= similarity_threshold


def build_poet_key(poet_name: Optional[str], poet_era: Optional[str], poet_url: Optional[str]) -> Optional[str]:
    if poet_url:
        return poet_url.strip()
    if not poet_name:
        return None
    era_suffix = f"|{poet_era.strip()}" if poet_era else ""
    return f"{poet_name.strip()}{era_suffix}"


def clean_description(text: str, settings: Settings) -> tuple[Optional[str], List[str], bool]:
    issues: List[str] = []
    if text is None:
        return None, ["missing"], False

    cleaned = collapse_spaces(text)
    cleaned = drop_boilerplate_prefix(cleaned)

    if not cleaned:
        return None, ["missing"], False

    if is_placeholder(cleaned):
        issues.append("placeholder")

    ratio = arabic_ratio(cleaned)
    if ratio < settings.min_arabic_ratio:
        issues.append("non_arabic")

    needs_resummarization = len(cleaned) > settings.description_max_chars

    return cleaned if cleaned else None, issues, needs_resummarization


def clean_poem_row(row: dict, settings: Settings) -> CleanedPoem:
    raw_desc = row.get("poem_description") or ""
    verses = row.get("poem_verses") or []
    description_clean, issues, needs_resummarization = clean_description(raw_desc, settings)

    if description_clean and description_matches_poem(
        description_clean, verses, similarity_threshold=settings.verse_similarity_threshold
    ):
        issues.append("description_matches_poem")

    has_bad_description = bool(issues) or not description_clean

    poet_key = build_poet_key(
        row.get("poet_name"), row.get("poet_era"), row.get("poet_url")
    )

    verse_preview = ""
    if verses:
        preview_lines = [v for v in verses[:3] if v]
        verse_preview = " | ".join(preview_lines)

    return CleanedPoem(
        poem_id=int(row["poem_id"]),
        poem_title=row.get("poem_title"),
        poem_meter=row.get("poem_meter") or "unknown",
        poem_theme=row.get("poem_theme"),
        poem_url=row.get("poem_url"),
        poet_name=row.get("poet_name"),
        poet_url=row.get("poet_url"),
        poet_description=row.get("poet_description"),
        poet_era=row.get("poet_era"),
        poet_location=row.get("poet_location"),
        poem_language_type=row.get("poem_language_type"),
        poem_verses=list(verses),
        num_verses=int(row.get("num_verses") or len(verses)),
        description_raw=raw_desc,
        description_clean=description_clean,
        has_bad_description=has_bad_description,
        needs_resummarization=needs_resummarization,
        issues=issues,
        source=domain_from_url(row.get("poem_url")),
        verse_preview=verse_preview,
        poet_key=poet_key,
    )

