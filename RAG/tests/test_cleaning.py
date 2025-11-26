from __future__ import annotations

from shaer_rag.cleaning import clean_poem_row
from shaer_rag.settings import Settings


def _base_row() -> dict:
    return {
        "poem_id": 1,
        "poem_title": "قصيدة تجريبية",
        "poem_meter": "البحر الطويل",
        "poem_theme": "الغزل",
        "poem_url": "https://example.com/poem/1",
        "poet_name": "شاعر تجريبي",
        "poet_url": "https://example.com/poet",
        "poet_description": "شاعر من العصر الحديث.",
        "poet_era": "العصر الحديث",
        "poet_location": "بيروت",
        "poem_language_type": "arabic",
        "poem_verses": ["أحبك حتى تعب الكلام", "وأشتاق شوق الغيم للغمام"],
        "num_verses": 2,
    }


def test_placeholder_marked_bad():
    row = _base_row()
    row["poem_description"] = "لا أعرف"
    poem = clean_poem_row(row, Settings())
    assert poem.has_bad_description
    assert "placeholder" in poem.issues


def test_non_arabic_marked_bad():
    row = _base_row()
    row["poem_description"] = "This is mostly english description without Arabic."
    poem = clean_poem_row(row, Settings(min_arabic_ratio=0.5))
    assert poem.has_bad_description
    assert "non_arabic" in poem.issues


def test_duplicate_description_detected():
    row = _base_row()
    row["poem_description"] = "أحبك حتى تعب الكلام وأشتاق شوق الغيم للغمام"
    poem = clean_poem_row(row, Settings(verse_similarity_threshold=0.8))
    assert poem.has_bad_description
    assert "description_matches_poem" in poem.issues


def test_valid_description_survives():
    row = _base_row()
    row["poem_description"] = "قصيدة قصيرة تصف شوق شاعر لرفيقته بلغة عذبة."
    poem = clean_poem_row(row, Settings())
    assert not poem.has_bad_description
    assert poem.description_clean
