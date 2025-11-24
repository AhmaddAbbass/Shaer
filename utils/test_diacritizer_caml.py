"""
Simple manual tests for the diacritization block.

Run with:
    python -m evaluation_engine.test_diacritizer
from the project root (/workspace/Shaer).
"""

from diacritizer import (
    add_diacritized_verse,
    diacritize_verse,
    strip_diacritics,
)


def test_plain_verse():
    sample = "ارى العمر في غير السرور مضيعا"
    print("=== Test 1: Plain undiacritized verse ===")
    print("Input      :", sample)
    print("Neutral    :", strip_diacritics(sample))
    print("Diacritized:", diacritize_verse(sample))
    print()


def test_already_diacritized():
    sample = "أَرَى العُمْرَ فِي غَيْرِ السُّرُورِ مُضَيَّعًا"
    print("=== Test 2: Already diacritized verse ===")
    print("Input      :", sample)
    print("Neutral    :", strip_diacritics(sample))
    print("Diacritized:", diacritize_verse(sample))
    print()


def test_mixed_verse():
    sample = "أَرى العمر في غير السُّرور مضيعا"
    print("=== Test 3: Mixed partial diacritization ===")
    print("Input      :", sample)
    print("Neutral    :", strip_diacritics(sample))
    print("Diacritized:", diacritize_verse(sample))
    print()


def test_empty_verse():
    sample = ""
    print("=== Test 4: Empty verse ===")
    print("Input      :", repr(sample))
    print("Neutral    :", repr(strip_diacritics(sample)))
    print("Diacritized:", repr(diacritize_verse(sample)))
    print()


def test_payload_roundtrip():
    payload = {
        "verse": "ارى العمر في غير السرور مضيعا",
        "meter_requested": "الطويل",
        "theme": "الحكمة",
        "description": "بيت يتحدث عن ضياع العمر دون سرور",
        "previous_verses": [],
        "poet_era": "العصر العباسي",
        "poet_name": "المتنبي",
    }

    print("=== Test 5: JSON payload roundtrip ===")
    result = add_diacritized_verse(payload)

    print("Original verse       :", payload["verse"])
    print("Diacritized in JSON  :", result["verse_diacritized"])

    # Sanity check: nothing else changed
    for key in ["meter_requested", "theme", "description", "poet_era", "poet_name"]:
        assert payload[key] == result[key], f"Field {key} was unexpectedly modified!"
    print("Metadata preserved ✅")
    print()


if __name__ == "__main__":
    test_plain_verse()
    test_already_diacritized()
    test_mixed_verse()
    test_empty_verse()
    test_payload_roundtrip()
