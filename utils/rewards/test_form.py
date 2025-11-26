# utils/rewards/test_from_reward.py

"""
Simple sanity tests for form_reward.

Run from project root:

    python utils/rewards/test_from_reward.py
"""

from pathlib import Path
import sys

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rewards.form_reward import form_reward  # noqa: E402


def run_tests():
    cases = {
        # Clean bayt with comma separator
        "good_bayt_sep_comma": "سافِرْ تَجِدْ عِوَضًا عَمَّنْ تُفَارِقُهُ، فَالنَّاسُ فِي الأَرْضِ كَالأَعْوَادِ وَالرُّتَبِ",
        # Clean bayt with semicolon separator
        "good_bayt_sep_semicolon": "وَمَا نَيْلُ المَطَالِبِ بِالتَمَنِّي؛ وَلَكِن تُؤْخَذُ الدُّنْيَا غِلابَا",
        # Clean bayt with ONLY spaces (no comma/semicolon)
        "good_bayt_space_only": "طَافَ الصَّبَاحُ عَلَى الأَوْطَانِ مُبْتَهِجًا فَاهْدَأْ فُؤَادِي فَقَدْ أَشْرَقْتَ فِي طَلَعِهِ",
        # Latin junk (should be ~0)
        "latin_junk": "hello my friend this is just some latin junk text without arabic letters at all",
        # Very long mushy Arabic line (too many words)
        "very_long_mush": (
            "هَذَا نَصٌّ عَرَبِيٌّ طَوِيلٌ جِدًّا يَسْتَمِرُّ فِي السُّطُورِ "
            "وَيُكْثِرُ مِنَ العِبَارَاتِ حَتَّى يَخْرُجَ عَنْ شَكْلِ البَيْتِ "
            "وَيُشْبِهُ الفَقْرَةَ النَّثْرِيَّةَ أَكْثَرَ مِمَّا يُشْبِهُ الشِّعْرَ"
        ),
        # Multiline text (should be penalized)
        "multiline": "هَذَا بَيْتٌ يَبْدَأُ هُنَا\nثُمَّ يَسْتَمِرُّ فِي سَطْرٍ آخَرَ بِصُورَةٍ غَيْرِ مَأْلُوفَةٍ",
        # Single very short Arabic word
        "single_very_short": "سلام",
        # Short phrase (Arabic, but not obviously bayt-shaped)
        "short_arabic_phrase": "هذا كلام بسيط جدا",
    }

    scores = {}
    print("=== form_reward test scores ===")
    for name, text in cases.items():
        score = form_reward([text])[0]
        scores[name] = score
        print(f"{name:25s} -> {score:.3f}")

    # --- basic structural expectations ---

    # Good bayts should get reasonably high scores.
    assert scores["good_bayt_sep_comma"] > 0.7, "Comma-separated bayt should score high."
    assert scores["good_bayt_sep_semicolon"] > 0.7, "Semicolon-separated bayt should score high."
    assert scores["good_bayt_space_only"] > 0.7, "Space-only bayt should still score high."

    # Latin junk should be essentially zero.
    assert scores["latin_junk"] < 0.05, "Pure latin text should get ~0 form score."

    # Very long mush should be lower than a clean bayt.
    assert scores["very_long_mush"] < scores["good_bayt_sep_comma"], \
        "Long mush line must score lower than a clean bayt."

    # Multiline should be penalized vs good bayt.
    assert scores["multiline"] < scores["good_bayt_sep_comma"], \
        "Multiline text must be penalized vs single-line bayt."

    # Very short single word should not be considered a good bayt.
    assert scores["single_very_short"] < 0.4, "Single very short word should get low score."

    # Short Arabic phrase should at least get something above pure junk.
    assert scores["short_arabic_phrase"] > scores["latin_junk"], \
        "Short Arabic phrase should score above pure latin junk."

    print("\nAll extended form_reward tests passed ✅")


if __name__ == "__main__":
    run_tests()
