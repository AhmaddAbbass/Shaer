# utils/rewards/test_clean_bayt.py

import re

# -------------------------------------------------------------------
# IMPORT YOUR CLEANER (used inside meaning_reward)
# -------------------------------------------------------------------
from utils.rewards.meter_reward import meter_reward, classifier_meter_scores
from utils.rewards.meaning_reward import _normalize_arabic, meaning_reward

# -------------------------------------------------------------------
# Helper: pretty print raw → clean pairs
# -------------------------------------------------------------------
def test_one(name, text):
    clean = _normalize_arabic(text)
    print(f"\n=== TEST: {name} ===")
    print("RAW:   ", text)
    print("CLEAN: ", clean)


# -------------------------------------------------------------------
# Test cases
# -------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Single perfect bayt with clear sadr / ajuz
    test_one(
        "perfect_wafir",
        "كَأَنّي إِذ نَزَلتُ عَلى المُعَلّى    نَزَلتُ عَلى البَواذِخِ مِن شَمامِ"
    )

    # 2. Perfect bayt with many spaces
    test_one(
        "extra_spaces",
        "ومـلكـت   نـجـمـة  السـعـد السـما       فـيـهـا فـمـا   عـنـهـا نـبـعـت   فـلك"
    )

    # 3. Model output with 2 bayts glued in same line (should NOT remove sadr/ajuz of first)
    test_one(
        "two_bayts_same_line",
        "لِي مِنْ هَوَاكَ بَعِيدُهُ وَقَرِيبُهُ، يا من أُعِيذُ جَمالَهُ مِنْ نَاظِرٍ يَصِيبُهُ"
    )

    # 4. Model output: sadr/ajuz + second bayt attempted continuation (we should NOT damage the first)
    test_one(
        "sadr_ajuz_plus_extra",
        "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ    بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ   وها أنا أكتب بيتاً آخر"
    )

    # 5. Long mush text, multiple sentences (we *should* cut to first sentence)
    test_one(
        "long_mush",
        "هذا نص طويل جداً جداً فيه أكثر من جملة وأكثر من معنى ولا يشبه بيتاً شعرياً لكنه مكتوب بالعربية وربما يسبب مشاكل لو تركناه كما هو بدون قصّ"
    )

    # 6. English garbage input
    test_one(
        "english",
        "hello world this is not arabic poetry and should be removed eventually"
    )

    # 7. Almost-bayt but with newlines (common in model output)
    test_one(
        "newline_bayt",
        "سَئِمْتُ تَكَالِيفَ الحَيَاةِ وَمَنْ يَعِشْ\nثَمَانِينَ حَوْلاً لا أَبَا لَكَ يَسْأَمِ"
    )

    # 8. Extra punctuation / emojis
    test_one(
        "punctuation_emojis",
        "يا ليلُ طالَ السُّرى 😢 واشتدَّ بي السقمُ!! فما لي سوى نجمةٍ تُهدي إليّ النَّسَمَ"
    )

    print("\n\nAll cleaner tests finished.\n")
