from services.agent_service.app.utils import extract_candidate_verses, format_poem_snippet


def test_extract_candidate_verses_filters_short_lines():
    text = """
    هذا سطر قصير
    بيت طويل يصف الشوق إلى الوطن

    آخر
    هذا البيت يكمل الصورة
    """
    verses = extract_candidate_verses(text)
    assert "بيت طويل يصف الشوق إلى الوطن" in verses
    assert "هذا البيت يكمل الصورة" in verses
    assert "هذا سطر قصير" not in verses


def test_format_poem_snippet_limits_lines():
    verses = [f"بيت رقم {i}" for i in range(1, 10)]
    snippet = format_poem_snippet(verses, max_lines=3)
    assert snippet.count("\n") == 2
    assert snippet.startswith("بيت رقم 1")
    assert snippet.endswith("بيت رقم 3")
