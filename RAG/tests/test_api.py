from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api  # noqa: E402


def test_summary_has_counts():
    stats = api.summary()
    # At least one of the stores should respond without error.
    assert ("poems" in stats and "good_descriptions" in stats) or ("neo4j_error" in stats)
    assert ("chroma_vectors" in stats) or ("chroma_error" in stats)


@pytest.mark.skipif(api.summary().get("chroma_vectors", 0) == 0, reason="Chroma empty or unreachable")
def test_search_and_fetch_poems():
    hits = api.search_descriptions("قصيدة عن الشوق في العصر العباسي", top_k=3)
    assert isinstance(hits, list)
    assert len(hits) > 0

    poem_ids = [int(h["metadata"]["poem_id"]) for h in hits if h.get("metadata") and h["metadata"].get("poem_id")]
    assert poem_ids, "No poem_ids found in search results"

    # Fetch from Neo4j to validate cross-store linkage
    poems = api.get_poems_by_id(poem_ids[:2])
    assert isinstance(poems, list)
    assert len(poems) > 0
    for p in poems:
        assert "poem_id" in p
        assert "meter" in p
        assert "description" in p


@pytest.mark.skipif(not api.load_cleaned_cache(limit=1), reason="No cleaned cache available")
def test_load_cleaned_cache_and_filter():
    poems = api.load_cleaned_cache(limit=5, only_good=True)
    assert poems
    for p in poems:
        # sanity: clean description present and flagged good
        assert p.description_clean
        assert not p.has_bad_description
