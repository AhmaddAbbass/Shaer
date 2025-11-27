from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from shaer_rag.cleaning import CleanedPoem
from shaer_rag.settings import Settings


class DummyEmbeddingFunction:
    """Stable, tiny embedding to avoid downloading real models during tests."""

    def __init__(self, *args, **kwargs):
        pass

    def name(self) -> str:
        return "dummy"

    def is_legacy(self) -> bool:
        return False

    @property
    def default_space(self) -> str:
        return "cosine"

    def __call__(self, input):
        texts = input if isinstance(input, list) else [input]
        return [[float(i), float(i), float(i)] for i, _ in enumerate(texts)]

    def embed_documents(self, input):
        return self.__call__(input)

    def embed_query(self, input):
        return self.__call__(input)


def build_store(monkeypatch):
    tmpdir = tempfile.mkdtemp()
    monkeypatch.setenv("RAG_ARTIFACT_DIR", tmpdir)
    monkeypatch.setenv("CHROMA_MODE", "persistent")
    monkeypatch.setenv("CHROMA_RESET", "1")

    # Patch embedding function to a cheap dummy.
    from shaer_rag import vector

    monkeypatch.setattr(
        vector, "SentenceTransformerEmbeddingFunction", DummyEmbeddingFunction
    )

    store = vector.ChromaPoemStore(Settings())
    return store, Path(tmpdir)


def sample_poems():
    return [
        CleanedPoem(
            poem_id=1,
            poem_title="demo1",
            poem_meter="الطويل",
            poem_theme="الشوق",
            poem_url=None,
            poet_name="شاعر ١",
            poet_url=None,
            poet_description=None,
            poet_era="العصر العباسي",
            poet_location=None,
            poem_language_type="arabic",
            poem_verses=["بيت"],
            num_verses=1,
            description_raw="",
            description_clean="قصيدة عن الشوق والحنين.",
            has_bad_description=False,
            needs_resummarization=False,
            issues=[],
            source="test",
            verse_preview="بيت",
            poet_key="شاعر ١|العصر العباسي",
        ),
        CleanedPoem(
            poem_id=2,
            poem_title="demo2",
            poem_meter="الكامل",
            poem_theme="الفخر",
            poem_url=None,
            poet_name="شاعر ٢",
            poet_url=None,
            poet_description=None,
            poet_era="العصر الأموي",
            poet_location=None,
            poem_language_type="arabic",
            poem_verses=["بيت"],
            num_verses=1,
            description_raw="",
            description_clean="قصيدة قصيرة في الفخر.",
            has_bad_description=False,
            needs_resummarization=False,
            issues=[],
            source="test",
            verse_preview="بيت",
            poet_key="شاعر ٢|العصر الأموي",
        ),
    ]


def test_chroma_build_and_query(monkeypatch):
    store, tmpdir = build_store(monkeypatch)
    poems = sample_poems()

    try:
        inserted = store.build(poems, batch_size=10)
        assert inserted == 2
        assert store.collection.count() == 2

        hits = store.query("قصيدة عن الشوق", top_k=1)
        assert len(hits) == 1
        assert hits[0]["metadata"]["poem_id"] in {1, 2}
        assert "document" in hits[0]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_agent_retrieve_reuses_index(monkeypatch):
    # Build with reset=1 then query with reset=0 to ensure persistence works.
    store, tmpdir = build_store(monkeypatch)
    poems = sample_poems()
    try:
        store.build(poems, batch_size=10)
        assert store.collection.count() == 2

        # Preserve index on next client init
        monkeypatch.setenv("CHROMA_RESET", "0")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from agent import retrieve_similar_descriptions

        hits = retrieve_similar_descriptions("قصيدة عن الشوق", top_k=2)
        assert len(hits) == 2
        ids = {int(h["id"]) for h in hits}
        assert ids == {1, 2}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
