"""
Thin façade over the Shaer RAG stack.

Exports a minimal, stable surface for:
- building/refreshing the index (HF -> clean -> Neo4j -> Chroma),
- semantic search over poem descriptions,
- sanity checks (counts in Neo4j and Chroma).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure RAG package is importable when running from repo root.
RAG_ROOT = Path(__file__).resolve().parent / "RAG"
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))

from shaer_rag.pipeline import build_pipeline
from shaer_rag.vector import ChromaPoemStore
from shaer_rag.settings import Settings
from shaer_rag.dataset import load_cleaned_parquet, filter_good_descriptions

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore

__all__ = [
    "build_index",
    "search_descriptions",
    "neo4j_counts",
    "chroma_count",
    "summary",
    "load_cleaned_cache",
    "get_poems_by_id",
]


def build_index(
    *,
    limit: Optional[int] = None,
    reuse_cleaned: bool = False,
    skip_neo4j: bool = False,
    skip_chroma: bool = False,
) -> Dict[str, Any]:
    """
    Run the full pipeline (or subsets) and return a summary dict.

    Args:
        limit: Optional row cap for faster dry runs.
        reuse_cleaned: If True, load artifacts/clean_poems.parquet when present.
        skip_neo4j: If True, do not write to Neo4j.
        skip_chroma: If True, do not build Chroma.
    """
    return build_pipeline(
        limit=limit,
        reuse_cleaned=reuse_cleaned,
        skip_neo4j=skip_neo4j,
        skip_chroma=skip_chroma,
    )


def search_descriptions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Semantic search over poem descriptions.
    Returns a list of {id, distance, document, metadata} dicts.
    """
    settings = Settings()
    store = ChromaPoemStore(settings)
    return store.query(query, top_k=top_k)


def chroma_count() -> int:
    """Return the number of vectors currently stored in Chroma."""
    settings = Settings()
    store = ChromaPoemStore(settings)
    return store.collection.count()


def load_cleaned_cache(limit: Optional[int] = None, only_good: bool = True):
    """
    Load rows from the cached cleaned parquet (artifacts/clean_poems.parquet).
    Args:
        limit: optionally slice the first N rows.
        only_good: if True, filter to poems with good descriptions.
    Returns:
        list of CleanedPoem objects.
    """
    settings = Settings()
    poems = load_cleaned_parquet(settings.cleaned_parquet)
    if only_good:
        poems = filter_good_descriptions(poems)
    if limit is not None:
        poems = poems[:limit]
    return poems


def neo4j_counts() -> Dict[str, int]:
    """
    Return basic counts from Neo4j: poems total, poems with good descriptions.
    Requires neo4j driver to be installed and reachable per .env settings.
    """
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver is not installed.")

    settings = Settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    with driver.session(database=settings.neo4j_database) as session:
        poems = session.run("MATCH (p:Poem) RETURN count(p) AS c").single()["c"]
        good = session.run(
            "MATCH (p:Poem {has_bad_description:false}) RETURN count(p) AS c"
        ).single()["c"]
    driver.close()
    return {"poems": poems, "good_descriptions": good}


def get_poems_by_id(poem_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch poem node metadata by poem_id from Neo4j.
    Returns a list of dicts with key properties (title, meter, era, theme, url, etc.).
    """
    if GraphDatabase is None:
        raise RuntimeError("neo4j driver is not installed.")
    if not poem_ids:
        return []

    settings = Settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )
    cypher = """
    MATCH (p:Poem)
    WHERE p.poem_id IN $ids
    OPTIONAL MATCH (p)-[:WRITTEN_BY]->(poet:Poet)
    OPTIONAL MATCH (p)-[:IN_ERA]->(era:Era)
    OPTIONAL MATCH (p)-[:IN_METER]->(m:Meter)
    OPTIONAL MATCH (p)-[:HAS_THEME]->(t:Theme)
    RETURN p.poem_id AS poem_id,
           p.title AS title,
           p.description_clean AS description,
           p.has_bad_description AS has_bad_description,
           p.num_verses AS num_verses,
           p.source AS source,
           poet.name AS poet_name,
           era.name AS era,
           m.name AS meter,
           t.name AS theme
    """
    results: List[Dict[str, Any]] = []
    with driver.session(database=settings.neo4j_database) as session:
        for rec in session.run(cypher, ids=[int(i) for i in poem_ids]):
            results.append({k: rec.get(k) for k in rec.keys()})
    driver.close()
    return results


def summary() -> Dict[str, Any]:
    """
    Combined quick stats: Neo4j counts (if reachable) and Chroma vector count.
    """
    stats: Dict[str, Any] = {}
    try:
        stats.update(neo4j_counts())
    except Exception as e:  # pragma: no cover - depends on env
        stats["neo4j_error"] = str(e)
    try:
        stats["chroma_vectors"] = chroma_count()
    except Exception as e:  # pragma: no cover - depends on env
        stats["chroma_error"] = str(e)
    return stats
