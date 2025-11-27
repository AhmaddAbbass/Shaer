from __future__ import annotations

import logging
from typing import Optional

from .dataset import clean_dataset, filter_good_descriptions, load_cleaned_parquet, save_cleaned_parquet
from .graph import Neo4jPoemGraph
from .settings import Settings
from .vector import ChromaPoemStore

logger = logging.getLogger(__name__)


def build_pipeline(
    *,
    limit: Optional[int] = None,
    skip_neo4j: bool = False,
    skip_chroma: bool = False,
    reuse_cleaned: bool = False,
) -> dict:
    """
    End-to-end builder:
      1) load dataset from HF
      2) clean descriptions + flag bad ones
      3) persist cleaned parquet under artifacts/
      4) upsert graph into Neo4j (optional)
      5) build Chroma vector index for good descriptions (optional)
    """
    settings = Settings()

    if reuse_cleaned and settings.cleaned_parquet.exists():
        logger.info("Loading cleaned dataset from %s", settings.cleaned_parquet)
        cleaned = load_cleaned_parquet(settings.cleaned_parquet)
    else:
        logger.info(
            "Loading dataset %s (split=%s, limit=%s)",
            settings.dataset_id,
            settings.dataset_split,
            limit,
        )
        cleaned = clean_dataset(settings, limit=limit)
        logger.info("Saving cleaned dataset to %s", settings.cleaned_parquet)
        save_cleaned_parquet(cleaned, settings.cleaned_parquet)

    good_for_vectors = filter_good_descriptions(cleaned)
    summary = {
        "total_rows": len(cleaned),
        "good_descriptions": len(good_for_vectors),
        "bad_descriptions": len(cleaned) - len(good_for_vectors),
        "parquet_path": str(settings.cleaned_parquet),
    }

    if not skip_neo4j:
        logger.info("Loading %d poems into Neo4j (%s)", len(cleaned), settings.neo4j_uri)
        graph = Neo4jPoemGraph(settings)
        graph.setup_constraints()
        graph.load(cleaned)
        graph.close()

    if not skip_chroma:
        logger.info(
            "Indexing %d poems into Chroma (mode=%s, collection=%s)",
            len(good_for_vectors),
            settings.chroma_mode,
            settings.chroma_collection,
        )
        store = ChromaPoemStore(settings)
        inserted = store.build(good_for_vectors)
        summary["vectors_indexed"] = inserted

    return summary

