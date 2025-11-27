from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the repo root (containing shaer_rag/) is importable when invoked as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaer_rag.pipeline import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Shaer RAG graph + vector index from the HF dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of poems ingested (useful for dry runs).",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip writing nodes/edges to Neo4j.",
    )
    parser.add_argument(
        "--skip-chroma",
        action="store_true",
        help="Skip building the Chroma vector index.",
    )
    parser.add_argument(
        "--reuse-cleaned",
        action="store_true",
        help="Reuse artifacts/clean_poems.parquet instead of re-downloading HF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = build_pipeline(
        limit=args.limit,
        skip_neo4j=args.skip_neo4j,
        skip_chroma=args.skip_chroma,
        reuse_cleaned=args.reuse_cleaned,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
