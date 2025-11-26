from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shaer_rag.settings import Settings
from shaer_rag.vector import ChromaPoemStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ad-hoc Chroma semantic search.")
    parser.add_argument("--text", required=True, help="Arabic query text to search with.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings()
    store = ChromaPoemStore(settings)
    hits = store.query(args.text, top_k=args.top_k)
    payload = json.dumps(hits, ensure_ascii=False, indent=2)
    # Write via buffer to avoid Windows console encoding errors on Arabic text.
    sys.stdout.buffer.write(payload.encode("utf-8"))
    sys.stdout.buffer.write(b"\n")


if __name__ == "__main__":
    main()
