"""
Lightweight retrieval helper for Shaer RAG.

Usage:
    from agent import retrieve_similar_descriptions
    hits = retrieve_similar_descriptions("قصيدة عن الشوق في العصر العباسي", top_k=5)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any


def _ensure_rag_on_path() -> Path:
    """Add RAG directory to sys.path so shaer_rag imports work from repo root."""
    rag_root = Path(__file__).resolve().parent / "RAG"
    if str(rag_root) not in sys.path:
        sys.path.insert(0, str(rag_root))
    return rag_root


def retrieve_similar_descriptions(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Given a raw user query, return the top-k most similar poem descriptions
    from the Chroma index (IDs match poem_id).
    """
    _ensure_rag_on_path()
    from shaer_rag.settings import Settings
    from shaer_rag.vector import ChromaPoemStore

    settings = Settings()
    store = ChromaPoemStore(settings)
    return store.query(query, top_k=top_k)


if __name__ == "__main__":
    import json

    rag_root = _ensure_rag_on_path()
    print(f"[agent] Using RAG root: {rag_root}")
    q = input("Enter your query (Arabic): ").strip()
    k_str = input("Top-k (default 5): ").strip()
    k = int(k_str) if k_str else 5
    hits = retrieve_similar_descriptions(q, top_k=k)
    print(json.dumps(hits, ensure_ascii=False, indent=2))
