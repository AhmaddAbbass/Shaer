from __future__ import annotations

from typing import Iterable, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .cleaning import CleanedPoem
from .settings import Settings
from .utils import batched, ensure_dir


class ChromaPoemStore:
    """Handles vector indexing of poem descriptions in Chroma."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=settings.embed_model
        )

        if settings.chroma_mode == "http":
            self.client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port
            )
        else:
            ensure_dir(settings.chroma_persist_dir)
            self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))

        if settings.chroma_reset:
            try:
                self.client.delete_collection(settings.chroma_collection)
            except Exception:
                # Collection might not exist yet; ignore
                pass

        self.collection = self.client.get_or_create_collection(
            settings.chroma_collection,
            embedding_function=self.embedding_function,
            metadata={"source": "shaer_poems"},
        )

    def build(self, poems: Iterable[CleanedPoem], *, batch_size: Optional[int] = None) -> int:
        batch_size = batch_size or self.settings.chroma_batch_size
        good_poems: List[CleanedPoem] = [
            p for p in poems if p.description_clean and not p.has_bad_description
        ]

        for batch in batched(good_poems, batch_size):
            ids = [str(p.poem_id) for p in batch]
            docs = [p.description_clean for p in batch]
            metadatas = [
                {
                    "poem_id": p.poem_id,
                    "poet_name": p.poet_name,
                    "poem_meter": p.poem_meter,
                    "poet_era": p.poet_era,
                    "source": p.source,
                    "num_verses": p.num_verses,
                    "needs_resummarization": p.needs_resummarization,
                }
                for p in batch
            ]
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
        return len(good_poems)

    def query(self, text: str, *, top_k: int = 5) -> List[dict]:
        res = self.collection.query(
            query_texts=[text],
            n_results=top_k,
            include=["distances", "metadatas", "documents"],
        )
        matches = []
        ids = res.get("ids", [[]])[0]
        dists = res.get("distances", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for idx, dist, doc, meta in zip(ids, dists, docs, metas):
            matches.append(
                {
                    "id": idx,
                    "distance": dist,
                    "document": doc,
                    "metadata": meta,
                }
            )
        return matches
