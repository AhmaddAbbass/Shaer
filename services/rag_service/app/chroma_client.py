
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions

from services.common_schemas.schemas import RagSearchHit, RagSearchResponse

from .config import Settings
from .logging import get_logger


class ChromaClient:
    """
    Thin wrapper over Chroma for semantic search against poem descriptions.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("rag_service.chroma")
        self.client: ClientAPI = self._build_client()
        self.embedding_function = self._embedding_function()
        self.collection = self._get_collection()

    def _embedding_function(self):
        # Prefer OpenAI if key is set
        if self.settings.openai_api_key:
            if "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = self.settings.openai_api_key
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key_env_var="OPENAI_API_KEY",
                model_name=self.settings.openai_embedding_model,
            )

        # Fallback to SentenceTransformer (no API key needed)
        try:
            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.settings.sentence_embed_model
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to create SentenceTransformer embedding function (%s); "
                "will rely on collection's stored embedding function if present.",
                exc,
            )
            return None

    def _build_client(self) -> ClientAPI:
        mode = (self.settings.chroma_mode or "persistent").lower()
        if mode == "http":
            self.logger.info(
                "Connecting to Chroma via HTTP at %s:%s",
                self.settings.chroma_host,
                self.settings.chroma_port,
            )
            return chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        self.logger.info(
            "Connecting to Chroma persistent store at %s",
            self.settings.chroma_persist_dir,
        )
        # Ensure the directory exists to avoid sqlite/open failures.
        try:
            Path(self.settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("Could not create CHROMA_PERSIST_DIR (%s): %s", self.settings.chroma_persist_dir, exc)
        return chromadb.PersistentClient(
            path=self.settings.chroma_persist_dir,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )

    def _get_collection(self):
        if self.settings.chroma_reset:
            try:
                self.client.delete_collection(self.settings.chroma_collection)
                self.logger.info(
                    "Deleted existing Chroma collection '%s' due to CHROMA_RESET",
                    self.settings.chroma_collection,
                )
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning("Failed to delete collection: %s", exc)

        # Try to reuse existing collection without forcing a new embedding function
        try:
            collection = self.client.get_collection(self.settings.chroma_collection)
            existing_fn = getattr(collection, "_embedding_function", None)
            if existing_fn:
                self.embedding_function = existing_fn
            return collection
        except NotFoundError:
            self.logger.info(
                "Creating Chroma collection '%s' with embedding function %s",
                self.settings.chroma_collection,
                type(self.embedding_function).__name__
                if self.embedding_function
                else "None",
            )
            return self.client.create_collection(
                name=self.settings.chroma_collection,
                embedding_function=self.embedding_function,
            )

    def search(self, query_text: str, top_k: int = 5) -> RagSearchResponse:
        if not query_text:
            return RagSearchResponse(hits=[])

        if self.embedding_function is None:
            raise RuntimeError(
                "Chroma embedding function not configured. "
                "Set OPENAI_API_KEY (and optional OPENAI_EMBEDDING_MODEL)."
            )

        result = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        metas = (result.get("metadatas") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]

        hits: List[RagSearchHit] = []
        for meta, doc, id_ in zip(metas, docs, ids):
            meta = meta or {}
            hits.append(
                RagSearchHit(
                    poem_id=str(meta.get("poem_id", id_)),
                    poem_title=None,
                    poet_name=meta.get("poet_name"),
                    poem_description=doc or "",
                    poem_meter=meta.get("poem_meter"),
                    poem_era=meta.get("poet_era"),
                    poem_theme=meta.get("poem_theme"),
                    has_bad_description=bool(meta.get("needs_resummarization", False)),
                )
            )

        return RagSearchResponse(hits=hits)

    def get_description(self, poem_id: str) -> Optional[str]:
        """
        Fetch a poem description directly by id (used for /similar fallback).
        """
        try:
            res = self.collection.get(ids=[poem_id], include=["documents"])
            docs = res.get("documents") or []
            if docs and docs[0]:
                return docs[0][0] if isinstance(docs[0], list) else docs[0]
        except Exception:
            return None
        return None

    def ping(self) -> bool:
        try:
            _ = self.collection.count()
            return True
        except Exception:
            return False
