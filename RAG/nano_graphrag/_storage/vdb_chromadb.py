import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb.errors import NotFoundError

from .._utils import logger
from ..base import BaseVectorStorage


def _sanitize_name(raw: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_\-]+", "_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_") or "default"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


async def _to_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


@dataclass
class ChromaDBStorage(BaseVectorStorage):
    """
    Vector store backed by a ChromaDB HTTP server.

    Each storage instance writes to its own Chroma collection. Pass custom
    connection details via `vector_db_storage_cls_kwargs`, e.g.:

        {
            "chromadb_host": "localhost",
            "chromadb_port": 8000,
            "chromadb_ssl": False,
            "chromadb_tenant": "default_tenant",
            "chromadb_database": "book_vectors",
            "chromadb_collection": "alice_in_wonderland",
            "chromadb_reset_collection": True,  # optional
        }
    """

    host: str = "localhost"
    port: int = 8000
    ssl: bool = False
    headers: Optional[Dict[str, str]] = None
    tenant: str = "default_tenant"
    database: str = "default_database"
    collection_name: Optional[str] = None
    reset_collection: bool = False
    collection_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        cfg = self.global_config.get("vector_db_storage_cls_kwargs", {})

        self.host = cfg.get(
            "host", cfg.get("chromadb_host", os.getenv("CHROMADB_HOST", self.host))
        )
        port_value = cfg.get(
            "port", cfg.get("chromadb_port", os.getenv("CHROMADB_PORT", self.port))
        )
        self.port = int(port_value) if port_value is not None else self.port
        ssl_value = cfg.get(
            "ssl", cfg.get("chromadb_ssl", os.getenv("CHROMADB_SSL"))
        )
        self.ssl = _coerce_bool(ssl_value, default=self.ssl)
        self.headers = cfg.get("headers", cfg.get("chromadb_headers", self.headers))
        self.tenant = cfg.get(
            "tenant", cfg.get("chromadb_tenant", os.getenv("CHROMADB_TENANT", self.tenant))
        )
        self.database = cfg.get(
            "database",
            cfg.get("chromadb_database", os.getenv("CHROMADB_DATABASE", self.database)),
        )
        requested_collection = cfg.get(
            "collection_name",
            cfg.get("chromadb_collection", os.getenv("CHROMADB_COLLECTION")),
        )
        if requested_collection:
            # If caller provided a logical base collection name (e.g., the book
            # id), suffix it with the storage namespace so 'entities' and
            # 'chunks' do not clobber each other or delete the same collection.
            base = _sanitize_name(requested_collection)
            self.collection_name = _sanitize_name(f"{base}_{self.namespace}")
        else:
            # Default to a name derived from the working dir and namespace.
            working_dir = _sanitize_name(self.global_config.get("working_dir", "wd"))
            self.collection_name = _sanitize_name(f"{working_dir}_{self.namespace}")
        reset_value = cfg.get(
            "reset_collection",
            cfg.get("chromadb_reset_collection", os.getenv("CHROMADB_RESET_COLLECTION")),
        )
        self.reset_collection = _coerce_bool(reset_value, default=False)

        self.collection_metadata = {
            "namespace": self.namespace,
            "working_dir": self.global_config.get("working_dir"),
        }
        self._max_batch_size = self.global_config.get("embedding_batch_num", 32)
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", 0.2
        )

        self._client = chromadb.HttpClient(
            host=self.host,
            port=self.port,
            ssl=self.ssl,
            headers=self.headers,
            tenant=self.tenant,
            database=self.database,
        )

        if self.reset_collection:
            try:
                self._client.delete_collection(self.collection_name)
                logger.info(
                    "Deleted existing Chroma collection '%s' as requested",
                    self.collection_name,
                )
            except NotFoundError:
                pass

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={**self.collection_metadata, "hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection ready",
            extra={
                "log_type": "chroma",
                "collection": self.collection_name,
                "host": self.host,
                "port": self.port,
                "tenant": self.tenant,
                "database": self.database,
            },
        )

    async def upsert(self, data: dict[str, dict]) -> List[str]:
        if not data:
            logger.warning("Attempted to upsert empty payload into Chroma collection")
            return []

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        contents: List[str] = []

        for doc_id, payload in data.items():
            ids.append(doc_id)
            contents.append(payload["content"])
            documents.append(payload["content"])
            metadata = {
                key: value for key, value in payload.items() if key in self.meta_fields
            }
            metadata["namespace"] = self.namespace
            metadatas.append(metadata)

        # Use the configured embedding function; Chroma expects float vectors
        embeddings = await self.embedding_func(contents)
        embeddings_payload = embeddings.tolist()

        await _to_thread(
            self._collection.upsert,
            ids=ids,
            embeddings=embeddings_payload,
            metadatas=metadatas,
            documents=documents,
        )
        logger.info(
            "Inserted vectors into Chroma",
            extra={
                "log_type": "chroma",
                "collection": self.collection_name,
                "count": len(ids),
                "host": self.host,
                "port": self.port,
            },
        )
        return ids

    async def query(self, query: str, top_k: int = 5) -> List[dict]:
        if top_k <= 0:
            return []

        embedding = await self.embedding_func([query])
        result = await _to_thread(
            self._collection.query,
            query_embeddings=[embedding[0].tolist()],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]

        out: List[dict] = []
        for idx, doc_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            document = documents[idx] if idx < len(documents) else None

            similarity = None
            if distance is not None:
                similarity = 1 - distance
                if similarity < self.cosine_better_than_threshold:
                    continue

            entry = {
                "id": doc_id,
                "distance": distance,
            }
            if similarity is not None:
                entry["similarity"] = similarity
            if document is not None:
                entry["content"] = document
            if metadata:
                entry.update(metadata)

            out.append(entry)

        logger.info(
            "Chroma query",
            extra={
                "log_type": "chroma",
                "collection": self.collection_name,
                "query_len": len(query),
                "returned": len(out),
                "host": self.host,
                "port": self.port,
            },
        )

        return out

    async def index_done_callback(self):
        # No flush required; Chroma persists immediately.
        return
