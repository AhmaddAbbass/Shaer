# nano_graphrag/_storage/vdb_neo4j.py

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from neo4j import AsyncGraphDatabase

from .._utils import logger
from ..base import BaseVectorStorage
from .gdb_neo4j import make_path_idable


neo4j_vector_lock = asyncio.Lock()


@dataclass
class Neo4jVectorStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    max_batch_size: int = 32

    def __post_init__(self):
        addon_params = self.global_config.get("addon_params", {})
        self.neo4j_url = addon_params.get("neo4j_url")
        self.neo4j_auth = addon_params.get("neo4j_auth")
        self.neo4j_database = addon_params.get("neo4j_database")

        if not self.neo4j_url or not self.neo4j_auth or not self.neo4j_database:
            raise ValueError(
                "Neo4jVectorStorage requires neo4j_url, neo4j_auth, and neo4j_database "
                "inside addon_params."
            )

        # label for the vector nodes in Neo4j
        prefix = make_path_idable(self.global_config["working_dir"])
        self.label = f"{prefix}__{self.namespace}_vector"

        # name of the vector index (separate from label)
        self.index_name = f"{self.label}_index"

        # batch + threshold settings (hooked to GraphRAG config)
        self.max_batch_size = self.global_config.get(
            "embedding_batch_num", self.max_batch_size
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "query_better_than_threshold", self.cosine_better_than_threshold
        )

        # embedding dim: try to read from kwargs, otherwise probe once later
        vdb_kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {}) or {}
        self.embedding_dim = vdb_kwargs.get("embedding_dim")  # may be None

        self.async_driver = AsyncGraphDatabase.driver(
            self.neo4j_url,
            auth=self.neo4j_auth,
            max_connection_pool_size=10,
        )

        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._create_constraints_and_index()
            self._initialized = True

    async def _create_constraints_and_index(self):
        """
        - Ensure (id) is unique for this label
        - Ensure a Neo4j vector index exists on (n.vector) for this label
        """
        async with self.async_driver.session(database=self.neo4j_database) as session:
            # 1) Unique constraint on id
            stmt = (
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{self.label}`) "
                "REQUIRE n.id IS UNIQUE"
            )
            await session.run(stmt)

            # 2) Figure out embedding dimension if we don't know it yet
            dim = self.embedding_dim
            if dim is None:
                logger.info("Neo4jVectorStorage: probing embedding dimension once...")
                emb = await self.embedding_func(["__dimension_probe__"])

                # NOTE: embedding_func may return either a list of vectors or a numpy array.
                # We must not use `not emb` on a numpy array (it is ambiguous),
                # so we normalize and check length explicitly.
                if emb is None:
                    raise RuntimeError(
                        "Neo4jVectorStorage: embedding_func returned None while probing dimension."
                    )

                if isinstance(emb, np.ndarray):
                    if emb.size == 0:
                        raise RuntimeError(
                            "Neo4jVectorStorage: embedding_func returned empty ndarray while probing dimension."
                        )
                    first_vec = emb[0]
                else:
                    if len(emb) == 0:
                        raise RuntimeError(
                            "Neo4jVectorStorage: embedding_func returned empty list while probing dimension."
                        )
                    first_vec = emb[0]

                if first_vec is None or (hasattr(first_vec, "__len__") and len(first_vec) == 0):
                    raise RuntimeError(
                        "Neo4jVectorStorage: embedding_func returned empty embedding vector while probing dimension."
                    )

                dim = len(first_vec)
                self.embedding_dim = dim
                logger.info(f"Neo4jVectorStorage: detected embedding_dim = {dim}")

            # 3) Check if vector index already exists (Neo4j 5 style)
            check_result = await session.run(
                """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE name = $index_name
                RETURN count(*) AS count
                """,
                index_name=self.index_name,
            )
            rec = await check_result.single()
            exists = rec and rec["count"] > 0

            if not exists:
                logger.info(
                    f"Creating Neo4j vector index '{self.index_name}' "
                    f"on label '{self.label}' property 'vector' (dim={dim})"
                )
                # Neo4j 5 vector index creation
                await session.run(
                    """
                    CALL db.index.vector.createNodeIndex(
                        $index_name,
                        $label,
                        'vector',
                        $dim,
                        'cosine'
                    )
                    """,
                    index_name=self.index_name,
                    label=self.label,
                    dim=dim,
                )
            else:
                logger.info(f"Vector index '{self.index_name}' already exists, skipping.")

    async def upsert(self, data: Dict[str, Dict]):
        await self._ensure_initialized()
        if not data:
            logger.warning("Neo4jVectorStorage.upsert received empty payload.")
            return []

        logger.info(f"Inserting {len(data)} vectors into Neo4j label {self.label}")

        items = list(data.items())
        contents = [item[1]["content"] for item in items]

        # --- batch embeddings ---
        batches = [
            contents[i : i + self.max_batch_size]
            for i in range(0, len(contents), self.max_batch_size)
        ]
        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = np.concatenate(embeddings_list)

        rows = []
        '''
        already ready to put meta data like author source info into Neo4j
        '''
        for (item_id, payload), embedding in zip(items, embeddings):
            meta = self._serialize_meta(payload)
            rows.append(
                {
                    "id": item_id,
                    "content": payload["content"],
                    "vector": embedding.tolist(),
                    "meta": meta,
                }
            )

        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:`{self.label}` {{id: row.id}})
        SET n.content = row.content,
            n.vector  = row.vector,
            n.namespace = $namespace,
            n.updated_at = datetime()
        SET n += row.meta
        """

        async with self.async_driver.session(database=self.neo4j_database) as session:
            await session.run(cypher, rows=rows, namespace=self.namespace)

        return rows

    async def query(self, query: str, top_k: int = 5):
        await self._ensure_initialized()

        # 1) embed query
        query_embedding = await self.embedding_func([query])
        query_vector = [float(x) for x in query_embedding[0]]

        # 2) use Neo4j's vector index to retrieve top-k
        async with self.async_driver.session(database=self.neo4j_database) as session:
            cypher = """
            CALL db.index.vector.queryNodes(
                $index_name,
                $top_k,
                $query_vector
            )
            YIELD node, score
            RETURN node.id      AS id,
                   node.content  AS content,
                   node          AS node,
                   score         AS score
            """
            result = await session.run(
                cypher,
                index_name=self.index_name,
                top_k=top_k,
                query_vector=query_vector,
            )
            records = [record async for record in result]

        if not records:
            return []

        scored = []
        for record in records:
            node = record["node"]
            sim = float(record["score"])     # similarity (higher is better)
            distance = 1.0 - sim             # keep "distance" API like before

            # optional filter: require similarity >= threshold
            if distance > 1 - self.cosine_better_than_threshold:
                # i.e. sim < cosine_better_than_threshold
                continue

            meta = self._extract_meta(node)
            scored.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "distance": distance,
                    **meta,
                }
            )

        # Neo4j already returns sorted by similarity desc, but sorting again is cheap
        scored.sort(key=lambda x: x["distance"])
        return scored[:top_k]

    def _serialize_meta(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = {}
        for field in self.meta_fields:
            if field in payload:
                value = payload[field]
                meta[field] = value
        return meta

    def _extract_meta(self, node_props: Dict[str, Any]) -> Dict[str, Any]:
        if not node_props:
            return {}
        return {field: node_props.get(field) for field in self.meta_fields if field in node_props}

    @staticmethod
    def _cosine_distance(a: List[float], b: List[float]) -> float | None:
        """
        Kept for compatibility, but no longer used in the main query path.
        """
        a_vec = np.array(a, dtype=float)
        b_vec = np.array(b, dtype=float)
        denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
        if denom == 0:
            return None
        cosine_sim = float(np.dot(a_vec, b_vec) / denom)
        return 1 - cosine_sim
