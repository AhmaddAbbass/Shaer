import json
import asyncio
import re
from collections import defaultdict
from typing import List
from neo4j import AsyncGraphDatabase
from dataclasses import dataclass
from typing import Union
try:
    from ..base import BaseGraphStorage, SingleCommunitySchema
    from .._utils import logger
    from ..prompt import GRAPH_FIELD_SEP
except ImportError:  # allow running outside the package tree
    from nano_graphrag.base import BaseGraphStorage, SingleCommunitySchema
    from nano_graphrag._utils import logger
    from nano_graphrag.prompt import GRAPH_FIELD_SEP

neo4j_lock = asyncio.Lock()

ROUTING_SCHEMES = ("neo4j://", "neo4j+s://", "neo4j+ssc://")


def make_path_idable(path):
    return path.replace(".", "_").replace("/", "__").replace("-", "_").replace(":", "_").replace("\\", "__")


def sanitize_database_name(name: str) -> str:
    """Neo4j DB names allow alphanumerics, dots, and dashes."""
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "-", (name or "").strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        raise ValueError("Provided neo4j_database is empty or only illegal characters.")
    return cleaned.lower()


def ensure_direct_scheme(uri: str) -> str:
    """Convert routing schemes to bolt equivalents for single-instance setups."""
    if uri.startswith("neo4j+s://"):
        return "bolt+s://" + uri[len("neo4j+s://") :]
    if uri.startswith("neo4j+ssc://"):
        return "bolt+ssc://" + uri[len("neo4j+ssc://") :]
    if uri.startswith("neo4j://"):
        return "bolt://" + uri[len("neo4j://") :]
    return uri


@dataclass
class Neo4jStorage(BaseGraphStorage):
    def __post_init__(self):
        self.neo4j_url = self.global_config["addon_params"].get("neo4j_url", None)
        self.neo4j_auth = self.global_config["addon_params"].get("neo4j_auth", None)
        raw_db_name = self.global_config["addon_params"].get("neo4j_database", None)
        self.neo4j_enable_routing = bool(self.global_config["addon_params"].get("neo4j_enable_routing", False))
        self.namespace = (
            f"{make_path_idable(self.global_config['working_dir'])}__{self.namespace}"
        )
        print("DEBUG ADDON PARAMS:", self.global_config.get("addon_params"))

        logger.info(f"Using the label {self.namespace} for Neo4j as identifier")
        if self.neo4j_url is None or self.neo4j_auth is None:
            raise ValueError("Missing neo4j_url or neo4j_auth in addon_params")
        if not raw_db_name:
            raise ValueError("Missing neo4j_database in addon_params")
        sanitized_db = sanitize_database_name(raw_db_name)
        if sanitized_db != raw_db_name:
            logger.warning(
                "Neo4j database name '%s' sanitized to '%s' (letters/numbers/dots/dashes only).",
                raw_db_name,
                sanitized_db,
            )
        self.neo4j_database = sanitized_db

        if not self.neo4j_enable_routing and self.neo4j_url.startswith(ROUTING_SCHEMES):
            # convert to bolt scheme to avoid routing-table lookups on single instances
            effective_uri = ensure_direct_scheme(self.neo4j_url)
            logger.info(
                "Routing disabled; using direct Neo4j URI '%s' instead of '%s'.",
                effective_uri,
                self.neo4j_url,
            )
        else:
            effective_uri = self.neo4j_url
        self._driver_uri = effective_uri

        self.async_driver = AsyncGraphDatabase.driver(
            self._driver_uri, auth=self.neo4j_auth, max_connection_pool_size=50,
        )

    async def create_database(self):
        """Ensure the author-specific database exists, creating it if missing."""
        db_name = self.neo4j_database
        escaped_db = db_name.replace("`", "``")
        try:
            async with self.async_driver.session(database="system") as session:
                result = await session.run(
                    "SHOW DATABASES WHERE name = $db_name",
                    db_name=db_name,
                )
                exists = False
                async for _ in result:
                    exists = True
                    break
                if not exists:
                    await session.run(f"CREATE DATABASE `{escaped_db}` IF NOT EXISTS")
                    logger.info("Created Neo4j database '%s'", db_name)
        except Exception as e:
            logger.error("Failed to ensure Neo4j database '%s': %s", db_name, e)
            raise

    async def _init_workspace(self):
        await self.async_driver.verify_authentication()
        await self.async_driver.verify_connectivity()
        await self.create_database()

    async def index_start_callback(self):
        logger.info("Init Neo4j workspace")
        await self._init_workspace()
        
        # create index for faster searching
        try:
            async with self.async_driver.session(database=self.neo4j_database) as session:
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.id)"
                )
                
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.entity_type)"
                )
                
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.communityIds)"
                )
                
                await session.run(
                    f"CREATE INDEX IF NOT EXISTS FOR (n:`{self.namespace}`) ON (n.source_id)"
                )          
                logger.info("Neo4j indexes created successfully")                
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise e

    async def has_node(self, node_id: str) -> bool:
        async with self.async_driver.session(database=self.neo4j_database) as session:
            result = await session.run(
                f"MATCH (n:`{self.namespace}`) WHERE n.id = $node_id RETURN COUNT(n) > 0 AS exists",
                node_id=node_id,
            )
            record = await result.single()
            return record["exists"] if record else False

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        async with self.async_driver.session(database=self.neo4j_database) as session:
            result = await session.run(
                f"""
                MATCH (s:`{self.namespace}`)
                WHERE s.id = $source_id
                MATCH (t:`{self.namespace}`)
                WHERE t.id = $target_id
                RETURN EXISTS((s)-[]->(t)) AS exists
                """,
                source_id=source_node_id,
                target_id=target_node_id,
            )
    
            record = await result.single()
            return record["exists"] if record else False

    async def node_degree(self, node_id: str) -> int:
        results = await self.node_degrees_batch([node_id])
        return results[0] if results else 0
        
    async def node_degrees_batch(self, node_ids: List[str]) -> List[str]:
        if not node_ids:
            return {}
                    
        result_dict = {node_id: 0 for node_id in node_ids}
        async with self.async_driver.session(database=self.neo4j_database) as session:
            result = await session.run(
                f"""
                UNWIND $node_ids AS node_id
                MATCH (n:`{self.namespace}`)
                WHERE n.id = node_id
                OPTIONAL MATCH (n)-[]-(m:`{self.namespace}`)
                RETURN node_id, COUNT(m) AS degree
                """,
                node_ids=node_ids
            )
                
            async for record in result:
                result_dict[record["node_id"]] = record["degree"]
                
        return [result_dict[node_id] for node_id in node_ids]
    
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        results = await self.edge_degrees_batch([(src_id, tgt_id)])
        return results[0] if results else 0

    async def edge_degrees_batch(self, edge_pairs: list[tuple[str, str]]) -> list[int]:
        if not edge_pairs:
            return []
        
        result_dict = {tuple(edge_pair): 0 for edge_pair in edge_pairs}
        
        edges_params = [{"src_id": src, "tgt_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session(database=self.neo4j_database) as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    
                    MATCH (s:`{self.namespace}`)
                    WHERE s.id = edge.src_id
                    WITH edge, s
                    OPTIONAL MATCH (s)-[]-(n1:`{self.namespace}`)
                    WITH edge, COUNT(n1) AS src_degree
                    
                    MATCH (t:`{self.namespace}`)
                    WHERE t.id = edge.tgt_id
                    WITH edge, src_degree, t
                    OPTIONAL MATCH (t)-[]-(n2:`{self.namespace}`)
                    WITH edge.src_id AS src_id, edge.tgt_id AS tgt_id, src_degree, COUNT(n2) AS tgt_degree
                    
                    RETURN src_id, tgt_id, src_degree + tgt_degree AS degree
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    src_id = record["src_id"]
                    tgt_id = record["tgt_id"]
                    degree = record["degree"]
                    
                    # 更新结果字典
                    edge_pair = (src_id, tgt_id)
                    result_dict[edge_pair] = degree
            
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge degree calculation: {e}")
            return [0] * len(edge_pairs)



    async def get_node(self, node_id: str) -> Union[dict, None]:
        result = await self.get_nodes_batch([node_id])
        return result[0] if result else None

    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, Union[dict, None]]:
        if not node_ids:
            return {}
            
        result_dict = {node_id: None for node_id in node_ids}

        try:
            async with self.async_driver.session(database=self.neo4j_database) as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (n:`{self.namespace}`)
                    WHERE n.id = node_id
                    RETURN node_id, properties(n) AS node_data
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    node_id = record["node_id"]
                    raw_node_data = record["node_data"]
                    
                    if raw_node_data:
                        raw_node_data["clusters"] = json.dumps(
                            [
                                {
                                    "level": index,
                                    "cluster": cluster_id,
                                }
                                for index, cluster_id in enumerate(
                                    raw_node_data.get("communityIds", [])
                                )
                            ]
                        )
                        result_dict[node_id] = raw_node_data
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node retrieval: {e}")
            raise e

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        results = await self.get_edges_batch([(source_node_id, target_node_id)])
        return results[0] if results else None

    async def get_edges_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> list[Union[dict, None]]:
        if not edge_pairs:
            return []
            
        result_dict = {tuple(edge_pair): None for edge_pair in edge_pairs}
        
        edges_params = [{"source_id": src, "target_id": tgt} for src, tgt in edge_pairs]
        
        try:
            async with self.async_driver.session(database=self.neo4j_database) as session:
                result = await session.run(
                    f"""
                    UNWIND $edges AS edge
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = edge.source_id AND t.id = edge.target_id
                    RETURN edge.source_id AS source_id, edge.target_id AS target_id, properties(r) AS edge_data
                    """,
                    edges=edges_params
                )
                
                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    edge_data = record["edge_data"]
                    
                    edge_pair = (source_id, target_id)
                    result_dict[edge_pair] = edge_data
            
            return [result_dict[tuple(edge_pair)] for edge_pair in edge_pairs]
        except Exception as e:
            logger.error(f"Error in batch edge retrieval: {e}")
            return [None] * len(edge_pairs)

    async def get_node_edges(
        self, source_node_id: str
    ) -> list[tuple[str, str]]:
        results = await self.get_nodes_edges_batch([source_node_id])
        return results[0] if results else []

    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> list[list[tuple[str, str]]]:
        if not node_ids:
            return []
            
        result_dict = {node_id: [] for node_id in node_ids}
        
        try:
            async with self.async_driver.session(database=self.neo4j_database) as session:
                result = await session.run(
                    f"""
                    UNWIND $node_ids AS node_id
                    MATCH (s:`{self.namespace}`)-[r]->(t:`{self.namespace}`)
                    WHERE s.id = node_id
                    RETURN s.id AS source_id, t.id AS target_id
                    """,
                    node_ids=node_ids
                )
                
                async for record in result:
                    source_id = record["source_id"]
                    target_id = record["target_id"]
                    
                    if source_id in result_dict:
                        result_dict[source_id].append((source_id, target_id))
            
            return [result_dict[node_id] for node_id in node_ids]
        except Exception as e:
            logger.error(f"Error in batch node edges retrieval: {e}")
            return [[] for _ in node_ids]

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        await self.upsert_nodes_batch([(node_id, node_data)])

    async def upsert_nodes_batch(self, nodes_data: list[tuple[str, dict[str, str]]]):
        if not nodes_data:
            return []
        
        nodes_by_type = {}
        for node_id, node_data in nodes_data:
            node_type = node_data.get("entity_type", "UNKNOWN").strip('"')
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append((node_id, node_data))
        
        async with self.async_driver.session(database=self.neo4j_database) as session:
            for node_type, type_nodes in nodes_by_type.items():
                params = [{"id": node_id, "data": node_data} for node_id, node_data in type_nodes]
                
                await session.run(
                    f"""
                    UNWIND $nodes AS node
                    MERGE (n:`{self.namespace}`:`{node_type}` {{id: node.id}})
                    SET n += node.data
                    """,
                    nodes=params
                )
        
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        await self.upsert_edges_batch([(source_node_id, target_node_id, edge_data)])


    async def upsert_edges_batch(
        self, edges_data: list[tuple[str, str, dict[str, str]]]
    ):
        if not edges_data:
            return
        
        edges_params = []
        for source_id, target_id, edge_data in edges_data:
            edge_data_copy = edge_data.copy() 
            edge_data_copy.setdefault("weight", 0.0)
            
            edges_params.append({
                "source_id": source_id,
                "target_id": target_id,
                "edge_data": edge_data_copy
            })
        
        async with self.async_driver.session(database=self.neo4j_database) as session:
            await session.run(
                f"""
                UNWIND $edges AS edge
                MATCH (s:`{self.namespace}`)
                WHERE s.id = edge.source_id
                WITH edge, s
                MATCH (t:`{self.namespace}`)
                WHERE t.id = edge.target_id
                MERGE (s)-[r:RELATED]->(t)
                SET r += edge.edge_data
                """,
                edges=edges_params
            )
        



    async def clustering(self, algorithm: str):
        if algorithm != "leiden":
            raise ValueError(
                f"Clustering algorithm {algorithm} not supported in Neo4j implementation"
            )

        random_seed = self.global_config["graph_cluster_seed"]
        max_level = self.global_config["max_graph_cluster_size"]
        async with self.async_driver.session(database=self.neo4j_database) as session:
            try:
                # Project the graph with undirected relationships
                await session.run(
                    f"""
                    CALL gds.graph.project(
                        'graph_{self.namespace}',
                        ['{self.namespace}'],
                        {{
                            RELATED: {{
                                orientation: 'UNDIRECTED',
                                properties: ['weight']
                            }}
                        }}
                    )
                    """
                )

                # Run Leiden algorithm
                result = await session.run(
                    f"""
                    CALL gds.leiden.write(
                        'graph_{self.namespace}',
                        {{
                            writeProperty: 'communityIds',
                            includeIntermediateCommunities: True,
                            relationshipWeightProperty: "weight",
                            maxLevels: {max_level},
                            tolerance: 0.0001,
                            gamma: 1.0,
                            theta: 0.01,
                            randomSeed: {random_seed}
                        }}
                    )
                    YIELD communityCount, modularities;
                    """
                )
                result = await result.single()
                community_count: int = result["communityCount"]
                modularities = result["modularities"]
                logger.info(
                    f"Performed graph clustering with {community_count} communities and modularities {modularities}"
                )
            finally:
                # Drop the projected graph
                await session.run(f"CALL gds.graph.drop('graph_{self.namespace}')")

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )

        async with self.async_driver.session(database=self.neo4j_database) as session:
            # Fetch community data
            result = await session.run(
                f"""
                MATCH (n:`{self.namespace}`)
                WITH n, n.communityIds AS communityIds, [(n)-[]-(m:`{self.namespace}`) | m.id] AS connected_nodes
                RETURN n.id AS node_id, n.source_id AS source_id, 
                       communityIds AS cluster_key,
                       connected_nodes
                """
            )

            # records = await result.fetch()

            max_num_ids = 0
            async for record in result:
                for index, c_id in enumerate(record["cluster_key"]):
                    node_id = str(record["node_id"])
                    source_id = record["source_id"]
                    level = index
                    cluster_key = str(c_id)
                    connected_nodes = record["connected_nodes"]

                    results[cluster_key]["level"] = level
                    results[cluster_key]["title"] = f"Cluster {cluster_key}"
                    results[cluster_key]["nodes"].add(node_id)
                    results[cluster_key]["edges"].update(
                        [
                            tuple(sorted([node_id, str(connected)]))
                            for connected in connected_nodes
                            if connected != node_id
                        ]
                    )
                    chunk_ids = source_id.split(GRAPH_FIELD_SEP)
                    results[cluster_key]["chunk_ids"].update(chunk_ids)
                    max_num_ids = max(
                        max_num_ids, len(results[cluster_key]["chunk_ids"])
                    )

            # Process results
            for k, v in results.items():
                v["edges"] = [list(e) for e in v["edges"]]
                v["nodes"] = list(v["nodes"])
                v["chunk_ids"] = list(v["chunk_ids"])
                v["occurrence"] = len(v["chunk_ids"]) / max_num_ids

            # Compute sub-communities (this is a simplified approach)
            for cluster in results.values():
                cluster["sub_communities"] = [
                    sub_key
                    for sub_key, sub_cluster in results.items()
                    if sub_cluster["level"] > cluster["level"]
                    and set(sub_cluster["nodes"]).issubset(set(cluster["nodes"]))
                ]

        return dict(results)

    async def index_done_callback(self):
        """
        Neo4j connections should remain open for subsequent queries.
        Commit work if needed; defer closing to explicit cleanup handled by the
        GraphRAG lifecycle (driver handles pooling internally).
        """
        return

    async def _debug_delete_all_node_edges(self):
        async with self.async_driver.session(database=self.neo4j_database) as session:
            try:
                # Delete all relationships in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`)-[r]-() DELETE r")

                # Delete all nodes in the namespace
                await session.run(f"MATCH (n:`{self.namespace}`) DELETE n")

                logger.info(
                    f"All nodes and edges in namespace '{self.namespace}' have been deleted."
                )
            except Exception as e:
                logger.error(f"Error deleting nodes and edges: {str(e)}")
                raise
