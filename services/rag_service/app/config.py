
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_env_files() -> List[str]:
    """
    Resolve a couple of reasonable .env locations:
    - repo root (.env)
    - RAG/.env (matches existing RAG pipeline)
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    candidates = [
        repo_root / ".env",
        repo_root / "RAG" / ".env",
    ]
    return [str(p) for p in candidates if p.exists()]


class Settings(BaseSettings):
    """
    Central configuration for the RAG service.
    Values are pulled from environment variables or optional .env files.
    """

    model_config = SettingsConfigDict(
        env_file=_default_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Neo4j
    neo4j_uri: str = Field(
        default="bolt://localhost:7687", description="Neo4j bolt endpoint"
    )
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j", description="Neo4j password")
    neo4j_db: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_enable_routing: bool = Field(
        default=False, description="Enable routing driver (for clustered setups)"
    )

    # Chroma
    chroma_mode: str = Field(
        default="persistent",
        description="Chroma client mode: 'persistent' or 'http'",
    )
    chroma_host: str = Field(default="localhost", description="Chroma HTTP host")
    chroma_port: int = Field(default=8000, description="Chroma HTTP port")
    chroma_persist_dir: str = Field(
        default=str(Path(__file__).resolve().parents[3] / "RAG" / "artifacts" / "chroma"),
        description="Path to persistent Chroma storage",
    )
    chroma_collection: str = Field(
        default="shaer_poems", description="Collection name to query"
    )
    chroma_reset: bool = Field(
        default=False, description="If true, drop and recreate the collection on boot"
    )

    # Embeddings (OpenAI-backed by default)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name used for query embeddings",
    )

    # Service settings
    log_level: str = Field(default="INFO", description="Logging level")
    service_name: str = Field(default="rag_service", description="Service name")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
