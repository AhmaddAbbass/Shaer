from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

# Load environment early so CLI scripts do not need to call load_dotenv manually.
load_dotenv(BASE_DIR / ".env")


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y"}


@dataclass
class Settings:
    """Container for all runtime configuration needed by the RAG pipeline."""

    dataset_id: str = os.getenv(
        "HF_DATASET_ID", "Shaer-AI/ashaar-full-desc-preprocessed"
    )
    dataset_split: str = os.getenv("HF_DATASET_SPLIT", "train")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")

    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("RAG_ARTIFACT_DIR", "artifacts")).resolve()
    )
    cleaned_parquet: Path = field(init=False)

    chroma_mode: str = os.getenv("CHROMA_MODE", "persistent")  # persistent | http
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "shaer_poems")
    chroma_persist_dir: Path = field(init=False)
    chroma_batch_size: int = int(os.getenv("CHROMA_BATCH_SIZE", "128"))
    chroma_reset: bool = _bool_env("CHROMA_RESET", False)

    embed_model: str = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")

    neo4j_uri: str = os.getenv("NEO4J_URI", os.getenv("NEO4J_URL", "bolt://localhost:7687"))
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_database: str = os.getenv("NEO4J_DB", os.getenv("NEO4J_DATABASE", "neo4j"))

    # Quality thresholds
    min_arabic_ratio: float = float(os.getenv("MIN_ARABIC_RATIO", "0.6"))
    description_max_chars: int = int(os.getenv("DESCRIPTION_MAX_CHARS", "1200"))
    verse_similarity_threshold: float = float(os.getenv("VERSE_SIMILARITY_THRESHOLD", "0.9"))

    def __post_init__(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir = self.artifacts_dir / "chroma"
        self.cleaned_parquet = self.artifacts_dir / "clean_poems.parquet"
        if self.chroma_mode not in {"persistent", "http"}:
            raise ValueError(
                f"Invalid CHROMA_MODE={self.chroma_mode!r}; use 'persistent' or 'http'."
            )
