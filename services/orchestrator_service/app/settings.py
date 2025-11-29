from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ORCH_", case_sensitive=False)

    service_name: str = Field(default="orchestrator_service")
    log_level: str = Field(default="INFO")

    rag_service_url: str = Field(default="http://localhost:8003")
    yehia_service_url: str = Field(default="http://localhost:8101")
    shaer_service_url: str = Field(default="http://localhost:8102")
    scoring_service_url: str = Field(default="http://localhost:8107")
    enhancer_service_url: str = Field(default="http://localhost:8105")

    http_timeout_seconds: float = Field(default=120.0, ge=1.0)
    default_top_k_rag: int = Field(default=5, ge=1, le=20)
    max_retries_per_bayt: int = Field(default=3, ge=1, le=5)
    default_num_verses: int = Field(default=6, ge=1, le=12)
    allow_only_poetry_topics: bool = Field(default=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
