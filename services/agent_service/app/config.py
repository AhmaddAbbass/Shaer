from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", case_sensitive=False)

    service_name: str = Field(default="agent_service")
    log_level: str = Field(default="INFO")
    orchestrator_base_url: str = Field(
        default="http://localhost:8000", description="Base URL for orchestrator_service"
    )
    http_timeout_seconds: float = Field(default=30.0, ge=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
