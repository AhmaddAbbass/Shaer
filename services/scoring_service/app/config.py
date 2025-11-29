from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCORING_", case_sensitive=False)

    service_name: str = Field(default="scoring_service")
    log_level: str = Field(default="INFO")
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    meter_base_url: str = Field(default="http://localhost:8104", description="Meter service URL")
    aspect_eval_base_url: str = Field(default="http://localhost:8106", description="Aspect evaluator URL")
    http_timeout_seconds: float = Field(default=30.0, ge=1.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
