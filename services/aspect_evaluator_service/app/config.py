from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ASPECT_EVAL_", case_sensitive=False)

    service_name: str = Field(default="aspect_evaluator_service")
    log_level: str = Field(default="INFO")
    yehia_base_url: str = Field(default="http://localhost:8101", description="Base URL for yehia_service")
    request_timeout_seconds: float = Field(default=30.0, ge=1.0)

    openai_api_key: str | None = Field(default=None, description="API key for OpenAI scoring")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_temperature: float = Field(default=0.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
