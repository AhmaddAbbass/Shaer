from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration for the ashaar_meter_service.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ashaar_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Optional multiplier applied to the Ashaar score before clamping to [0, 1].",
    )
    log_level: str = Field(default="INFO", description="Log level for the service.")
    service_name: str = Field(default="ashaar_meter_service", description="Service name for logging.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
