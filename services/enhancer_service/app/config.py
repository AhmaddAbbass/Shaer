from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENHANCER_", case_sensitive=False)

    service_name: str = Field(default="enhancer_service")
    log_level: str = Field(default="INFO")
    shaer_base_url: str = Field(default="http://localhost:8102", description="Base URL for shaer_service")
    request_timeout_seconds: float = Field(default=30.0, ge=1.0)

    max_extra_guidance: int = Field(default=2, ge=0, le=3, description="Maximum custom bullets appended to إرشادات مهمة")
    enable_description_tightening: bool = Field(default=True)
    meter_focus_threshold: int = Field(default=80, ge=0, le=100)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
