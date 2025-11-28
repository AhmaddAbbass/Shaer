from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _get_env(name: str, default: str | None = None) -> str:
    val = os.getenv(name, default)
    return val if val is not None else ""


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    runpod_base_url: str = _get_env("YEHIA_RUNPOD_BASE_URL", "https://api.runpod.ai/v2")
    yehia_endpoint_id: str = _get_env("YEHIA_ENDPOINT_ID", "")
    runpod_api_key: str = _get_env("RUNPOD_API_KEY", "")
    max_new_tokens: int = _get_int("YEHIA_MAX_NEW_TOKENS", 256)
    temperature: float = _get_float("YEHIA_TEMPERATURE", 0.7)
    top_p: float = _get_float("YEHIA_TOP_P", 0.9)
    request_timeout: float = _get_float("YEHIA_RUNPOD_TIMEOUT", 120.0)
    log_level: str = _get_env("YEHIA_LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.yehia_endpoint_id:
        raise RuntimeError("YEHIA_ENDPOINT_ID environment variable must be set.")
    if not settings.runpod_api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable must be set.")
    return settings
