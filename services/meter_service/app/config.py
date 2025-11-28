from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration for meter_service.
    Paths default to the BiLSTM assets under models/Ashaar_runtime.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    meter_model_path: str = Field(
        default=str(
            Path(__file__).resolve().parents[3]
            / "models"
            / "Ashaar_runtime"
            / "bilstm_only"
            / "bilstm_model"
            / "poem_meter_bilstm_v2.h5"
        ),
        description="Path to the saved BiLSTM meter model (H5 or SavedModel dir)",
    )
    meter_label_encoder_path: str = Field(
        default=str(Path(__file__).resolve().parents[3] / "models" / "Ashaar_runtime" / "bilstm_only" / "bilstm_model" / "training" / "meter_label_encoder.joblib"),
        description="Path to the label encoder joblib file",
    )
    meter_vocab_config_path: str = Field(
        default=str(Path(__file__).resolve().parents[3] / "models" / "Ashaar_runtime" / "bilstm_only" / "bilstm_model" / "training" / "bilstm_vocab_config.json"),
        description="Path to the vocab config JSON used by the BiLSTM",
    )

    meter_score_threshold: int = Field(default=80, ge=0, le=100, description="Threshold to mark a bayt as on_meter")
    meter_max_top: int = Field(default=3, ge=1, description="Top-N predictions to mention in notes if useful")

    log_level: str = Field(default="INFO", description="Log level for the service")
    service_name: str = Field(default="meter_service", description="Service name for logging")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
