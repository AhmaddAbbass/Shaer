from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the agent/orchestrator service."""

    model_config = SettingsConfigDict(env_prefix="AGENT_", case_sensitive=False)

    service_name: str = Field(default="agent_service")
    log_level: str = Field(default="INFO")

    # Base URLs for the downstream services
    yehia_base_url: str = Field(default="http://localhost:8101", description="Base URL for Yehia service")
    shaer_base_url: str = Field(default="http://localhost:8102", description="Base URL for Shaer service")
    rag_base_url: str = Field(default="http://localhost:8003", description="Base URL for RAG service")
    meter_base_url: str = Field(default="http://localhost:8104", description="Base URL for meter service")

    http_timeout_seconds: float = Field(default=30.0)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    compose_temperature: float = Field(default=0.7)

    default_meter: str = Field(default="الكامل")
    default_theme: str = Field(default="شعر وجداني")
    default_num_verses: int = Field(default=6, ge=1, le=12)
    max_bayt_retries: int = Field(default=3, ge=1, le=5)

    explanation_system_prompt: str = Field(
        default=(
            "أنت يحيى، معلم شعر عربي يساعد المستخدم على فهم المفاهيم العروضية "
            "وشرح الأبيات بأسلوب واضح ومختصر مع أمثلة عند الحاجة."
        )
    )
    openai_api_key: Optional[str] = Field(default=None, description="API key used to call GPT-4o for tool selection")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model used for intent/tool selection")
    tool_selection_system_prompt: str = Field(
        default=(
            "أنت منسق مهام لشاعر آلي. أمامك أدوات داخلية يمكن استخدامها: generate (لإنشاء قصيدة جديدة)، "
            "fix (لإصلاح أبيات موجودة)، search (لعرض قصائد من المكتبة)، explain (للإجابة عن أسئلة نظرية). "
            "اقرأ رسالة المستخدم وحدد أفضل أداة واحدة فقط، ثم أعد JSON على شكل "
            '{"mode": "generate", "reason": "لماذا اخترت الأداة"} بالاعتماد على نيته.'
        )
    )


class ServiceURLs(BaseModel):
    yehia: str
    shaer: str
    rag: str
    meter: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_service_urls() -> ServiceURLs:
    settings = get_settings()
    return ServiceURLs(
        yehia=settings.yehia_base_url,
        shaer=settings.shaer_base_url,
        rag=settings.rag_base_url,
        meter=settings.meter_base_url,
    )
