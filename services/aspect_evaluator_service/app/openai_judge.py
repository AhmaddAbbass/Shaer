from __future__ import annotations

import json
from dataclasses import dataclass

from openai import AsyncOpenAI

from .config import Settings
from .prompts import JUDGE_RESPONSE_SCHEMA, JUDGE_SYSTEM_PROMPTS


@dataclass
class JudgeResult:
    score_0_1: float
    notes: str


class AspectJudge:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.openai_api_key:
            self._client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    async def score(self, aspect: str, user_prompt: str) -> JudgeResult:
        if not self._client:
            raise RuntimeError("OpenAI judge is not configured")
        system_prompt = JUDGE_SYSTEM_PROMPTS.get(aspect, JUDGE_SYSTEM_PROMPTS["meaning"])
        response = await self._client.responses.create(
            model=self.settings.openai_model,
            temperature=self.settings.openai_temperature,
            response_format={"type": "json_schema", "json_schema": JUDGE_RESPONSE_SCHEMA},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            content = response.output[0].content[0].text  # type: ignore[index]
            data = json.loads(content)
            score = float(data["score_0_1"])
            notes = str(data.get("notes", "")).strip()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to parse OpenAI judge response: {exc}") from exc
        return JudgeResult(score_0_1=max(0.0, min(1.0, score)), notes=notes)
