from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Optional

from openai import AsyncOpenAI


@dataclass
class ToolSelectionResult:
    mode: str
    reason: str | None = None


class ToolSelector:
    """Lightweight helper that asks GPT-4o which tool/intent to run first."""

    def __init__(self, api_key: str, model: str, system_prompt: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._system_prompt = system_prompt

    async def select_mode(self, user_message: str, allowed_modes: Iterable[str]) -> Optional[ToolSelectionResult]:
        if not user_message.strip():
            return None

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_message.strip()},
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=messages,
        )
        if not response.choices:
            return None
        content = (response.choices[0].message.content or "").strip()
        if not content:
            return None

        mode = None
        reason = None
        try:
            data = json.loads(content)
            candidate = str(data.get("mode", "")).strip().lower()
            if candidate:
                mode = candidate
            reason = str(data.get("reason")) if data.get("reason") else None
        except json.JSONDecodeError:
            # fallback: look for the first word matching allowed modes
            lowered = content.lower()
            for m in allowed_modes:
                if m in lowered:
                    mode = m
                    break
            reason = content if mode else None

        if not mode:
            return None

        allowed_lower = {m.lower() for m in allowed_modes}
        if mode not in allowed_lower:
            return None

        return ToolSelectionResult(mode=mode, reason=reason)
