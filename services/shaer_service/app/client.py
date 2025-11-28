from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .config import Settings, get_settings
from .prompt import build_messages
from .schemas import BaytGenerationRequest


class ShaerRunpodClient:
    """Lightweight HTTP client that proxies requests to the Shaer RunPod endpoint."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = logging.getLogger("shaer_service.runpod_client")

    async def generate_bayt(self, request: BaytGenerationRequest) -> str:
        messages = build_messages(request)
        payload = {
            "input": {
                "messages": messages,
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "max_tokens": self.settings.max_new_tokens,
            }
        }
        url = f"{self.settings.runpod_base_url}/{self.settings.shaer_endpoint_id}/runsync"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.runpod_api_key}",
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        verse_text = self._extract_text(data)
        self.logger.debug("RunPod generation complete for seq=%s", request.sequence_number)
        return verse_text

    def _extract_text(self, response_data: Dict[str, Any]) -> str:
        output = response_data.get("output")
        candidates: list[str] = []

        def push(value: Any) -> None:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
            elif isinstance(value, list):
                # Some responses return a list of tokens; join them.
                tokens = [t for t in value if isinstance(t, str)]
                if tokens:
                    joined = "".join(tokens).strip()
                    if joined:
                        candidates.append(joined)

        def walk(value: Any) -> None:
            if isinstance(value, str):
                push(value)
            elif isinstance(value, dict):
                for key in ("text", "generated_text", "output_text", "content", "response", "result"):
                    if key in value:
                        walk(value[key])
                choices = value.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict):
                            walk(choice.get("message"))
                            walk(choice.get("text"))
                            walk(choice.get("tokens"))
                            # Some shapes nest tokens under choices[...] -> tokens directly
                            if "tokens" in choice:
                                walk(choice["tokens"])
                message = value.get("message")
                if isinstance(message, dict):
                    walk(message.get("content"))
            elif isinstance(value, list):
                # Try joining list of tokens first
                push(value)
                for item in value:
                    walk(item)

        walk(output)
        if candidates:
            return candidates[0]

        self.logger.error("Unexpected RunPod response payload: %s", response_data)
        raise RuntimeError("RunPod response did not include generated text.")


_CLIENT: ShaerRunpodClient | None = None


def get_client() -> ShaerRunpodClient:
    global _CLIENT
    if _CLIENT is None:
        settings = get_settings()
        _CLIENT = ShaerRunpodClient(settings)
    return _CLIENT
