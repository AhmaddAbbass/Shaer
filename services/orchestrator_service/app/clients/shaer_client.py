from __future__ import annotations

from typing import List

from services.common_schemas.schemas import PoemSpec

from .base import BaseClient


class ShaerServiceClient(BaseClient):
    async def generate_bayt(
        self,
        spec: PoemSpec,
        sequence_number: int,
        previous_verses: List[str],
    ) -> str:
        payload = {
            "poem_meter": spec.poem_meter,
            "poem_description": spec.poem_description,
            "num_verses": spec.num_verses,
            "sequence_number": sequence_number,
            "previous_verses": previous_verses,
            "poem_era": spec.poem_era,
            "poet_name": spec.poet_name,
            "poem_title": spec.poem_title,
        }
        response = await self._client.post("/generate-bayt", json=payload)
        response.raise_for_status()
        data = response.json()
        return str(data.get("verse_text", "")).strip()
