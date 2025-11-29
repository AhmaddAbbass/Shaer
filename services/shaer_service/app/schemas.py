from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, conint


class BaytGenerationRequest(BaseModel):
    poem_meter: str = Field(..., description="Target meter, e.g. 'البسيط'.")
    poem_description: str = Field(..., description="Ashaar-style prose description.")
    num_verses: conint(gt=0) = Field(..., description="Total verses in the poem.")
    sequence_number: conint(gt=0) = Field(..., description="1-based index for the requested bayt.")
    previous_verses: List[str] = Field(
        default_factory=list,
        description="Ordered list of previous bayts (صدر + عجز per line).",
    )
    poem_era: Optional[str] = Field(None, description="Optional era descriptor.")
    poet_name: Optional[str] = Field(None, description="Optional poet or style hint.")
    poem_title: Optional[str] = Field(None, description="Optional poem title.")
    extra_guidance: List[str] = Field(
        default_factory=list,
        description="Optional short bullet points appended under إرشادات مهمة for extra focus.",
    )


class BaytGenerationResponse(BaseModel):
    verse_text: str = Field(..., description="Generated bayt, صدر + عجز in a single line.")
    sequence_number: int = Field(..., description="Sequence number echoed from the request.")
