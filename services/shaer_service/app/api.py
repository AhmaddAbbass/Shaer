from fastapi import APIRouter, Depends

from .client import ShaerRunpodClient, get_client
from .schemas import BaytGenerationRequest, BaytGenerationResponse

router = APIRouter()


@router.post("/generate-bayt", response_model=BaytGenerationResponse)
async def generate_bayt(
    request: BaytGenerationRequest,
    client: ShaerRunpodClient = Depends(get_client),
) -> BaytGenerationResponse:
    verse_text = await client.generate_bayt(request)
    return BaytGenerationResponse(
        verse_text=verse_text,
        sequence_number=request.sequence_number,
    )
