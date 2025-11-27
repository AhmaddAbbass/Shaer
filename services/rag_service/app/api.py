
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from services.common_schemas.schemas import RagPoemRecord, RagSearchResponse

from .chroma_client import ChromaClient
from .logging import get_logger
from .neo4j_client import Neo4jClient
from .schemas import FilterResponse, HealthResponse, SimilarResponse

router = APIRouter()
logger = get_logger("rag_service.api")


# Dependency helpers -----------------------------------------------------
def get_chroma_client(request: Request) -> ChromaClient:
    return request.app.state.chroma_client


def get_neo4j_client(request: Request) -> Neo4jClient:
    return request.app.state.neo4j_client


# Routes -----------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health(
    chroma: ChromaClient = Depends(get_chroma_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    return HealthResponse(
        status="ok",
        chroma="ok" if chroma.ping() else "down",
        neo4j="ok" if neo4j.ping() else "down",
    )


@router.get("/search", response_model=RagSearchResponse)
async def search(
    text: str = Query(..., description="Arabic query text to search for"),
    top_k: int = Query(5, gt=0, le=50),
    chroma: ChromaClient = Depends(get_chroma_client),
):
    try:
        return chroma.search(text, top_k=top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/poem/{poem_id}", response_model=RagPoemRecord)
async def get_poem(
    poem_id: str,
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    poem = neo4j.get_poem(poem_id)
    if not poem:
        raise HTTPException(status_code=404, detail="Poem not found")
    return poem


@router.get("/filter", response_model=FilterResponse)
async def filter_poems(
    poet_name: str | None = Query(default=None),
    meter: str | None = Query(default=None),
    era: str | None = Query(default=None),
    theme: str | None = Query(default=None),
    limit: int = Query(default=20, gt=0, le=100),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    hits = neo4j.filter_poems(
        poet_name=poet_name,
        meter=meter,
        era=era,
        theme=theme,
        limit=limit,
    ).hits
    return FilterResponse(hits=hits)


@router.get("/similar/{poem_id}", response_model=SimilarResponse)
async def similar_poems(
    poem_id: str,
    top_k: int = Query(default=5, gt=1, le=50),
    chroma: ChromaClient = Depends(get_chroma_client),
    neo4j: Neo4jClient = Depends(get_neo4j_client),
):
    """
    Rough similarity search:
    - Try to fetch the description from Chroma by id.
    - Fallback: use the first couple of verses from Neo4j as a query.
    """
    description = chroma.get_description(poem_id)

    if not description:
        poem = neo4j.get_poem(poem_id)
        if poem and poem.verses:
            description = " ".join(poem.verses[:2])

    if not description:
        raise HTTPException(status_code=404, detail="Poem description not found")

    try:
        res = chroma.search(description, top_k=top_k + 1)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Drop self-hit if present
    filtered_hits = [
        h for h in res.hits if str(h.poem_id) != str(poem_id)
    ][:top_k]
    return SimilarResponse(hits=filtered_hits)
