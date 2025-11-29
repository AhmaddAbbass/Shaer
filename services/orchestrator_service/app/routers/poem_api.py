from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from services.common_schemas.schemas import PoemSpec

from ..context import ServiceRegistry
from ..schemas import AgentStep, GeneratePoemRequest, GeneratePoemResponse, VerseWithScore
from ..settings import Settings
from ..utils.feedback import summarize_feedback
from ..utils.safety import guard_poetry_topic

router = APIRouter(prefix="/poem", tags=["poem"])


def get_registry(request: Request) -> ServiceRegistry:
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise RuntimeError("Orchestrator registry not initialized")
    return registry


def _ensure_num_verses(spec: PoemSpec, desired: int | None, default_num: int) -> PoemSpec:
    num = desired or spec.num_verses or default_num
    return spec.model_copy(update={"num_verses": num})


@router.post("/generate", response_model=GeneratePoemResponse)
async def generate_poem(
    payload: GeneratePoemRequest,
    registry: ServiceRegistry = Depends(get_registry),
) -> GeneratePoemResponse:
    settings: Settings = registry.settings
    guard_poetry_topic(payload.user_query, settings)

    rag_hits = []
    agent_trace: list[AgentStep] = []
    if payload.use_rag:
        try:
            rag_hits = (
                await registry.rag_client.search(payload.user_query, settings.default_top_k_rag)
            ).hits
            agent_trace.append(
                AgentStep(
                    step=len(agent_trace) + 1,
                    agent="orchestrator",
                    tool="rag_service.search",
                    summary=f"استرجاع {len(rag_hits)} نتائج للإلهام.",
                )
            )
        except Exception:
            rag_hits = []

    spec = await registry.yehia_client.build_spec(payload.user_query, rag_hits)
    spec = _ensure_num_verses(spec, payload.desired_num_verses, settings.default_num_verses)
    agent_trace.append(
        AgentStep(
            step=len(agent_trace) + 1,
            agent="yehia_service",
            tool="build-spec",
            summary=f"تحديد مواصفات القصيدة ({spec.poem_meter}, {spec.num_verses} أبيات).",
        )
    )

    verses: list[VerseWithScore] = []
    previous_verses: list[str] = []

    for idx in range(1, spec.num_verses + 1):
        verse_text = await registry.shaer_client.generate_bayt(spec, idx, previous_verses)
        agent_trace.append(
            AgentStep(
                step=len(agent_trace) + 1,
                agent="shaer_service",
                tool="generate-bayt",
                summary=f"توليد البيت رقم {idx}.",
            )
        )
        scoring = await registry.scoring_client.score_bayt(
            verse_text, spec, previous_verses, use_openai_scoring=payload.use_openai_scoring
        )
        attempts = 1
        while not scoring.passed and attempts < settings.max_retries_per_bayt:
            summary = summarize_feedback(scoring)
            agent_trace.append(
                AgentStep(
                    step=len(agent_trace) + 1,
                    agent="scoring_service",
                    tool="score",
                    summary=f"البيت {idx} لم يجتز، ملخص الملاحظات: {' / '.join(summary)[:120]}",
                )
            )
            verse_text = await registry.enhancer_client.enhance_bayt(
                spec,
                previous_verses,
                idx,
                verse_text,
                scoring,
                summary,
            )
            scoring = await registry.scoring_client.score_bayt(
                verse_text, spec, previous_verses, use_openai_scoring=payload.use_openai_scoring
            )
            attempts += 1
        agent_trace.append(
            AgentStep(
                step=len(agent_trace) + 1,
                agent="scoring_service",
                tool="score",
                summary=f"البيت {idx} معتمد (محاولات {attempts}).",
            )
        )
        previous_verses.append(verse_text)
        verses.append(VerseWithScore(text=verse_text, scoring=scoring))

    return GeneratePoemResponse(
        spec=spec,
        verses=verses,
        rag_hits=[h.poem_id for h in rag_hits] if rag_hits else None,
        agent_trace=agent_trace or None,
    )
