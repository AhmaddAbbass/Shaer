# Yehia Service Implementation Plan

Goal: FastAPI gateway to the Yehia RunPod endpoint with three main tools for the orchestrator:
- `/chat` (generic LLM chat)
- `/build-spec` (turn user query + optional RAG hits into `PoemSpec`)
- `/feedback` (score/explain a bayt vs. a `PoemSpec`)

## Inputs/Outputs
- Reuse shared models from `services/common_schemas/schemas.py`:
  - `PoemSpec`, `YehiaFeedback`, `RagSearchHit`, `LLMMessage`
- Service-local schemas for HTTP:
  - `ChatRequest { messages: list[LLMMessage] }` → `ChatResponse { text: str }`
  - `BuildSpecRequest { user_query: str, rag_hits?: list[RagSearchHit] }` → `BuildSpecResponse { spec: PoemSpec }`
  - `FeedbackRequest { verse_text: str, spec: PoemSpec }` → `FeedbackResponse { feedback: YehiaFeedback }`

## Prompting
- `prompt.py` should build Yehia messages for:
  - spec-building (system prompt to extract meter/theme/era/poet/style/description/num_verses)
  - feedback (system prompt to critique bayt vs spec)
  - generic chat (pass-through user messages with optional system preamble)
- Preserve the Arabic templates already described in `services.md` (and mirror Shaer prompt style).

## Config & Env
- Required envs (from `.env` sample):
  - `RUNPOD_API_KEY`
  - `YEHIA_ENDPOINT_ID`
- Optional defaults:
  - `YEHIA_RUNPOD_BASE_URL` (default `https://api.runpod.ai/v2`)
  - `YEHIA_MAX_NEW_TOKENS` (default e.g. 256)
  - `YEHIA_TEMPERATURE` (default 0.7)
  - `YEHIA_TOP_P` (default 0.9)
  - `YEHIA_RUNPOD_TIMEOUT` (default 120s)
  - `LOG_LEVEL`

## Client
- Similar to Shaer client: `client.py` builds payload `{ input: { messages, temperature, top_p, max_tokens } }` and calls `/{endpoint_id}/runsync`.
- Parse generated text defensively (choices/message/content/output_text/etc.).
- Fail fast if required envs are missing.

## API Layer
- FastAPI router with:
  - `POST /chat`
  - `POST /build-spec`
  - `POST /feedback`
- Health: `GET /health` (return `{"status": "ok"}` and maybe `{"env": "ok"}`).
- Dependency injection: singleton client via `get_client()`.

## Tests
- Add `tests/conftest.py` to set `sys.path`.
- Unit tests:
  - `test_prompt.py` (ensure prompts include meter/description, use defaults when missing).
  - `test_client.py` (extract text from various RunPod shapes).
  - `test_api.py` (FastAPI TestClient with a fake Yehia client to return stub text/spec/feedback).
- Mock RunPod calls; do not hit network in tests.

## Docker
- Base on `python:3.11-slim`.
- Copy `common_schemas` and service code; install deps from `requirements.txt`.
- Expose port `8001` (to avoid collisions with Shaer 8002, RAG 8003, meter 8004).
- Runtime: supply envs via `--env-file` or `-e`.

## README
- Document endpoints, env vars, run locally, docker build/run.
- Mention that inference is proxied to RunPod; no weights needed locally.
