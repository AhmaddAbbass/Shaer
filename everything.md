# Shaer Repository – Everything in One Place

This document fuses the vision in `project.md` with what is actually implemented in the repo so far. It covers architecture, data pipeline, services, models, utilities, and the current gaps.

## 1) Big Picture
- Goal: Arabic poetry platform powered by a specialized Shaer-7B poet, Yehia-7B for reasoning/feedback, RAG over ~118k poem metadata, and agentic workflows that chain services.
- User stories: generate new poems, fix poems, retrieve poems by topic/poet/meter, explain/analyze poems. The orchestrator LLM chooses tools in sequence (search -> build spec -> generate -> meter/feedback loops).
- No frontend yet; focus is on data, services, and agent tool surface.

## 2) Repository Layout (what exists vs. planned)
- Root docs: `project.md` (deep product+architecture narrative), `README.md` (quickstart for venv), `requirements.txt` (HF/transformers/vLLM stack).
- Services (under `services/`):
  - `common_schemas/`: shared Pydantic models for PoemSpec, Verse, Poem, Rag types, BaytMeterEval, YehiaFeedback, LLMMessage.
  - `rag_service/`: fully implemented FastAPI over Neo4j + Chroma (see section 4).
  - `shaer_client/`, `yehia_client/`, `meter_service/`, `ashaar_meter_service/`: scaffolds/docs outlining intended APIs (meter service now implemented, ashaar service newly added).
  - `services.md`: design notes for all services and how they should behave.
- Data/RAG stack (`RAG/`): complete pipeline to clean HF dataset, load Neo4j graph, build Chroma vectors. Includes docs (`RAG/README.md`, `rag.md`, `repo.md`), scripts, tests.
- Models (`models/`):
  - `how_to_run_models.py`: smoke test for RunPod OpenAI-compatible endpoints for Shaer and Yehia.
  - `judge_server.py`: vLLM-based scorer that returns 0–10 for bayts vs. descriptions (GPU service).
  - Placeholders for `yehia_model.py`, `shaer_model.py`, `scansion_model.py`, `attributes_model.py`.
  - Large `Ashaar_runtime/` with diacritizer, meter reward LSTM assets, configs, and classification/training artifacts.
- Utils (`utils/`): EDA on Ashaar dataset, diacritizer utilities, GRPO reward functions (meter/meaning/form) built on Ashaar_runtime assets.
- Frontend: directory exists but is empty.

## 3) Data & RAG Pipeline (implemented)
- Dataset: `Shaer-AI/ashaar-full-desc-preprocessed` from HF. Each row = poem metadata, verses, prose description.
- Cleaning rules (`RAG/shaer_rag/cleaning.py`):
  - Drop/flag placeholders (“ma a3rif” patterns), low Arabic-ratio, English-ish text, or descriptions that duplicate verses (fuzzy match).
  - Trim boilerplate intros; mark overly long descriptions for resummarization.
  - Build `poet_key`, `verse_preview`, and source domain.
- Pipeline (`shaer_rag/pipeline.py` + `scripts/build_pipeline.py`):
  1. Load HF dataset (optional `--limit`).
  2. Clean + persist parquet under `artifacts/`.
  3. Load all poems into Neo4j with constraints; link Poem -> Poet/Era/Meter/Theme/SourceSite.
  4. Index good descriptions into Chroma (doc id = poem_id) using SentenceTransformer embeddings.
  - Config via `RAG/.env` or defaults; supports Chroma persistent vs. HTTP.
- Query script: `scripts/query_chroma.py` for ad-hoc semantic search.
- Tests: `RAG/tests/` cover cleaning heuristics and parquet loader edge cases.
- Docker: `RAG/docker-compose.yml` spins up Neo4j + Chroma with volumes under `RAG/artifacts/`.

## 4) RAG Service (implemented)
- Location: `services/rag_service`.
- FastAPI app wiring (`app/main.py`) creates shared `ChromaClient` + `Neo4jClient`, configures logging, and exposes router.
- Config (`app/config.py`): env-driven Neo4j/Chroma/OpenAI embedding settings; auto-loads `.env` from repo root or `RAG/.env`.
- Chroma client (`app/chroma_client.py`):
  - Supports persistent or HTTP Chroma.
  - Uses OpenAI embeddings (requires `OPENAI_API_KEY` unless server has embeddings).
  - `search(query_text, top_k)` returns RagSearchResponse; `get_description(id)` helper for similarity endpoint.
- Neo4j client (`app/neo4j_client.py`):
  - `get_poem(poem_id)` returns RagPoemRecord with verses handling (list or newline string).
  - `filter_poems` supports contains-style filters on poet/meter/era/theme; returns RagSearchResponse hits.
  - `ping` for health check.
- API (`app/api.py`):
  - `GET /health` -> checks Chroma/Neo4j.
  - `GET /search?text&top_k` -> Chroma semantic search.
  - `GET /poem/{id}` -> Neo4j fetch (404 if missing).
  - `GET /filter?poet_name&meter&era&theme&limit` -> metadata filters.
  - `GET /similar/{id}` -> uses stored description (from Chroma or verse fallback) to find neighbors, drops self-hit.
- Schemas (`app/schemas.py`): HealthResponse, FilterResponse, SimilarResponse wrap shared Rag models.
- Tests (`services/rag_service/tests`): health/search/poem endpoints with dummy clients.
- Docker/usage: see `services/rag_service/README.md`; run `uvicorn app.main:app --reload --port 8003`.

## 5) Other Services (planned, stubbed)
- `services/common_schemas`: fully defined shared models; service-level schemas should compose these.
- `services/shaer_client`: intended to build exact SFT-style prompt and call Shaer RunPod; endpoints `/generate-bayt` (and optional `/generate-poem`). Files are empty today; tests outline expected behavior.
- `services/yehia_client`: intended to expose `/chat`, `/build-spec`, `/feedback` using Yehia prompts and RunPod. Files empty; tests/docs describe target shapes.
- `services/meter_service`: BiLSTM-based `/eval-bayt` implementation (now populated).
- `services/ashaar_meter_service`: Ashaar structural scoring service (new).
- `services/services.md` documents the design, env vars, and docker-compose expectations for all services.

## 6) Models and Evaluation
- `judge_server.py`: FastAPI + vLLM scorer service.
  - Loads Yehia-7B (default) via vLLM; forces GPU 1 by env.
  - Endpoint `POST /score_batch` accepts items with verse, description, optional previous_verses; builds chat prompts and parses 0–10 score from generation.
  - Uses deterministic sampling (temperature 0) and retries GPU init with adjusted utilization on OOM.
- `models/how_to_run_models.py`: CLI to test RunPod OpenAI-compatible endpoints for Shaer/Yehia or fall back to RunPod `/runsync`.
- `models/Ashaar_runtime`: now split into `bilstm_only/` (BiLSTM classifier assets + reward helpers) and `ashaar_only/` (Ashaar structural analysis stack, diacritizer code, configs, pretrained weights).

## 7) Utils
- `utils/data/data_view.py`: EDA script for the older Ashaar dataset; prints distribution stats and example rows (contains a placeholder HF token to replace).
- `utils/evaluation/diacritizer.py` and `diacritizer_caml.py`: Fine-Tashkeel/ByT5 and CAMeL-based diacritization pipelines with helper to attach diacritized verse to payloads.
- `utils/rewards/`: GRPO-friendly reward functions:
  - `form_reward.py`: helpers to extract text from completions and basic form checks.
  - `meter_reward.py`: loads BiLSTM meter classifier assets, returns meter probability or target-meter score; patches TF/JL quirks.
  - `meaning_reward.py`: (not deeply wired here) placeholder for semantic reward.
  - Tests under `utils/rewards/test_*` target reward functions.

## 8) Tests and Quality Checks
- Service tests exist only for `rag_service`. Other service test files are empty stubs.
- RAG pipeline tests cover cleaning heuristics, dataset loading, and vector helpers.
- Reward/diacritizer tests cover utility behaviors.
- No CI config in repo; no root docker-compose yet (only RAG-level compose).

## 9) How to Run Key Pieces
- Base environment: `python -m venv .venv`, activate, `pip install -r requirements.txt`.
- RAG pipeline: `cd RAG && pip install -r requirements-dev.txt && docker compose up -d && python scripts/build_pipeline.py --limit 500`.
- RAG service: `cd services/rag_service && uvicorn app.main:app --reload --port 8003` (ensure Neo4j/Chroma reachable and `OPENAI_API_KEY` set if needed).
- Judge server: `python judge_server.py` with GPU access and `YEHIA_VLLM_MODEL_PATH` if overriding default.
- RunPod endpoint smoke: set `RUNPOD_API_KEY`, `YEHIA_ENDPOINT_ID`, `SHAER_ENDPOINT_ID`, then `python models/how_to_run_models.py`.

## 10) Gaps / Next Steps (observed from codebase)
- Implement the Yehia/Shaer/meter services per `services.md` (currently empty files).
- Add root `docker-compose.yml` to orchestrate all services plus Neo4j/Chroma.
- Wire orchestrator/agent tooling (not present) to chain services.
- Replace placeholder HF token in `utils/data/data_view.py` with env-based loading.
- Consider trimming large Ashaar_runtime artifacts or document their usage to avoid confusion.
- Add CI and linting/formatting; expand tests beyond RAG service/pipeline.

This doc should give you an end-to-end understanding of what is present and what is still conceptual so you can merge the planned workflows with the implemented RAG layer and utilities.
