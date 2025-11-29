
1. `services/common_schemas`
2. `services/shaer_client`
3. `services/yehia_client`
4. `services/rag_service`
5. `services/meter_service`
6. `services/ashaar_meter_service`
7. `services/enhancer_service`
8. `services/scoring_service`
9. `services/aspect_evaluator_service`
10. `services/orchestrator_service`
11. `services/agent_service` (proxy to orchestrator)
12. Root-level `docker-compose.yml` + basic tests.

---

## 1. `services/common_schemas/`

### Files

* `__init__.py`

  * Just makes the package importable.

* `schemas.py`

  * **Single source of truth for data types** used across services.
  * Should define lightweight classes / models for:

    * `PoemSpec`
    * `Verse`
    * `Poem`
    * `BaytMeterEval`
    * `YehiaFeedback`
    * `RagSearchRequest`, `RagSearchHit`, `RagSearchResponse`, `RagPoemRecord`
    * `LLMMessage` (role + content) – used by Yehia/Shaer.

**Goal for the team:**
Write concise models that match our verbal spec so every service can import and agree on field names & shapes.

---

## 2. `services/shaer_client/` – Shaer gateway

**High-level role:**
Given `PoemSpec + previous verses + sequence_number`, build the **exact SFT-style prompt** and call the deployed Shaer model on RunPod. Return bayts or full poems.

### Files & responsibilities

* `app/__init__.py`

  * Marks package; can expose a `create_app()` helper if you want.

* `app/main.py`

  * Creates the web app instance (FastAPI / Flask).
  * Includes router from `api.py`.
  * Defines a basic `GET /health` endpoint for health checks.

* `app/api.py`

  * Declares HTTP routes:

    * `POST /generate-bayt`
    * (optionally) `POST /generate-poem`
  * For each route:

    * Parse JSON into models from `schemas.py`.
    * Call functions from `prompt.py` to **build messages**.
    * Call `client.py` to talk to RunPod.
    * Wrap the generated text into a `Verse` / `Poem` object and return.

* `app/schemas.py`

  * “HTTP-facing models” for this service.
  * Likely very thin wrappers over `PoemSpec`, `Verse`, `Poem` imported from `common_schemas.schemas`.
  * Example structures:

    * `GenerateBaytRequest` → { `spec`, `previous_verses`, `sequence_number` }.
    * `GenerateBaytResponse` → { `verse` }.

* `app/client.py`

  * Low-level HTTP client to the **Shaer RunPod endpoint**.
  * Responsibilities:

    * Build the JSON body expected by RunPod: includes `messages`, `temperature`, `max_tokens`, etc.
    * Add authentication headers (`SHAER_RUNPOD_API_KEY`).
    * Handle HTTP errors / timeouts and surface clean errors to `api.py`.
    * Return plain generated text to the caller (no business logic).

* `app/config.py`

  * Central place to read environment variables and default settings:

    * `SHAER_RUNPOD_API_BASE`
    * `SHAER_RUNPOD_API_KEY`
    * `SHAER_RUNPOD_MODEL_ID` (if needed)
    * `SHAER_TEMPERATURE`
    * `SHAER_MAX_TOKENS`
    * `SHAER_TIMEOUT_SECONDS`
  * Should expose a simple config object that other files import.

* `app/logging.py`

  * Configure log format and level.
  * Make a function like `get_logger(name)` that other modules use.
  * Useful to log every request & response metadata (not the full text).

* `app/prompt.py`

  * **Very important file.** Owns the Shaher prompt style.
  * Contains:

    * Constant `SHAER_SYSTEM_PROMPT` – exactly the system message used in SFT.
    * Helper to format previous verses into the “الأبيات السابقة في القصيدة” block.
    * Helper to build the “إرشادات مهمة” section (fixed bullets + maybe optional hints).
    * Core function:

      * `build_shaer_messages(spec, previous_verses, sequence_number)`
        → returns `list[LLMMessage]` exactly matching your dataset, so the RunPod endpoint just applies `chat_template` and generates.

* `tests/test_prompt.py`

  * Check that `build_shaer_messages`:

    * Contains the right meter/theme/era/poet/description.
    * Formats previous verses as expected.
    * Matches the dataset conventions (e.g., “لا توجد أبيات سابقة…” for first bayt).

* `tests/test_generate_bayt.py`

  * Integration-ish test (can use a fake client during dev) that:

    * Calls `/generate-bayt` with a minimal `PoemSpec` and empty `previous_verses`.
    * Asserts the response shape: object with `.verse.text` and `.verse.index`.

* `Dockerfile`

  * Build a container that:

    * Uses a Python base image.
    * Installs this service’s dependencies (FastAPI + http client + common_schemas).
    * Sets the CMD to run `uvicorn app.main:app` (or similar).
  * One image per service.

* `README.md`

  * Short doc for the team:

    * Purpose of the service.
    * Routes with input/output examples.
    * Env vars it expects.
    * How to run locally vs via Docker.

---

## 3. `services/yehia_client/` – Yehia gateway

**High-level role:**
Talk to Yehia on RunPod and provide higher-level features:

* generic `/chat`,
* `/build-spec` (construct `PoemSpec` from user query + RAG hits),
* `/feedback` (give `YehiaFeedback` for a bayt).

### Files & responsibilities

Structure is parallel to `shaer_client`:

* `app/__init__.py`

* `app/main.py`

  * Create app, add routes, `GET /health`.

* `app/api.py`

  * Define endpoints:

    * `POST /chat`
    * `POST /build-spec`
    * `POST /feedback`
  * For each:

    * Validate input via `schemas.py`.
    * Build messages with `prompt.py` helpers.
    * Call `client.py`.
    * Post-process Yehia’s output into `PoemSpec` or `YehiaFeedback`.

* `app/schemas.py`

  * HTTP request/response models for:

    * Chat (`ChatRequest`, `ChatResponse`).
    * Spec building (`BuildSpecRequest` { `user_query`, optional list of `RagSearchHit` }).
    * Feedback (`FeedbackRequest` { `verse_text`, `spec` }).
  * All reuse core types from `common_schemas.schemas`.

* `app/client.py`

  * Low-level HTTP client to **Yehia RunPod endpoint**.
  * Same pattern as Shaer:

    * Build JSON with `messages`, `temp`, `max_tokens`.
    * Add `YEHIA_RUNPOD_API_KEY`.
    * Return raw generated text.

* `app/config.py`

  * Reads:

    * `YEHIA_RUNPOD_API_BASE`
    * `YEHIA_RUNPOD_API_KEY`
    * `YEHIA_RUNPOD_MODEL_ID`
    * `YEHIA_TEMPERATURE`
    * `YEHIA_MAX_TOKENS`
    * `YEHIA_TIMEOUT_SECONDS`

* `app/logging.py`

  * Same idea as Shaer.

* `app/prompt.py`

  * Yehia prompt templates and builders:

    * System messages for spec building (“أنت مخطط قصائد…”).
    * System messages for feedback (“أنت ناقد شعري…”).
    * `build_spec_messages(user_query, rag_hits)` → messages for Yehia.
    * `build_feedback_messages(verse_text, spec)` → messages for Yehia to evaluate a bayt.

* `tests/test_chat.py`

  * Smoke test that `/chat` returns a string.

* `tests/test_build_spec.py`

  * Given a simple `user_query` and some fake `rag_hits`, ensure the service returns a sensible `PoemSpec` structure (all required fields exist, have types etc.).

* `tests/test_feedback.py`

  * Given a verse + spec, check that `/feedback` returns a `YehiaFeedback` with `ok` boolean and a text.

* `Dockerfile` + `README.md`

  * Same pattern as Shaer: dependencies, run command, docs.

---

## 4. `services/rag_service/` – RAG API

**High-level role:**
Offer clean endpoints over Neo4j + Chroma:

* search by text,
* get full poem by ID,
* filter by metadata,
* (optionally) similar-by-id.

### Files & responsibilities

* `app/__init__.py`

* `app/main.py`

  * Create app, connect to DBs on startup (or lazy inside clients).
  * Mount routers from `api.py`.
  * `GET /health` should check DB connectivity if possible.

* `app/api.py`

  * Endpoints:

    * `GET /search?text=&top_k=`
    * `GET /poem/{poem_id}`
    * `GET /filter?poet_name=&meter=&era=&theme=`
    * `GET /similar/{poem_id}` (optional)
  * Each:

    * Parses query params.
    * Calls `chroma_client` or `neo4j_client`.
    * Formats results using `schemas.py` models.

* `app/schemas.py`

  * HTTP models for:

    * `SearchResponse` containing `list[RagSearchHit]`.
    * `PoemResponse` containing `RagPoemRecord`.

* `app/neo4j_client.py`

  * Encapsulate all interactions with Neo4j:

    * Connection creation using `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DB`.
    * Functions:

      * `get_poem(poem_id)`
      * `filter_poems(poet, meter, era, theme, limit)`
  * Returns Python dicts that `api.py` converts to schemas.

* `app/chroma_client.py`

  * Encapsulate calls to Chroma:

    * For HTTP mode: host/port from env.
    * For persistent local mode: path from env.
    * Function:

      * `search(text, top_k)` → returns list of hits with metadata.

* `app/config.py`

  * Reads:

    * `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DB`
    * `CHROMA_MODE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_RESET`
    * optional `RAG_LOG_LEVEL`

* `app/logging.py`

  * Standard logging utilities.

* `tests/test_search.py`

  * Use either a test Neo4j/Chroma instance or mocks.
  * Test `/search` returns correct structure.

* `tests/test_get_poem.py`

  * Test `/poem/{poem_id}` returns `RagPoemRecord`.

* `Dockerfile` + `README.md`

  * Docker: install DB drivers + HTTP libs.
  * README: how to run along with `RAG/docker-compose` (Neo4j + Chroma services).

---

## 5. Meter services – bayt meter + Ashaar checks

We now expose two focused services:

1. `services/meter_service/` – wraps the BiLSTM classifier (keras/tf) to return `BaytMeterEval`.
2. `services/ashaar_meter_service/` – wraps Ashaar structural scoring to return float similarity scores + notes.

### 5.1 `services/meter_service/`

**High-level role:** Given `verse_text` + `target_meter`, return `BaytMeterEval`.

### Files & responsibilities

* `app/__init__.py`
* `app/main.py` – create app, load scansion assets on startup, expose `/health`.
* `app/api.py` – `POST /eval-bayt` returning `BaytMeterEval`.
* `app/schemas.py` – HTTP request/response wrappers using `BaytMeterEval`.
* `app/scansion.py` – loads BiLSTM assets, scores, and produces notes.
* `app/config.py` – env paths to BiLSTM weights, score threshold, log level.
* `app/logging.py` – common logging helpers.
* `tests/test_eval_bayt.py`, `tests/test_scansion.py`.
* `Dockerfile` + `README.md`.

### 5.2 `services/ashaar_meter_service/`

**High-level role:** Given `verse_text`, return Ashaar structural similarity metrics.

* `app/main.py`, `app/api.py`, `app/schemas.py` – expose `/ashaar-score`.
* `app/ashaar.py` – wraps `Models.Ashaar_runtime.ashaar_only.reward`.
* `app/config.py` – optional knobs (weights, log level).
* `tests/test_ashaar.py` – ensure scoring pipeline works with dummy BaitAnalysis.
* Dockerfile installs Ashaar deps (PyTorch optional) plus FastAPI.

### 5.3 `services/enhancer_service/`

**High-level role:** Receives the failed bayt plus its evaluation metadata and produces a single improved candidate by re-calling `shaer_service` without breaking the Shaer SFT prompt scaffold.

### Files & responsibilities

* `app/__init__.py`
* `app/main.py` – bootstraps FastAPI, wires the Shaer HTTP client and the enhancement planner, exposes `/health` and `/enhance-bayt`.
* `app/api.py` – validates `EnhanceBaytRequest`, runs the planner, calls `shaer_service`, and returns the normalized candidate + applied policy notes.
* `app/schemas.py` – request/response models composed of `BaytGenerationRequest`, `BaytMeterEval`, and `YehiaFeedback`.
* `app/config.py` – env vars for downstream URL/timeout plus knobs such as `MAX_EXTRA_GUIDANCE`, `ENABLE_DESCRIPTION_TIGHTENING`, `METER_FOCUS_THRESHOLD`.
* `app/clients.py` – lightweight HTTPX wrapper for `/generate-bayt` on `shaer_service`.
* `app/policy.py` – core enhancement policy (decides when to tighten description vs. add short instructions, trims feedback summaries to 1–3 bullets, normalizes outputs).
* `app/logging.py`, optional `README.md`.

**Guardrails enforced:**

* Never edits the poem directly – it only tweaks allowed request fields and adds up to a few short guidance bullets before delegating to Shaer.
* Meter issues trigger a meter-focused bullet; semantic issues tighten the description and optionally add a semantic bullet.
* Returns a single normalized bayt along with a list of applied adjustments so the orchestrator knows what was changed.

### 5.4 `services/scoring_service/`

**High-level role:** Compute the deterministic weighted final score for a bayt evaluation so the orchestrator doesn’t hard-code scoring logic.

* `app/main.py`, `app/api.py` – exposes `/health` and `/score` (accepts normalized component scores and optional Yehia score, returns final score + pass flag + breakdown).
* `app/schemas.py` – Pydantic models validating that each component is in `[0,1]` (and Yehia score in `[0,100]`).
* `app/scoring.py` – implements the fixed equation and clamps output to `[0,1]` before comparing against `SCORING_PASS_THRESHOLD`.
* `app/config.py`, `app/logging.py`, `README.md` – standard service plumbing.

This service is stateless and arithmetically combines the inputs: `0.5*meter + 0.15*meaning + 0.15*fluency + 0.1*poeticness + 0.1*cohesion`, marking `passed` when the result ≥ threshold.

### 5.5 `services/aspect_evaluator_service/`

**High-level role:** For each generated bayt, gather Yehia’s aspect-focused critiques (meaning, cohesion, fluency, poeticness) and transform them into calibrated numeric scores via OpenAI Structured Outputs.

* `app/clients.py` – async Yehia client hitting `/feedback` with the `aspect` hint.
* `app/openai_judge.py` – wraps the OpenAI Responses API (gpt-4o-mini) with a JSON schema (`score_0_1`, `notes`) and deterministic temperature.
* `app/evaluator.py` – orchestrates 4× Yehia calls + OpenAI scoring, normalizing verse text and including previous verses for cohesion checks.
* `app/schemas.py` – request/response models for `/evaluate-bayt` (all aspects) and `/evaluate-aspect` (single aspect) with stable keys.
* `prompts.py` – aspect rubrics + schema definition; `README.md` documents env vars (`ASPECT_EVAL_*`).

The service never fixes poetry nor computes final scores; it only returns `{aspect: {yehia_feedback, judge}}` so `scoring_service` and `enhancer_service` can reuse the structured outputs.

### 5.6 `services/orchestrator_service/`

**High-level role:** The UI only calls this service. It interprets intent and executes deterministic workflows by chaining the downstream microservices (RAG → Yehia → Shaer → scoring/enhancer).

* `app/main.py` wires shared settings + downstream HTTP clients and mounts the routers.
* `routers/poem_api.py` – `/poem/generate` handles bayt-by-bayt generation (optional RAG search, `yehia_service` spec building, `shaer_service` generation, `scoring_service` evaluation, `enhancer_service` retries).
* `routers/bayt_api.py` – `/bayt/fix` and `/bayt/score` normalize single verses, call `scoring_service`, and optionally trigger the enhancer loop.
* `routers/library_api.py` – `/library/search` proxies to `rag_service` for retrieval flows.
* `clients/` – httpx clients for rag, yehia, shaer, scoring, enhancer so the orchestrator never generates/fixes poetry itself.
* `utils/` – normalization helper plus scoring-feedback summarizer (1–3 actionable bullets) passed into enhancer requests to keep Shaer’s prompt scaffold intact.

---

## 6. Running with Docker & testing

### 6.1. Dockerfiles (per service)

For each service (yehia_client, shaer_client, rag_service, meter_service, ashaar_meter_service):

* Use similar pattern:

  * `FROM python:3.12-slim` (or similar).
  * `WORKDIR /app`
  * `COPY` that service folder + `common_schemas`.
  * `pip install -r requirements.txt` (you can keep a central `services/requirements.txt` or per-service).
  * `ENV` for any default config if you like.
  * `CMD` to start the web server (`uvicorn app.main:app --host 0.0.0.0 --port 8000`).

Team can share a **common base pattern** and tweak only the ports/env per service.

### 6.2. Root `docker-compose.yml`

At project root, create a compose file that:

* Defines networks & volumes.
* Brings up:

  * Neo4j + Chroma (from your existing `RAG/docker-compose.yml` or copy those services).
  * `rag_service`
  * `yehia_client`
  * `shaer_client`
  * `meter_service`
  * `ashaar_meter_service`
* Sets env for each service using either:

  * `env_file: ./services/<name>/.env`
  * or inline `environment:`.

Example idea (without actual YAML):

* `service: yehia_client`

  * `build: ./services/yehia_client`
  * `ports: ["8001:8000"]`
* `service: shaer_client`

  * `build: ./services/shaer_client`
  * `ports: ["8002:8000"]`
* etc.

So you can access:

* `http://localhost:8001` → Yehia
* `http://localhost:8002` → Shaer
* `http://localhost:8003` → RAG
* `http://localhost:8004` → Meter
* `http://localhost:8005` → Ashaar Meter
* `http://localhost:8004` → Meter

### 6.3. How to run

From repo root:

1. Ensure `RAG/docker-compose` services are either:

   * included in root compose, OR
   * started separately (`cd RAG && docker compose up -d`).

2. Build and start all services:

```bash
docker compose up -d --build
```

3. Check containers:

```bash
docker ps
```

4. Hit health endpoints:

* `curl http://localhost:8001/health` → Yehia
* `curl http://localhost:8002/health` → Shaer
* `curl http://localhost:8003/health` → RAG
* `curl http://localhost:8004/health` → Meter
* `curl http://localhost:8005/health` → Ashaar Meter

If all respond OK, you’re good.

### 6.4. Testing flows

* **Unit tests**: from repo root:

  ```bash
  pytest services/yehia_client/tests
  pytest services/shaer_client/tests
  pytest services/rag_service/tests
  pytest services/meter_service/tests
  pytest services/ashaar_meter_service/tests
  ```

* **Manual HTTP tests** (Postman / curl):

  * Call `/search` on `rag_service` with some Arabic query.
  * Use its result to call `/build-spec` on `yehia_client`.
  * Take the spec and call `/generate-bayt` on `shaer_client`.
  * Send the generated bayt + `spec.poem_meter` to `/eval-bayt` on `meter_service`.
  * Optionally call `/ashaar-score` on `ashaar_meter_service` for structural similarity.

This “manual chain” is exactly what your future orchestrator agent will do automatically.
