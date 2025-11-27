
1. `services/common_schemas`
2. `services/shaer_client`
3. `services/yehia_client`
4. `services/rag_service`
5. `services/meter_service`
6. Root-level `docker-compose.yml` + basic tests.

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

## 5. `services/meter_service/` – bayt meter checker

**High-level role:**
Given `verse_text` + `target_meter`, return `BaytMeterEval`.

### Files & responsibilities

* `app/__init__.py`

* `app/main.py`

  * Create app, optionally load scansion model on startup.
  * `GET /health` should confirm the model is loaded.

* `app/api.py`

  * Single main endpoint:

    * `POST /eval-bayt`

      * Input: `verse_text`, `target_meter`.
      * Output: `BaytMeterEval` (score, on_meter, notes).
  * Flow:

    * Parse JSON from `schemas.py`.
    * Call `scansion.py`.
    * Map scansion raw output → `BaytMeterEval`.

* `app/schemas.py`

  * HTTP-level models:

    * `EvalBaytRequest` (verse_text, target_meter).
    * `EvalBaytResponse` ({ `result: BaytMeterEval` }).

* `app/scansion.py`

  * Wraps the actual meter model(s).
  * Responsibilities:

    * Load model (local or remote).
    * Provide a function like `evaluate_bayt(text, target_meter)` that returns raw metrics (distances, scores).
    * Convert raw metrics to:

      * integer `meter_score` (0–100),
      * `on_meter` boolean based on threshold from env,
      * text notes.

* `app/config.py`

  * Reads:

    * `METER_MODEL_PATH` or `METER_API_BASE`
    * `METER_SCORE_THRESHOLD` (e.g. 80)
    * `METER_TIMEOUT_SECONDS`
    * `METER_LOG_LEVEL`

* `app/logging.py`

  * Standard logging.

* `tests/test_eval_bayt.py`

  * Minimal tests:

    * When input is perfect example, `on_meter` should be True.
    * When input is clear nonsense, `on_meter` False.

* `Dockerfile` + `README.md`

  * Docker: install whatever deps the scansion model needs.
  * README: expected env vars, example request/response.

---

## 6. Running with Docker & testing

### 6.1. Dockerfiles (per service)

For each service (yehia_client, shaer_client, rag_service, meter_service):

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

If all respond OK, you’re good.

### 6.4. Testing flows

* **Unit tests**: from repo root:

  ```bash
  pytest services/yehia_client/tests
  pytest services/shaer_client/tests
  pytest services/rag_service/tests
  pytest services/meter_service/tests
  ```

* **Manual HTTP tests** (Postman / curl):

  * Call `/search` on `rag_service` with some Arabic query.
  * Use its result to call `/build-spec` on `yehia_client`.
  * Take the spec and call `/generate-bayt` on `shaer_client`.
  * Send the generated bayt + `spec.poem_meter` to `/eval-bayt` on `meter_service`.

This “manual chain” is exactly what your future orchestrator agent will do automatically.
