Okay, let’s zoom all the way in on **yehia-service** and treat it like a mini product spec.
No code, just what every piece *does* and how it all fits together.

I’ll assume:

* This service lives at: `services/yehia_client/`
* `.env` inside that folder has:

  ```text
  RUNPOD_API_KEY=...         # shared key for all RunPod endpoints
  YEHIA_ENDPOINT_ID=8t5d87yaj86cmf
  SHAER_ENDPOINT_ID=e0q99jgj3psqlu   # available if we ever want cross-calls
  ```

---

## 1. Yehia-service: What is it, conceptually?

**Role:** A small HTTP service that wraps the Yehia-7B model hosted on RunPod and gives you three “smart” capabilities:

1. **Generic chat** – "/chat": talk to Yehia like a normal LLM.
2. **Spec builder** – "/build-spec": given a fuzzy user request + optional RAG hints, return a clean `PoemSpec` (meter, theme, era, poet, description, num_verses).
3. **Poetry critic** – "/feedback": given a single bayt + `PoemSpec`, return structured feedback (`YehiaFeedback`).

It **hides**:

* the RunPod API details,
* the prompt engineering for each task,
* the parsing of Yehia’s outputs.

It **exposes**:

* simple typed JSON endpoints for your orchestrator / other services.

---

## 2. Folder structure recap

```text
services/yehia_client/
  app/
    __init__.py
    main.py
    api.py
    schemas.py
    client.py
    config.py
    logging.py
    prompt.py
  tests/
    __init__.py
    test_chat.py
    test_build_spec.py
    test_feedback.py
  Dockerfile
  README.md
  .env
```

Now let’s go file by file.

---

## 3. `app/config.py` – configuration + .env

**Goal:** centralize all config; no other file touches environment variables directly.

### What it reads

From the `.env` beside the service:

* `RUNPOD_API_KEY`
* `YEHIA_ENDPOINT_ID`
* `SHAER_ENDPOINT_ID` (might be unused for now, but good to expose)
* Optional extras you may add later:

  * `YEHIA_TEMPERATURE` (default temp for generative calls)
  * `YEHIA_MAX_NEW_TOKENS`
  * `YEHIA_TIMEOUT_SECONDS`
  * `YEHIA_LOG_LEVEL`

### What it exposes

A small config object / values that other modules import, for example:

* `RUNPOD_API_KEY`

* `YEHIA_BASE_URL` – built from `YEHIA_ENDPOINT_ID`.
  For RunPod serverless this is typically something like:

  > “[https://api.runpod.ai/v2/{YEHIA_ENDPOINT_ID}/runsync”](https://api.runpod.ai/v2/{YEHIA_ENDPOINT_ID}/runsync”)

* `DEFAULT_TEMPERATURE`

* `DEFAULT_MAX_NEW_TOKENS`

* `REQUEST_TIMEOUT`

The team should make sure:

* The file fails loudly (clear error) if `RUNPOD_API_KEY` or `YEHIA_ENDPOINT_ID` are missing.
* Nothing logs the actual API key.

---

## 4. `app/logging.py` – logging setup

**Goal:** consistent logs across endpoints.

What it should do:

* Set a log level from config (`YEHIA_LOG_LEVEL` or default).
* Define a helper `get_logger(name)`:

  * attaches a formatter with timestamps, level, module.
  * maybe logs to stdout only (good for Docker).

Other files call this to get a logger and log:

* incoming requests (without sensitive data),
* outgoing RunPod calls (endpoint ID, latency),
* errors / exceptions.

---

## 5. `app/schemas.py` – HTTP-level request/response shapes

**Goal:** define what JSON bodies this service accepts/returns.

It should import “domain” types from `common_schemas.schemas` and wrap them for each endpoint.

### Core imported types

From `common_schemas.schemas`:

* `PoemSpec`
* `YehiaFeedback`
* `RagSearchHit`
* `LLMMessage`

### HTTP models to define

1. **Chat**

   * `ChatRequest`:

     * `messages: list[LLMMessage]`
     * Optional overrides: `temperature?`, `max_new_tokens?`

   * `ChatResponse`:

     * `completion: str`

2. **Build Spec**

   * `BuildSpecRequest`:

     * `user_query: str`
     * `rag_hits: list[RagSearchHit]` (optional; can be empty)

   * `BuildSpecResponse`:

     * `spec: PoemSpec`
     * Optionally: `debug_notes: str` (if you want to expose how decisions were made)

3. **Feedback**

   * `FeedbackRequest`:

     * `verse_text: str`
     * `spec: PoemSpec`

   * `FeedbackResponse`:

     * `feedback: YehiaFeedback`

These are what `api.py` uses to validate incoming/outgoing JSON.

---

## 6. `app/prompt.py` – all the smart prompts

**Goal:** concentrate all **prompt engineering** in one file, so:

* if you refine instructions, you do it here,
* the rest of the system just calls functions like “build spec messages”.

### 6.1 System prompts

At least two core system prompt texts:

1. **Spec builder system prompt**

   Something like (in Arabic, your style):

   * “أنت مخطط قصائد عربي محترف… وظيفتك تلخيص طلب المستخدم وبناء مواصفات قصيدة (بحر، موضوع، عصر، شاعر، وصف، عدد أبيات)… لا تكتب القصيدة، فقط المواصفات…”

   Purpose:

   * Guide Yehia to think like a planner, not a poet.

2. **Feedback system prompt**

   * “أنت ناقد شعري عربي متمكن من العروض واللغة… وظيفتك تقييم بيت واحد بناءً على مواصفات القصيدة، وبيان ما إذا كان مناسبًا من ناحية المعنى والأسلوب والانسجام مع الوصف…”

   Purpose:

   * Guide Yehia to be a constructive critic: accept / reject + explanation.

You might also keep a generic chat system prompt (“مساعد لغوي عربي عام…”) for `/chat`.

### 6.2 Message builder helpers

1. **`build_chat_messages(messages: list[LLMMessage])`**

   * For `/chat`, this might just return them directly.
   * You may optionally prepend a default system message if the caller didn’t include one.

2. **`build_spec_messages(user_query, rag_hits)`**

   * Takes:

     * the user’s raw query (Arabic)
     * some candidate poems from RAG (each with title, meter, description, etc.)

   * Builds a message list like:

     * System: spec-builder system prompt.
     * User: a message that includes:

       * the original user query,
       * a short bullet list summarizing the retrieved poems (e.g. “قصيدة 1: meter=..., theme=..., description=...”),
       * instructions to Yehia to:

         * fill in `PoemSpec` fields:

           * meter,
           * theme,
           * era,
           * poet style,
           * description,
           * num_verses
         * use information from RAG hits when user didn’t specify something,
         * keep user’s intent as the main guide,
         * **respond in a very structured way** so the service can parse it (for example: explicit labels or JSON-like output).

   * Important: this builder must **force** Yehia to output something parseable, e.g. bullet list or pseudo-JSON; the parsing logic will live in `api.py`.

3. **`build_feedback_messages(verse_text, spec)`**

   * Takes:

     * `verse_text`
     * `PoemSpec` (so Yehia knows the target meter, theme, era, style).

   * Builds messages where:

     * System: feedback system prompt.
     * User: contains:

       * the spec in a readable bullet list,
       * the bayt itself, highlighted,
       * explicit instructions like:

         * say whether the bayt fits the description and style,
         * indicate if meaning is clear and coherent,
         * do not rewrite the bayt (or: you may propose an improved alternative, depending on your design),
         * end with a structured verdict (“مقبول/غير مقبول” plus reasons).

   * Again, the structure of Yehia’s answer should be constrained so that `api.py` can map it to `YehiaFeedback`.

---

## 7. `app/client.py` – RunPod HTTP client

**Goal:** the only place that knows how to talk to RunPod’s serverless API.

### What it needs to know

* `RUNPOD_API_KEY` (from `config.py`)
* `YEHIA_ENDPOINT_ID` (to build the URL)
* Timeout, temperature, max tokens defaults.

### How it works conceptually

RunPod serverless usually expects:

* An HTTP POST to a URL that includes the endpoint ID.
* A JSON body with some `input` field that your handler in the container understands.

Since we’re **consumers** of RunPod endpoints, not deploying them here, the team needs to:

* Decide on the RunPod endpoint contract (e.g. does it expect `{"input": {"messages": [...], "temperature": ...}}`?), or
* Align with how the endpoint was defined in RunPod UI.

**Client responsibilities:**

* Assemble the request JSON from:

  * messages built by `prompt.py`,
  * temperature / max tokens.

* Add headers:

  * `Authorization: Bearer <RUNPOD_API_KEY>`
  * `Content-Type: application/json`

* Send request to Yehia endpoint:

  * URL similar to:
    `https://api.runpod.ai/v2/{YEHIA_ENDPOINT_ID}/runsync`
    (exact path depends on RunPod spec, but this is the idea.)

* Handle:

  * network errors (timeout, connection errors),
  * non-200 HTTP status,
  * malformed response JSON.

* Return **just the generated text** (string) to callers:

  * For `/chat`: the whole reply.
  * For `/build-spec`: the textual spec Yehia wrote.
  * For `/feedback`: the textual critique.

Other modules don’t care about RunPod structure; they just call `run_yehia(messages, temperature, max_tokens)` and get a string.

---

## 8. `app/api.py` – HTTP endpoints and business logic

**Goal:** connect HTTP world ↔ prompt builders ↔ RunPod client ↔ parsing.

### Endpoint 1: `POST /chat`

Flow:

1. Receive `ChatRequest` JSON → validate into `ChatRequest` model.
2. Build messages:

   * either use request’s messages directly, or wrap them with a default system prompt.
3. Call `client.run_yehia_chat(messages, temperature, max_new_tokens)`.
4. Get completion string from client.
5. Wrap it into `ChatResponse` and return.

Usage: orchestrator uses this when it just wants free-form reasoning/explanations.

---

### Endpoint 2: `POST /build-spec`

Flow:

1. Receive `BuildSpecRequest`:

   * read `user_query`,
   * read optional `rag_hits` (list of `RagSearchHit`).

2. Use `prompt.build_spec_messages(user_query, rag_hits)` to create messages.

3. Call RunPod via `client`: get spec text from Yehia.

4. **Parse** Yehia’s output into a `PoemSpec`:

   * This is an important step: you must define a simple parsing convention, for example:

     * Yehia responds in pseudo-JSON:

       ```text
       {
         "poem_meter": "الكامل",
         "poem_theme": "الحنين",
         "poem_era": "العصر الحديث",
         "poet_name": "أسلوب نزار قباني",
         "poem_description": "…",
         "num_verses": 8
       }
       ```

     * Or a strongly structured bullet list with labels (“البحر: …”, “الموضوع: …”).
   * api.py then extracts each field, handles missing values (fallbacks), and creates a `PoemSpec` object.

5. Return `BuildSpecResponse` with that `PoemSpec`.

This endpoint encodes the **brain** that transforms “يا يحيى اكتب لي قصيدة عن…” into a coherent spec that Shaer can use.

---

### Endpoint 3: `POST /feedback`

Flow:

1. Receive `FeedbackRequest` (`verse_text`, `spec`).

2. Use `prompt.build_feedback_messages(verse_text, spec)` to create messages.

3. Call RunPod via `client`: get critique string from Yehia.

4. Parse the response into `YehiaFeedback`:

   * Determine:

     * `ok` boolean (acceptable vs not).
     * `score` (if you choose to have one; can be extracted from e.g. “تقييمي: 85/100”).
     * `feedback` (the textual explanation).

5. Return `FeedbackResponse` with `feedback`.

This is what your enhancer agent will use to decide whether to accept a bayt or ask Shaer to regenerate.

---

## 9. `app/main.py` and `app/__init__.py`

**`__init__.py`**

* Just marks `app` as a package.
* Optionally exposes a `create_app()` function that builds the FastAPI/Flask app.

**`main.py`**

* Actual entrypoint for the web service.

* Responsibilities:

  * Create application instance.
  * Include router defined in `api.py`.
  * Define basic health endpoint:

    * `GET /health` → returns `{status: "ok"}`.
    * Optionally: does a tiny test call or just says “up”.

* This is what `uvicorn` / Docker will point to (e.g. `app.main:app`).

---

## 10. Tests (`tests/`)

**Goal:** give the team confidence that the service contract is stable.

* `test_chat.py`

  * Uses a fake or mocked `client.run_yehia_chat` to avoid hitting RunPod.
  * Asserts `/chat` endpoint:

    * accepts correct body,
    * returns `completion` string.

* `test_build_spec.py`

  * Mocks client to return a fixed “spec text”.
  * Asserts:

    * `PoemSpec` fields are parsed correctly,
    * defaults are applied properly when Yehia omits something.

* `test_feedback.py`

  * Mocks client to return a fixed feedback text.
  * Asserts:

    * `YehiaFeedback.ok` is correctly derived,
    * `feedback` string is not empty,
    * score is inside [0, 100] if used.

These tests are **pure Python** with no external HTTP calls; they just check parsing & business logic.

---

## 11. Docker & runtime

**Dockerfile (conceptually):**

* Use a Python base image.
* Copy `services/common_schemas` and `services/yehia_service`.
* Install Python deps (FastAPI/Flask, HTTP client, dotenv, any needed libs).
* At runtime:

  * Load `.env` inside the container.
  * Start web server (e.g. `uvicorn app.main:app --host 0.0.0.0 --port 8000`).

**Environment:**

* In `docker-compose.yml`, for `yehia_service` service, you mount or copy `.env` and ensure:

  * `RUNPOD_API_KEY` is set (from your local secrets),
  * `YEHIA_ENDPOINT_ID` matches the RunPod endpoint,
  * `SHAER_ENDPOINT_ID` is optional.

**Testing manually:**

* After `docker compose up`:

  * `curl http://localhost:8001/health` → should be OK.
  * `POST /chat` with a tiny example.
  * Once you’re happy, orchestrator can start calling `/build-spec` and `/feedback`.
