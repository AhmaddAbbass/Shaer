Yesss, let’s spec **shaer_service** properly now 😌
Think of this as the “how to implement” doc you hand to your team.

---

## 0. What is `shaer_service`?

**Role:** This service is the **only gateway** to your fine-tuned **Shaer-7B-v1** poet model on RunPod.

It:

* Knows exactly **how Shaer was trained** (system+user messages format).
* Takes a structured **`PoemSpec` + previous verses + sequence_number**.
* Builds the **correct SFT-style messages**.
* Calls the **RunPod Shaer endpoint**.
* Returns a clean **`Verse`** object (or full `Poem` if you choose to support multi-bayt).

It does **not**:

* Decide whether a verse is good or bad (that’s critic/enhancer).
* Fix meter or semantics.
* Build specs from user queries (that’s Yehia).

It’s a **pure generation adapter**.

---

## 1. Folder structure recap

```text
services/shaer_service/
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
    test_prompt.py
    test_generate_bayt.py
  Dockerfile
  README.md
  .env         # for this service only
```

You already created empty files; this tells the team what each must contain.

---

## 2. `.env` for `shaer_service`

Inside `services/shaer_service/.env`, you’ll have something like:

```text
RUNPOD_API_KEY=...                 # same as yehia service
SHAER_ENDPOINT_ID=e0q99jgj3psqlu   # the serverless Shaer endpoint
```

Optional extras (good to define early):

```text
SHAER_TEMPERATURE=0.7
SHAER_TOP_P=0.9
SHAER_MAX_NEW_TOKENS=128          # enough for one bayt
SHAER_TIMEOUT_SECONDS=60
SHAER_LOG_LEVEL=INFO
```

No other file should touch env vars directly; they go through `config.py`.

---

## 3. `app/config.py` – configuration

**Goal:** meaningfully wrap `.env` for the rest of the service.

### Responsibilities

* Load environment variables:

  * `RUNPOD_API_KEY`
  * `SHAER_ENDPOINT_ID`
  * `SHAER_TEMPERATURE`, `SHAER_TOP_P`, `SHAER_MAX_NEW_TOKENS`, `SHAER_TIMEOUT_SECONDS`, `SHAER_LOG_LEVEL` (with sensible defaults).

* Build the **RunPod base URL**:

  * Something like `SHAER_BASE_URL = f"https://api.runpod.ai/v2/{SHAER_ENDPOINT_ID}/runsync"`
    (exact path depends on RunPod, but this is the idea).

* Optionally expose a simple `Settings`/config object that other modules import.

### Requirements for team

* If `RUNPOD_API_KEY` or `SHAER_ENDPOINT_ID` is missing → fail clearly at startup.
* Don’t log secrets.

---

## 4. `app/logging.py` – logging

**Goal:** consistent and simple logging for this service.

### Responsibilities

* Set a log level from `SHAER_LOG_LEVEL` (e.g. `DEBUG`, `INFO`).
* Define `get_logger(name)`:

  * uses a formatter with timestamp, level, module.
  * logs to stdout (good for Docker).

**Expected usage:**

* `api.py` logs incoming requests, sequence_number, spec summary.
* `client.py` logs outgoing RunPod calls (endpoint ID, latency).
* `prompt.py` can log debug info like “previous_verses block built for 3 bayts” (optional).

---

## 5. `app/schemas.py` – HTTP-level models

**Goal:** define the JSON contract of `shaer_service` endpoints.

It should **import** the core types from `common_schemas.schemas`:

* `PoemSpec`
* `Verse`
* `Poem`

Then define HTTP request/response wrappers.

### 5.1. Generate a single bayt

* **`GenerateBaytRequest`**:

  * `spec: PoemSpec`
  * `previous_verses: list[Verse]` (can be empty for first bayt)
  * `sequence_number: int` (1 for first bayt, 2 for second, …)

  Optional override fields:

  * `temperature?`, `top_p?`, `max_new_tokens?` (to override defaults from config if the caller wants).

* **`GenerateBaytResponse`**:

  * `verse: Verse`
  * Optional debug fields:

    * `raw_completion?: str` (exact text returned from the model before cleaning),
    * `used_temperature?: float`, `used_top_p?: float`.

### 5.2. (Optional) Generate full poem

You **don’t have workflow_service now**, but you may still have a convenience endpoint that loops internally:

* **`GeneratePoemRequest`**:

  * `spec: PoemSpec`
  * Optional: `max_retries_per_bayt?` (internal resilience).

* **`GeneratePoemResponse`**:

  * `poem: Poem`
  * Optional: `raw_verses?: list[str]`

This is optional; you can keep `shaer_service` minimal (only `/generate-bayt`) and let the orchestrator or another service handle loops.

---

## 6. `app/prompt.py` – the sacred Shaer prompt

**This is the most important file for Shaer.**
Its job is to ensure that every call to Shaer uses **exactly the same structure** as the SFT dataset.

### 6.1. System prompt constant

* `SHAER_SYSTEM_PROMPT`

This string should be **identical** to the system messages used during SFT, e.g.:

> "أنت شاعر عربي متمكن من مختلف البحور والأغراض الشعرية، وقادر على محاكاة أساليب الشعراء عبر العصور مع الحفاظ على سلامة اللغة، والوزن، والقافية."

No experimenting here; this is fixed.

### 6.2. Helpers to build user content

You want to recreate the exact template you showed:

```text
المطلوب منك في هذه المهمة أن تولّد بيتًا شعريًا واحدًا فقط، وفق المواصفات التالية:

- البحر: البسيط
- الوصف العام لموضوع القصيدة: ...
- العصر: ...
- الشاعر: ...
- عدد أبيات القصيدة الكلي: 6
- ترتيب البيت المطلوب داخل القصيدة: 1

الأبيات السابقة في القصيدة:
لا توجد أبيات سابقة، فهذا هو البيت الأول في القصيدة.

إرشادات مهمة:
- أخرج بيتًا واحدًا مكوّنًا من صدر وعجز في سطر واحد.
- التزم بالبحر الشعري...
- لا تكرّر أي بيت سابق...
```

So `prompt.py` should contain **logic, not code** like:

1. **`format_spec_block(spec: PoemSpec, sequence_number: int)`**

   * Builds the “المطلوب منك…” paragraph and bullet list:

     * `- البحر: {spec.poem_meter}`
     * `- الوصف العام لموضوع القصيدة: {spec.poem_description}`
     * `- العصر: {spec.poem_era or 'غير محدد'}`
     * `- الشاعر: {spec.poet_name or 'أسلوب عام'}`
     * `- عدد أبيات القصيدة الكلي: {spec.num_verses}`
     * `- ترتيب البيت المطلوب داخل القصيدة: {sequence_number}`

   * The wording should stay as close as possible to training texts.

   * Any missing fields (era, poet name, theme) are filled with a neutral phrase, not removed, so the structure remains constant.

2. **`format_previous_verses_block(previous_verses: list[Verse])`**

   * If list is empty:

     * Return the exact training phrasing, e.g.:

       * “لا توجد أبيات سابقة، فهذا هو البيت الأول في القصيدة.”

   * If there are verses:

     * Build:

       ```text
       الأبيات السابقة في القصيدة:
       1) {verse1.text}
       2) {verse2.text}
       ...
       ```

   * All verses go on **one line each**, consistent with dataset.

3. **`build_instructions_block()`**

   * This is the “إرشادات مهمة:” section.

   * It should contain **almost exactly** what you trained on:

     * “- أخرج بيتًا واحدًا مكوّنًا من صدر وعجز في سطر واحد.”
     * “- التزم بالبحر الشعري، وبجوّ ومعنى القصيدة كما في الوصف والأبيات السابقة (إن وُجدت).”
     * “- لا تكرّر أي بيت سابق ولا تضف شروحًا أو عناوين أو علامات خاصة؛ الناتج هو البيت فقط.”

   * You can *very lightly* add bullets like “- تجنّب الإطالة المفرطة” later, but the core must remain stable.

### 6.3. Core builder: `build_shaer_messages`

Functionally (no code here):

* Inputs:

  * `spec: PoemSpec`
  * `previous_verses: list[Verse]`
  * `sequence_number: int`

* Steps:

  1. Call `format_spec_block`.

  2. Call `format_previous_verses_block`.

  3. Call `build_instructions_block`.

  4. Concatenate into one big **user content string** following training order:

     ```text
     المطلوب منك في هذه المهمة أن تولّد بيتًا شعريًا واحدًا فقط، وفق المواصفات التالية:

     <spec bullet list>

     الأبيات السابقة في القصيدة:
     <previous verses block>

     إرشادات مهمة:
     <instructions bullets>
     ```

  5. Return the messages array:

     ```json
     [
       { "role": "system", "content": SHAER_SYSTEM_PROMPT },
       { "role": "user", "content": <constructed content> }
     ]
     ```

Team rule:

> **We never change the structure; we only change the plugged-in values (`spec` & previous verses) and, very cautiously, add small hints inside the same sections.**

---

## 7. `app/client.py` – RunPod client for Shaer

**Goal:** talk to the Shaer model on RunPod, given `messages`.

### Responsibilities

* Receive:

  * `messages: list[LLMMessage]`
  * `temperature`, `top_p`, `max_new_tokens` (either from request or config).

* Build the **RunPod request body**, something like (conceptually):

  ```json
  {
    "input": {
      "messages": [...],
      "temperature": 0.7,
      "top_p": 0.9,
      "max_new_tokens": 128
    }
  }
  ```

* Attach headers:

  * `Authorization: Bearer {RUNPOD_API_KEY}`
  * `Content-Type: application/json`

* POST to `SHAER_BASE_URL` computed in `config.py`.

* Wait for RunPod to return a response with the generated text.

  * Inside the container running on RunPod, your Shaer handler will:

    * take `messages`,
    * apply Yehia's chat_template (with `add_generation_prompt=True`),
    * call `model.generate(...)`,
    * return the decoded string.

* `client.py` then:

  * Parses RunPod’s JSON to extract the output string.
  * Normalizes it:

    * strip leading/trailing whitespace,
    * choose the first non-empty line as the bayt (if the model accidentally adds extra content).
  * Returns:

    * `generated_text: str`
    * maybe metadata like used top_p, etc., to `api.py`.

**Important:**
All connection errors, timeouts, non-200 status codes are caught and surfaced as clean errors to the API layer (e.g., “Shaer backend unavailable”).

---

## 8. `app/api.py` – HTTP endpoints + generation logic

**Goal:** expose clean, stable HTTP endpoints for the rest of your system.

### Endpoint 1: `POST /generate-bayt`

Flow:

1. Receive JSON body → validate into `GenerateBaytRequest`.

   * Check:

     * `spec.num_verses` ≥ `sequence_number`.
     * `previous_verses` length == `sequence_number - 1` (or allow more relaxed checks, but at least consistent).

2. Use `prompt.build_shaer_messages(spec, previous_verses, sequence_number)` to build messages.

3. Call `client.generate(messages, temperature, top_p, max_new_tokens)`.

4. Take `generated_text` string and convert it into a `Verse` object:

   * `text` = cleaned bayt string (one line).
   * `index` = `sequence_number`.

5. Return `GenerateBaytResponse`:

   * `verse` containing the bayt.
   * Optionally `raw_completion` for debugging.

This is the **core** of shaer_service; orchestrator will call this repeatedly.

---

### (Optional) Endpoint 2: `POST /generate-poem`

You can implement or postpone this; if you do, the flow is:

1. Receive `GeneratePoemRequest` with `spec`.

2. Initialize `verses = []`.

3. For each `sequence_number` from 1 to `spec.num_verses`:

   * Build messages with `verses` as `previous_verses`.
   * Call `client.generate`.
   * Append `Verse` to `verses`.

4. Wrap in `Poem(spec=spec, verses=verses)` and return.

Note:

* This endpoint should **not** try to evaluate or repair verses; that’s the enhancer’s job.
* It’s just a naive sequential generator.

---

## 9. `app/main.py` and `app/__init__.py`

**`__init__.py`**

* Marks `app` as a package.
* May define a `create_app()` function that instantiates the web app and registers routes.

**`main.py`**

* Creates the actual app instance (FastAPI/Flask).

* Includes the router defined in `api.py`.

* Defines:

  * `GET /health` → returns `{ "status": "ok" }` (and optionally checks a tiny internal thing, like config loaded).

* This is the entrypoint used by Docker/CMD, e.g.:

  * `uvicorn app.main:app --host 0.0.0.0 --port 8000`

---

## 10. Tests (`tests/`)

### `test_prompt.py`

* Tests the prompt-building logic **without** calling RunPod or the model.

What to check:

* For a `PoemSpec` with all fields filled and `previous_verses=[]`, `build_shaer_messages`:

  * returns exactly 2 messages (system + user),
  * system content = `SHAER_SYSTEM_PROMPT`,
  * user content contains:

    * the correct meter, description, era, poet, num_verses, sequence_number,
    * the phrase “لا توجد أبيات سابقة” for first bayt,
    * “إرشادات مهمة:” and bullets.

* For non-empty `previous_verses`, ensures they are numbered correctly and appear in the “الأبيات السابقة…” section.

This protects you from accidental prompt drift.

### `test_generate_bayt.py`

* Uses a mocked `client.generate` that returns a fake bayt like:

  * `"والنفسُ تَطْمَعُ في دُنْيا تُجَمِّلُها"`.

* Tests `/generate-bayt` endpoint:

  * Accepts valid `GenerateBaytRequest` JSON.
  * Returns a `GenerateBaytResponse`:

    * `verse.text` not empty,
    * `verse.index` equals `sequence_number`.

No external HTTP calls; just business logic and JSON wiring.

---

## 11. Dockerfile & runtime

**Dockerfile** (conceptually):

* Base: `python:3.12-slim` (or similar).
* `WORKDIR /app`.
* `COPY`:

  * `services/common_schemas/` → `/app/common_schemas/`
  * `services/shaer_service/` → `/app/shaer_service/`
* Install dependencies for:

  * web framework (FastAPI/Flask),
  * HTTP client (requests/httpx),
  * environment management,
  * Pydantic/whatever you use for schemas.
* `CMD` to run the app:

  * `uvicorn app.main:app --host 0.0.0.0 --port 8000`

**docker-compose.yml** (root-level):

* Simplest version:

  ```yaml
  shaer_service:
    build: ./services/shaer_service
    ports:
      - "8002:8000"
    env_file:
      - ./services/shaer_service/.env
  ```

Then:

* `docker compose up -d --build`
* `curl http://localhost:8002/health` → should say `"ok"`.
* `POST http://localhost:8002/generate-bayt` from Postman or a small script once your RunPod endpoint is ready.

---

## 12. How orchestrator will use `shaer_service`

Once `shaer_service` is live, your orchestrator/agents will:

1. Get a `PoemSpec` from `yehia_client.build-spec`.
2. For each `sequence_number`:

   * Call `shaer_service /generate-bayt` with:

     * `spec`
     * `previous_verses` (verses generated so far)
     * `sequence_number`
3. Combine these bayts into a `Poem`.
4. Pass each bayt to:

   * `meter_service /eval-bayt`
   * `yehia_client /feedback`
     to decide whether to accept/regenerate.

So `shaer_service` is absolutely **central** but **simple**:

> “You give me structured spec + context, I give you one bayt, using the exact SFT template. Nothing more, nothing less.”

---

If you want, next we can draft a small **SERVICES_SPEC.md** section combining the Yehia + Shaer specs, with bullet-pointed “contract” for each endpoint that you can paste into the repo / send to teammates.
