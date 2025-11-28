# Agent Service

FastAPI orchestration layer that exposes a single endpoint (`POST /api/chat`) for the Shaer Studio.
It routes user requests to downstream microservices (Yehia, Shaer, RAG, Meter) to generate, fix, search or explain poems.

## Architecture
-	The orchestrator is a ChatGPT (GPT‑4o) session that runs with the team’s OpenAI API key. The UI/backend send the entire chat history to this service, which forwards it to the ChatGPT orchestrator along with the tool catalog. ChatGPT is responsible for deciding whether to generate, fix, search or explain, and triggers the right microservice tools. The very first user input is sent verbatim to ChatGPT together with an instruction block that lists the available tools/services (`rag.search`, `yehia.build_spec`, `shaer.generate_bayt`, `meter.eval_bayt`, etc.); GPT‑4o chooses which tool to invoke then the workflow continues as described below.
-	Only one public input exists: `POST /api/chat`. Every user action (new poem, fix, search, explain) hits the orchestrator, which selects tools such as `rag.search`, `yehia.build_spec`, `shaer.generate_bayt`, `meter.eval_bayt`, etc.
-	Tool usage is logged through the shared `agent_trace`. Each step records which tool ran, by which agent, and a short Arabic summary so the UI can render the execution timeline.
-	Default/fallback values guard against downstream failures (e.g., if meter service is unreachable we fall back to `AGENT_DEFAULT_METER`, if RAG is down we keep building the spec using only the user query, etc.).

### Tool choreography
-	**Generate:** Orchestrator calls RAG to fetch similar poems, uses Yehia to build/complete the spec (meter, theme, era, poet voice, verse count). A Shaer-ready prompt is assembled and sent to `shaer_service`. Each generated verse is evaluated (meter + Yehia feedback); if a verse fails it is sent to the enhancer loop (regen up to `AGENT_MAX_BAYT_RETRIES`, currently 3). Accepted verses are appended to the prompt context to guide subsequent verses.
-	**Fix:** User verses go directly to the feedback/eval loop (meter + Yehia). Any verse that fails is regenerated via Shaer with the same prompt skeleton and the already-accepted verses in context. This continues verse-by-verse until every line passes or retries are exhausted.
-	**Search:** Orchestrator queries RAG using the user text. Results are sorted alphabetically by era, then poet, then poem title, and the formatted reply includes `العصر – الشاعر – العنوان` followed by a short snippet (usually the opening bayt). Library hits are also surfaced in `library_context`.
-	**Explain:** Yehia is prompted in its teacher persona to answer theoretical questions or explain verses; no Shaer/meter usage happens in this mode.

### Fallback defaults
-	If RAG fails: proceed with the raw user description, log a warning step, and still call Yehia/spec builder.
-	If Yehia cannot determine meter/theme/count: fill `poem_meter`, `poem_theme`, and `num_verses` using the configured defaults, and flag that adjustment in `agent_trace` + `warnings`.
-	If Shaer times out: retry up to `AGENT_MAX_BAYT_RETRIES`, otherwise insert a placeholder verse to keep the workflow moving.
-	If meter service is unavailable or misbehaves: log the issue, mark the verse as accepted, and report that the default meter was assumed.

## Endpoints
-	`GET /health` – simple heartbeat.
-	`POST /api/chat` – accepts the full chat history and optional `mode` hint, returns the aggregated `ChatResponse` (reply text, poem spec, verses, agent trace, library context).

## Configuration
Environment variables (prefixed with `AGENT_`):

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_YEHIA_BASE_URL` | `http://localhost:8101` | Yehia service base URL |
| `AGENT_SHAER_BASE_URL` | `http://localhost:8102` | Shaer service base URL |
| `AGENT_RAG_BASE_URL` | `http://localhost:8003` | RAG service base URL |
| `AGENT_METER_BASE_URL` | `http://localhost:8104` | Meter service base URL |
| `AGENT_HTTP_TIMEOUT_SECONDS` | `30.0` | HTTP timeout for downstream calls |
| `AGENT_RAG_TOP_K` | `5` | Number of retrieved poems for context/search |
| `AGENT_DEFAULT_METER` | `الكامل` | Default meter when unknown |
| `AGENT_DEFAULT_THEME` | `شعر وجداني` | Default theme |
| `AGENT_DEFAULT_NUM_VERSES` | `6` | Default poem length |
| `AGENT_MAX_BAYT_RETRIES` | `3` | Regeneration attempts per bayt |
| `AGENT_OPENAI_API_KEY` | `null` | OpenAI key used to call GPT-4o for tool selection |
| `AGENT_OPENAI_MODEL` | `gpt-4o-mini` | GPT model identifier used for the selector |

## Local run
```bash
cd services/agent_service
uvicorn app.main:app --reload --port 8000
```

Ensure the dependent services are reachable via the configured base URLs.

## Docker
```bash
docker build -t agent-service -f services/agent_service/Dockerfile .
docker run --rm -p 8000:8000 \
  -e AGENT_YEHIA_BASE_URL=http://host.docker.internal:8101 \
  -e AGENT_SHAER_BASE_URL=http://host.docker.internal:8102 \
  -e AGENT_RAG_BASE_URL=http://host.docker.internal:8003 \
  -e AGENT_METER_BASE_URL=http://host.docker.internal:8104 \
  agent-service
```
