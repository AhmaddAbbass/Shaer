# Agent Service

FastAPI orchestration layer that exposes a single endpoint (`POST /api/chat`) for the Shaer Studio.
It routes user requests to downstream microservices (Yehia, Shaer, RAG, Meter) to generate, fix, search or explain poems.

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
