# Agent Service (Proxy)

A thin FastAPI proxy that forwards user requests to `orchestrator_service`. It does not generate or process poetry itself; all logic runs downstream.

## Endpoints
- `GET /health` – basic heartbeat.
- `POST /proxy` – forwards the JSON payload to the orchestrator. If `target` is omitted, it defaults to `/poem/generate`.

Example:
```bash
curl -X POST http://localhost:8001/proxy \
  -H "Content-Type: application/json" \
  -d '{"target": "/poem/generate", "payload": {"user_query": "اكتب قصيدة حنين", "desired_num_verses": 3}}'
```

## Environment variables (`AGENT_` prefix)
- `ORCHESTRATOR_BASE_URL` (default `http://localhost:8000`)
- `HTTP_TIMEOUT_SECONDS` (default `30`)
- `LOG_LEVEL` (default `INFO`)

## Run locally
```bash
cd services/agent_service
uvicorn app.main:app --reload --port 8001
```
