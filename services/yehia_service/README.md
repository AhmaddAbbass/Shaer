# yehia_service

FastAPI gateway to the Yehia RunPod endpoint. Provides three tools for the orchestrator:
- `/chat` — generic chat
- `/build-spec` — infer `PoemSpec` from user query + optional RAG hits
- `/feedback` — critique a bayt against a `PoemSpec`

Inference is proxied to RunPod; no local weights/GPU needed.

## API
### POST /chat
Request:
```json
{ "messages": [ { "role": "user", "content": "..." } ] }
```
Response: `{ "text": "<generated text>" }`

### POST /build-spec
Request:
```json
{
  "user_query": "قصيدة عن ...",
  "rag_hits": [ { "poem_id": "1", "poet_name": "...", "poem_description": "...", "poem_meter": "..." } ]
}
```
Response: `{ "spec": PoemSpec }` (always non-empty meter/description, num_verses > 0; defaults applied if missing).

### POST /feedback
Request:
```json
{
  "verse_text": "بيت شعر ...",
  "spec": { "poem_meter": "...", "poem_description": "...", "num_verses": 4 }
}
```
Response: `{ "feedback": { "ok": bool, "score": int?, "feedback": "..." } }`

### GET /health
Returns `{"status": "ok"}` (config validated; no RunPod call).

## Environment
- Required: `RUNPOD_API_KEY`, `YEHIA_ENDPOINT_ID`
- Optional:
  - `YEHIA_RUNPOD_BASE_URL` (default `https://api.runpod.ai/v2`)
  - `YEHIA_MAX_NEW_TOKENS` (default 256)
  - `YEHIA_TEMPERATURE` (default 0.7)
  - `YEHIA_TOP_P` (default 0.9)
  - `YEHIA_RUNPOD_TIMEOUT` (default 120s)
  - `YEHIA_LOG_LEVEL` (default INFO)

## Run locally
```bash
cd services/yehia_service
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## Docker
```bash
docker build -t yehia_service -f services/yehia_service/Dockerfile .
docker run --rm -p 8001:8001 --env-file services/yehia_service/.env yehia_service
```
- Container listens on `8001`. Supply your own env file; do not commit secrets.

## Notes
- Prompts request JSON-style outputs for spec/feedback to parse into `PoemSpec` and `YehiaFeedback`.
- RunPod responses are parsed defensively; build-spec enforces defaults if fields are missing.
- Startup touches the client to fail fast on missing env vars (no external calls).
