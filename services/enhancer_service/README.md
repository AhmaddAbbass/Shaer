# Enhancer Service

Bayt-level repair service that adjusts a `BaytGenerationRequest` using meter and Yehia feedback, then re-calls `shaer_service` to obtain an improved verse.

## Endpoints

- `GET /health` – basic heartbeat.
- `POST /enhance-bayt` – accepts an `EnhanceBaytRequest` and returns a single improved bayt candidate.

## Environment variables (prefixed with `ENHANCER_`)

| Variable | Default | Description |
| --- | --- | --- |
| `SHAER_BASE_URL` | `http://localhost:8102` | Base URL of `shaer_service` |
| `REQUEST_TIMEOUT_SECONDS` | `30` | HTTP timeout for downstream calls |
| `MAX_EXTRA_GUIDANCE` | `2` | Max guidance bullets appended to the prompt |
| `ENABLE_DESCRIPTION_TIGHTENING` | `true` | Whether to tighten poem descriptions based on feedback |
| `METER_FOCUS_THRESHOLD` | `80` | Meter score threshold that triggers meter emphasis |

Run locally with:

```bash
cd services/enhancer_service
uvicorn app.main:app --reload --port 8105
```
