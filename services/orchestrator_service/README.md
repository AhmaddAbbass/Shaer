# Orchestrator Service

Single entry point for the front-end. Routes intents and orchestrates downstream microservices:
`rag_service`, `yehia_service`, `shaer_service`, `scoring_service`, and `enhancer_service`.

## Endpoints

- `GET /health`
- `POST /poem/generate`
- `POST /bayt/fix`
- `POST /bayt/score`
- `POST /library/search`

## Environment variables (`ORCH_` prefix)

| Variable | Default | Description |
| --- | --- | --- |
| `RAG_SERVICE_URL` | `http://localhost:8003` | RAG search service |
| `YEHIA_SERVICE_URL` | `http://localhost:8101` | Yehia gateway |
| `SHAER_SERVICE_URL` | `http://localhost:8102` | Shaer gateway |
| `SCORING_SERVICE_URL` | `http://localhost:8107` | Aggregated scoring service |
| `ENHANCER_SERVICE_URL` | `http://localhost:8105` | Enhancer loop service |
| `DEFAULT_TOP_K_RAG` | `5` | Inspiration hits |
| `DEFAULT_NUM_VERSES` | `6` | Default target length |
| `MAX_RETRIES_PER_BAYT` | `3` | Enhancer retries |

Run locally:
```bash
cd services/orchestrator_service
uvicorn app.main:app --reload --port 8000
```
