# Ashaar Meter Service

FastAPI microservice that wraps `Models/Ashaar_runtime/ashaar_only` utilities to compute Ashaar structural scores for a bayt.

## Endpoints

* `GET /health` – returns `{status, assets_loaded}`.
* `POST /ashaar-score` – body `{ "verse_text": "..." }`, response `{ "result": { "reward": 0.xx, "ashaar_score": 0.xx, "notes": "..." } }`.

## Local run

```bash
cd services/ashaar_meter_service
uvicorn app.main:app --reload --port 8005
```

Environment variables (optional):

| Name | Description | Default |
| ---- | ----------- | ------- |
| `ASHAAR_WEIGHT` | Multiplier applied to Ashaar score before clamping to [0, 1]. | `1.0` |

## Tests

```bash
pytest services/ashaar_meter_service/tests
```
