# Aspect Evaluator Service

Evaluates a generated bayt across four semantic aspects (meaning, cohesion, fluency, poeticness). Each aspect uses Yehia for Arabic feedback and OpenAI (gpt-4o-mini) for a normalized numeric score.

## Endpoints

- `GET /health`
- `POST /evaluate-bayt` – returns evaluations for all four aspects.
- `POST /evaluate-aspect` – evaluates a single aspect (debugging aid).

## Request example

```json
{
  "verse_text": "قفا نبك من ذكرى حبيب ومنزل",
  "spec": {"poem_meter": "الكامل", "poem_description": "قصيدة في الحنين", "num_verses": 6},
  "previous_verses": ["سلام على الدار التي كنت أعشق"],
  "use_openai_scoring": true
}
```

## Environment variables (`ASPECT_EVAL_` prefix)

| Variable | Default | Description |
| --- | --- | --- |
| `YEHIA_BASE_URL` | `http://localhost:8101` | Yehia service URL |
| `REQUEST_TIMEOUT_SECONDS` | `30` | HTTP timeout for downstream calls |
| `OPENAI_API_KEY` | `null` | API key for OpenAI scoring (required for numeric scores) |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used for scoring |
| `OPENAI_TEMPERATURE` | `0.0` | Temperature for scoring calls |

Run locally:
```bash
cd services/aspect_evaluator_service
uvicorn app.main:app --reload --port 8106
```
