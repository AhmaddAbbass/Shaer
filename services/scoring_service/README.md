# Scoring Service

Aggregates bayt-level evaluation scores by calling `meter_service` and `aspect_evaluator_service`, then applies the fixed weighted formula:

```
FinalScore = 0.5 * meter + 0.15 * meaning + 0.15 * fluency + 0.10 * poeticness + 0.10 * cohesion
```

**Pass condition:** `final_score >= 0.7` (configurable via `SCORING_PASS_THRESHOLD`).

## Endpoints

- `GET /health`
- `POST /score`
  ```json
  {
    "verse_text": "يا قلبُ هل تذكرُ الأيامَ والوطنا",
    "spec": {
      "poem_meter": "الكامل",
      "poem_description": "قصيدة حنين",
      "num_verses": 6
    },
    "previous_verses": ["سلامٌ على دارٍ تفتّح طيبُها"],
    "use_openai_scoring": true
  }
  ```

Response:
```json
{
  "meter_eval": { "meter_score": 83, "on_meter": true, "notes": "..." },
  "aspects": {
    "meaning": { "yehia_feedback": {...}, "judge": {"score_0_1": 0.82, "notes": "..."} },
    "cohesion": { ... },
    "fluency": { ... },
    "poeticness": { ... }
  },
  "final_score": 0.78,
  "passed": true,
  "breakdown": {
    "meter_score": 0.41,
    "meaning": 0.12,
    "fluency": 0.11,
    "poeticness": 0.08,
    "cohesion": 0.06,
    "final_score": 0.78
  }
}
```

## Environment variables (`SCORING_` prefix)

| Variable | Default | Description |
| --- | --- | --- |
| `PASS_THRESHOLD` | `0.7` | Final score cutoff |
| `METER_BASE_URL` | `http://localhost:8104` | `meter_service` URL |
| `ASPECT_EVAL_BASE_URL` | `http://localhost:8106` | `aspect_evaluator_service` URL |
| `HTTP_TIMEOUT_SECONDS` | `30` | Downstream HTTP timeout |

Run locally:
```bash
cd services/scoring_service
uvicorn app.main:app --reload --port 8107
```
