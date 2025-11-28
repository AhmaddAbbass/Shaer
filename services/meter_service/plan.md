# Meter Service Implementation Plan (meter reward)

Goal: Implement `/eval-bayt` that returns `BaytMeterEval` using the existing BiLSTM meter classifier (see `utils/rewards/meter_reward.py` + `Models/Ashaar_runtime/bilstm_only/bilstm_model/training` assets).

## Approach
- Reuse the classifier-only reward: `classifier_meter_scores(text)` → probability dist over meters.
- Map target meter prob to a 0–100 `meter_score`; `on_meter` = `score >= threshold` (env).
- Provide short Arabic `notes` summarizing pass/fail and the top meter guess.
- Load model/label encoder/vocab once on startup; keep CPU-only TensorFlow as in util.

## API & Schemas
- `POST /eval-bayt`
  - Request: `{ "verse_text": str, "target_meter": str }`
  - Response: `{ "result": BaytMeterEval }` (from `services.common_schemas.schemas`)
- `GET /health`
  - Checks model loaded + simple self-test (no heavy inference; maybe returns `ok` boolean).
- Define service-local Pydantic models in `app/schemas.py` wrapping the shared types.

## Modules to Implement
- `app/config.py`
  - Env vars: `METER_MODEL_PATH`, `METER_LABEL_ENCODER_PATH`, `METER_VOCAB_CONFIG_PATH`, `METER_SCORE_THRESHOLD` (default 80), `METER_MAX_TOP` (top-N meters to include in notes), `LOG_LEVEL`.
- `app/logging.py`
  - Basic formatter + `get_logger`.
- `app/scansion.py`
  - Thin wrapper around `classifier_meter_scores`.
  - Functions:
    - `load_assets()` on import/startup.
    - `evaluate_bayt(text, target_meter, threshold) -> BaytMeterEval`.
    - Build notes: if pass → “على البحر المطلوب”; if fail → mention best-matching meter + score.
- `app/api.py`
  - FastAPI router with `/eval-bayt` and `/health`.
  - Inject settings and a singleton scansion helper via `Depends`.
- `app/main.py`
  - Create FastAPI app, configure logging, attach router, handle shutdown if needed.

## Tests
- Unit tests in `services/meter_service/tests/test_eval_bayt.py`:
  - Mock `classifier_meter_scores` to avoid heavy TF load.
  - Happy path: high prob on target → `on_meter` True, score >= threshold.
  - Fail path: low prob → `on_meter` False, notes mention top meter.
  - Input validation: empty verse or meter returns 422.
- Optionally a smoke test that patches paths to small dummy assets if available.

## Ops/Docs
- Update `services/meter_service/README.md` with endpoint examples, env vars, run command: `uvicorn app.main:app --port 8004`.
- Consider adding Dockerfile mirroring other services (python:3.12-slim, install TF deps).

## Open Questions / Assumptions
- Accept slight meter label mismatches: we’ll use case-insensitive match when checking target key against label encoder output.
- Threshold default 80 (matches earlier narrative); adjustable via env.
- Running on CPU is fine; no GPU assumed for this service.
