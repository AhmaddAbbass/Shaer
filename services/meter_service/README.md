Meter Service
=============

FastAPI microservice that exposes `/eval-bayt` and `/health` using the BiLSTM meter classifier from `Models/Ashaar_runtime/bilstm_only`.

### Run locally

```bash
cd services/meter_service
uvicorn app.main:app --reload --port 8004
```

### Env vars

See `app/config.py` for defaults (`METER_MODEL_PATH`, `METER_LABEL_ENCODER_PATH`, `METER_VOCAB_CONFIG_PATH`, `METER_SCORE_THRESHOLD`, etc.).
