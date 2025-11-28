# shaer_service

FastAPI gateway that forwards `/generate-bayt` calls to the hosted Shaer RunPod endpoint while preserving the exact training-style prompt (system + user).

## What it does
- Builds the Shaer prompt with meter, description, era, poet/style, num_verses, sequence_number, and previous verses (with sensible defaults for era/poet).
- Calls RunPod `/runsync` with your Shaer endpoint ID, sending temperature/top_p/max_tokens.
- Returns the generated bayt (`verse_text`) and echoes `sequence_number`; no state is stored.

## API
### POST /generate-bayt
Request
```json
{
  "poem_meter": "OU,O\"O3USO�",
  "poem_description": "U,O�USO_Oc O1U+ O'U^U, OU,U.O�O�O�O\".",
  "num_verses": 8,
  "sequence_number": 3,
  "previous_verses": ["..."],
  "poem_era": "OU,O1O�O� OU,O-O_USO�",
  "poet_name": "O�O3U,U^O\" U,O�USO\" ..."
}
```
Response
```json
{
  "verse_text": "O�O_O� ... O1O�O�",
  "sequence_number": 3
}
```

## Environment
- Required: `RUNPOD_API_KEY`, `SHAER_ENDPOINT_ID`
- Optional:
  - `SHAER_RUNPOD_BASE_URL` (default `https://api.runpod.ai/v2`)
  - `SHAER_MAX_NEW_TOKENS` (default 64)
  - `SHAER_TEMPERATURE` (default 0.7)
  - `SHAER_TOP_P` (default 0.9)
  - `SHAER_RUNPOD_TIMEOUT` (default 120s)

## Run locally
```bash
cd services/shaer_service
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

## Docker
```bash
docker build -t shaer_service -f services/shaer_service/Dockerfile .
docker run --rm -p 8002:8002 --env-file services/shaer_service/.env shaer_service
```
- Container listens on port `8002`. Provide your own env file; do not commit secrets.

## Notes
- Startup touches the client to fail fast if env vars are missing.
- Prompt defaults fill missing era/poet with built-in Arabic defaults.
- RunPod responses are parsed defensively to extract generated text; errors if none found.
