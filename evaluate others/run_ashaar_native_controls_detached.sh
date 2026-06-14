#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-ashaar_native_controls_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/evaluation/outputs/$RUN_NAME}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

REPO_ID="${REPO_ID:-Shaer-AI/shaer-eval-ashaar-native-controls}"
SOURCE_DATASET="${SOURCE_DATASET:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
METER_MODEL_ID="${METER_MODEL_ID:-Shaer-AI/4BiLSTM-meter-classification-pytorch}"
SAMPLES_PER_ROW="${SAMPLES_PER_ROW:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
UPLOAD_EVERY_ROWS="${UPLOAD_EVERY_ROWS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"

MANIFEST="$RUN_DIR/ashaar_native_manifest_controls.jsonl"
GENERATIONS="$RUN_DIR/ashaar_native_generations.jsonl"
NORMALIZED="$RUN_DIR/ashaar_native_generations_normalized.jsonl"
SCORED="$RUN_DIR/ashaar_native_generations_scored.jsonl"
BUNDLE_DIR="$RUN_DIR/bundle"
STATUS="$RUN_DIR/pipeline_status.json"

write_status() {
  local status="$1"
  local step="$2"
  local message="${3:-}"
  "$PYTHON_BIN" - "$STATUS" "$status" "$step" "$message" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status, step, message = sys.argv[1:5]
payload = {
    "status": status,
    "step": step,
    "message": message,
    "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY
}

run_step() {
  local name="$1"
  shift
  write_status "running" "$name"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $name" | tee -a "$LOG_DIR/pipeline.log"
  "$@" 2>&1 | tee -a "$LOG_DIR/${name}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE $name" | tee -a "$LOG_DIR/pipeline.log"
}

{
  write_status "running" "bootstrap"
  echo "run_dir=$RUN_DIR"
  echo "repo_id=$REPO_ID"
  echo "source_dataset=$SOURCE_DATASET"
  echo "meter_model_id=$METER_MODEL_ID"

  "$PYTHON_BIN" -m pip install \
    "transformers==4.43.0" \
    "datasets==2.20.0" \
    "huggingface_hub==0.23.4" \
    "tokenizers==0.19.1" \
    "safetensors==0.4.3" \
    "accelerate==0.31.0" \
    "pandas==2.2.2" \
    "python-dotenv==1.0.1" \
    "tqdm==4.66.4" \
    "sentencepiece==0.2.0" \
    --quiet

  export METER_MODEL_ID

  run_step build_manifest "$PYTHON_BIN" "evaluate others/build_ashaar_native_manifest.py" \
    --source-dataset "$SOURCE_DATASET" \
    --split test \
    --prompt-mode controls \
    --output-jsonl "$MANIFEST"

  run_step generate "$PYTHON_BIN" "evaluate others/generate_ashaar_native_baselines.py" \
    --manifest-jsonl "$MANIFEST" \
    --run-dir "$RUN_DIR" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --upload-every-rows "$UPLOAD_EVERY_ROWS" \
    --upload-repo-id "$REPO_ID"

  run_step normalize "$PYTHON_BIN" "evaluate others/normalize_ashaar_generations.py" \
    --input-jsonl "$GENERATIONS" \
    --output-jsonl "$NORMALIZED"

  run_step score "$PYTHON_BIN" "evaluate others/score_baseline_meter_count.py" \
    --input-jsonl "$NORMALIZED" \
    --output-jsonl "$SCORED"

  run_step bundle "$PYTHON_BIN" "evaluate others/export_model_results_bundle.py" \
    --input-jsonl "$SCORED" \
    --output-dir "$BUNDLE_DIR" \
    --expected-source-rows 3481 \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --dataset-repo-id "$REPO_ID" \
    --source-dataset "$SOURCE_DATASET"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    run_step upload "$PYTHON_BIN" "evaluate others/upload_dataset_bundle.py" \
      --folder "$BUNDLE_DIR" \
      --repo-id "$REPO_ID" \
      --commit-message "Upload Ashaar native-control evaluation results"
    write_status "completed" "done" "uploaded to $REPO_ID"
  else
    write_status "needs_hf_token" "upload" "Generation/scoring/bundle completed, but HF_TOKEN is missing so upload was skipped."
    echo "HF_TOKEN missing; upload skipped. Bundle is at $BUNDLE_DIR" | tee -a "$LOG_DIR/pipeline.log"
  fi
} >> "$LOG_DIR/pipeline.stdout.log" 2>> "$LOG_DIR/pipeline.stderr.log"
