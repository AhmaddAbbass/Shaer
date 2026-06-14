#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/Shaer_clean

if [[ -f /root/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/.env
  set +a
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PYTHONUNBUFFERED=1
export METER_MODEL_ID="${METER_MODEL_ID:-Shaer-AI/4BiLSTM-meter-classification-pytorch}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
SUBSET_JSONL="${SUBSET_JSONL:-evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/outputs/runpod_raw_chunked_$(date -u +%Y%m%dT%H%M%SZ)}"
SAMPLES_PER_ROW="${SAMPLES_PER_ROW:-1}"
CHUNK_ROWS="${CHUNK_ROWS:-200}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"
HF_OWNER="${HF_OWNER:-Shaer-AI}"
PRIVATE_UPLOAD="${PRIVATE_UPLOAD:-0}"
SCORE_CHUNKS="${SCORE_CHUNKS:-0}"
SCORE_FINAL="${SCORE_FINAL:-1}"

mkdir -p "$OUTPUT_ROOT"
LOG_DIR="$OUTPUT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Put it in /root/.env or export HF_TOKEN before running." >&2
  exit 1
fi

if [[ ! -f "$SUBSET_JSONL" ]]; then
  "$PYTHON_BIN" "evaluate others/build_full_baseline_manifest.py" \
    --continuation-policy auto \
    --per-source first \
    --scored-repo "" \
    --output-jsonl "$SUBSET_JSONL"
fi

TOTAL_ROWS=$("$PYTHON_BIN" - <<PY2
from pathlib import Path
path = Path("$SUBSET_JSONL")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY2
)
if [[ -n "$LIMIT_ROWS" ]]; then
  TOTAL_ROWS="$LIMIT_ROWS"
fi

UPLOAD_PRIVATE_ARGS=()
if [[ "$PRIVATE_UPLOAD" == "1" ]]; then
  UPLOAD_PRIVATE_ARGS+=(--private)
fi

row_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
  else
    wc -l < "$path"
  fi
}

prepare_upload_dir() {
  local model_name="$1"
  local repo_id="$2"
  local bundle_dir="$3"
  local upload_dir="$4"
  local chunk_end="$5"
  local score_mode="$6"

  rm -rf "$upload_dir"
  mkdir -p "$upload_dir/data" "$upload_dir/artifacts"
  if [[ "$score_mode" == "1" && -f "$bundle_dir/generations_scored.jsonl" ]]; then
    cp "$bundle_dir/generations_scored.jsonl" "$upload_dir/data/test.jsonl"
    cp "$bundle_dir/generations_scored.csv" "$upload_dir/artifacts/generations_scored.csv"
  else
    cp "$bundle_dir/generations.jsonl" "$upload_dir/data/test.jsonl"
  fi
  cp "$bundle_dir/generations.csv" "$upload_dir/artifacts/generations.csv"
  cp "$bundle_dir/validation.json" "$upload_dir/artifacts/validation.json"
  cp "$bundle_dir/aggregate.json" "$upload_dir/artifacts/aggregate.json"
  cp "$bundle_dir/bundle_metadata.json" "$upload_dir/artifacts/bundle_metadata.json"
  cat > "$upload_dir/README.md" <<README
---
pretty_name: Raw Shaer Continuation Generations - $model_name
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.jsonl
---

# Raw Shaer Continuation Generations - $model_name

This dataset contains cumulative raw continuation generations for \`$model_name\`.

The main table is \`data/test.jsonl\`; it includes source/prompt/reference fields plus the model output in \`generated_text\` and \`raw_generated_text\`.

Current uploaded progress target: \`$chunk_end\` rows out of \`$TOTAL_ROWS\`.
Scored upload: \`$score_mode\`. When scored upload is true, reward/evaluation columns are included in the main table.

Artifacts such as validation and CSV exports are stored under \`artifacts/\`.
README
}

upload_bundle() {
  local model_name="$1"
  local repo_id="$2"
  local run_dir="$3"
  local bundle_dir="$4"
  local upload_dir="$5"
  local expected_rows="$6"
  local score_mode="$7"
  local generations="$run_dir/continuation_generations.jsonl"
  local bundle_input="$generations"
  if [[ "$model_name" == "ashaar_model" && "$score_mode" == "1" ]]; then
    bundle_input="$run_dir/continuation_generations_ashaar_normalized.jsonl"
    "$PYTHON_BIN" "evaluate others/normalize_ashaar_generations.py" \
      --input-jsonl "$generations" \
      --output-jsonl "$bundle_input" \
      2>&1 | tee -a "$LOG_DIR/${model_name}_normalize.log"
  fi

  "$PYTHON_BIN" "evaluate others/validate_baseline_generations.py" \
    --input-jsonl "$bundle_input" \
    --expected-models "$model_name" \
    --expected-source-rows "$expected_rows" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --summary-json "$run_dir/continuation_generations.validation.json" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_validate_upload.log"

  local bundle_cmd=(
    "$PYTHON_BIN" "evaluate others/export_model_results_bundle.py"
    --input-jsonl "$bundle_input"
    --output-dir "$bundle_dir"
    --expected-source-rows "$expected_rows"
    --samples-per-row "$SAMPLES_PER_ROW"
    --dataset-repo-id "$repo_id"
  )
  if [[ "$score_mode" != "1" ]]; then
    bundle_cmd+=(--skip-score)
  fi
  "${bundle_cmd[@]}" 2>&1 | tee -a "$LOG_DIR/${model_name}_bundle_upload.log"

  prepare_upload_dir "$model_name" "$repo_id" "$bundle_dir" "$upload_dir" "$expected_rows" "$score_mode"

  "$PYTHON_BIN" "evaluate others/upload_dataset_bundle.py" \
    --folder "$upload_dir" \
    --repo-id "$repo_id" \
    "${UPLOAD_PRIVATE_ARGS[@]}" \
    --commit-message "Upload raw $model_name generations through row $expected_rows" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_hf_upload.log"
}

run_model_chunked() {
  local model_name="$1"
  local batch_size="$2"
  local repo_id="$3"
  shift 3
  local extra_args=("$@")
  local run_dir="$OUTPUT_ROOT/$model_name"
  local bundle_dir="$OUTPUT_ROOT/${model_name}_bundle"
  local upload_dir="$OUTPUT_ROOT/${model_name}_hf_dataset"
  local generations="$run_dir/continuation_generations.jsonl"

  echo "============================================================"
  echo "MODEL=$model_name BATCH_SIZE=$batch_size REPO=$repo_id"
  echo "RUN_DIR=$run_dir"
  echo "CHUNK_ROWS=$CHUNK_ROWS TOTAL_ROWS=$TOTAL_ROWS"
  echo "============================================================"

  local chunk_end="$CHUNK_ROWS"
  while [[ "$chunk_end" -lt "$TOTAL_ROWS" ]]; do
    local existing
    existing=$(row_count "$generations")
    if [[ "$existing" -lt "$chunk_end" ]]; then
      "$PYTHON_BIN" "evaluate others/generate_continuation_baselines.py" \
        --subset-jsonl "$SUBSET_JSONL" \
        --models "$model_name" \
        --run-dir "$run_dir" \
        --samples-per-row "$SAMPLES_PER_ROW" \
        --batch-size "$batch_size" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --repetition-penalty "$REPETITION_PENALTY" \
        --limit-rows "$chunk_end" \
        "${extra_args[@]}" \
        2>&1 | tee -a "$LOG_DIR/${model_name}_generate.log"
    fi
    existing=$(row_count "$generations")
    upload_bundle "$model_name" "$repo_id" "$run_dir" "$bundle_dir" "$upload_dir" "$existing" "$SCORE_CHUNKS"
    chunk_end=$((chunk_end + CHUNK_ROWS))
  done

  "$PYTHON_BIN" "evaluate others/generate_continuation_baselines.py" \
    --subset-jsonl "$SUBSET_JSONL" \
    --models "$model_name" \
    --run-dir "$run_dir" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --batch-size "$batch_size" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --limit-rows "$TOTAL_ROWS" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_generate.log"
  upload_bundle "$model_name" "$repo_id" "$run_dir" "$bundle_dir" "$upload_dir" "$TOTAL_ROWS" "$SCORE_FINAL"
}

run_model_chunked "fanar_2_diwan_prefix" "${BATCH_FANAR:-24}" "${REPO_FANAR:-$HF_OWNER/shaer-eval-raw-fanar-diwan-prefix}" --trust-remote-code
run_model_chunked "ashaar_model" "${BATCH_ASHAAR:-48}" "${REPO_ASHAAR:-$HF_OWNER/shaer-eval-raw-ashaar-model}"
run_model_chunked "gpt2_small_arabic_poetry" "${BATCH_GPT2_SMALL:-96}" "${REPO_GPT2_SMALL:-$HF_OWNER/shaer-eval-raw-gpt2-small-arabic-poetry}"
run_model_chunked "gpt2_medium_arabic_poetry" "${BATCH_GPT2_MEDIUM:-64}" "${REPO_GPT2_MEDIUM:-$HF_OWNER/shaer-eval-raw-gpt2-medium-arabic-poetry}"

echo "DONE output_root=$OUTPUT_ROOT"
