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

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
SUBSET_JSONL="${SUBSET_JSONL:-evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/outputs/runpod_raw_continuation_$(date -u +%Y%m%dT%H%M%SZ)}"
SAMPLES_PER_ROW="${SAMPLES_PER_ROW:-1}"
LIMIT_ROWS="${LIMIT_ROWS:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"
HF_OWNER="${HF_OWNER:-Shaer-AI}"
PRIVATE_UPLOAD="${PRIVATE_UPLOAD:-0}"

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

if [[ -n "$LIMIT_ROWS" ]]; then
  EXPECTED_SOURCE_ROWS="$LIMIT_ROWS"
else
  EXPECTED_SOURCE_ROWS=$("$PYTHON_BIN" - <<PY2
from pathlib import Path
path = Path("$SUBSET_JSONL")
print(sum(1 for line in path.open(encoding="utf-8") if line.strip()))
PY2
)
fi

UPLOAD_PRIVATE_ARGS=()
if [[ "$PRIVATE_UPLOAD" == "1" ]]; then
  UPLOAD_PRIVATE_ARGS+=(--private)
fi

run_model() {
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
  echo "BUNDLE_DIR=$bundle_dir"
  echo "UPLOAD_DIR=$upload_dir"
  echo "============================================================"

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
    ${LIMIT_ROWS:+--limit-rows "$LIMIT_ROWS"} \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_DIR/${model_name}_generate.log"

  "$PYTHON_BIN" "evaluate others/validate_baseline_generations.py" \
    --input-jsonl "$generations" \
    --expected-models "$model_name" \
    --expected-source-rows "$EXPECTED_SOURCE_ROWS" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --summary-json "$run_dir/continuation_generations.validation.json" \
    2>&1 | tee "$LOG_DIR/${model_name}_validate.log"

  "$PYTHON_BIN" "evaluate others/export_model_results_bundle.py" \
    --input-jsonl "$generations" \
    --output-dir "$bundle_dir" \
    --expected-source-rows "$EXPECTED_SOURCE_ROWS" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --dataset-repo-id "$repo_id" \
    --skip-score \
    2>&1 | tee "$LOG_DIR/${model_name}_bundle.log"

  rm -rf "$upload_dir"
  mkdir -p "$upload_dir/data" "$upload_dir/artifacts"
  cp "$bundle_dir/generations.jsonl" "$upload_dir/data/test.jsonl"
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

This dataset contains one row per evaluated prompt for \`$model_name\`.

The main table is \`data/test.jsonl\`; it includes the source/prompt/reference fields plus the model output in \`generated_text\` and \`raw_generated_text\`. Reward/evaluation columns are intentionally not included yet and can be added later.

Artifacts such as validation and CSV exports are stored under \`artifacts/\`.
README

  "$PYTHON_BIN" "evaluate others/upload_dataset_bundle.py" \
    --folder "$upload_dir" \
    --repo-id "$repo_id" \
    "${UPLOAD_PRIVATE_ARGS[@]}" \
    --commit-message "Upload raw continuation generation table for $model_name" \
    2>&1 | tee "$LOG_DIR/${model_name}_upload.log"
}

run_model "fanar_2_diwan_prefix" "${BATCH_FANAR:-8}" "${REPO_FANAR:-$HF_OWNER/shaer-eval-raw-fanar-diwan-prefix}" --trust-remote-code
run_model "ashaar_model" "${BATCH_ASHAAR:-32}" "${REPO_ASHAAR:-$HF_OWNER/shaer-eval-raw-ashaar-model}"
run_model "gpt2_small_arabic_poetry" "${BATCH_GPT2_SMALL:-64}" "${REPO_GPT2_SMALL:-$HF_OWNER/shaer-eval-raw-gpt2-small-arabic-poetry}"
run_model "gpt2_medium_arabic_poetry" "${BATCH_GPT2_MEDIUM:-32}" "${REPO_GPT2_MEDIUM:-$HF_OWNER/shaer-eval-raw-gpt2-medium-arabic-poetry}"

echo "DONE output_root=$OUTPUT_ROOT"
