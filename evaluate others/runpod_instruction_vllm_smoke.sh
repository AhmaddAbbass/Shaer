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
export LD_LIBRARY_PATH="/usr/local/lib/python3.11/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python}"
SUBSET_JSONL="${SUBSET_JSONL:-evaluation/outputs/full_baseline_manifest_unique_prompts.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-evaluation/outputs/runpod_instruction_vllm_$(date -u +%Y%m%dT%H%M%SZ)}"
LIMIT_ROWS="${LIMIT_ROWS:-50}"
SAMPLES_PER_ROW="${SAMPLES_PER_ROW:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
TEMPERATURE="${TEMPERATURE:-0.55}"
TOP_P="${TOP_P:-0.9}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.08}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_ID="${MODEL_ID:-Navid-AI/Yehia-7B-preview}"
HF_OWNER="${HF_OWNER:-Shaer-AI}"
PRIVATE_UPLOAD="${PRIVATE_UPLOAD:-0}"

mkdir -p "$OUTPUT_ROOT/logs"
LOG_DIR="$OUTPUT_ROOT/logs"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Put it in /root/.env or export HF_TOKEN before running." >&2
  exit 1
fi

if pgrep -af 'runpod_generate_upload_raw_chunked|generate_continuation_baselines' >/dev/null; then
  echo "ERROR: continuation generation is still running. Stop/wait before vLLM smoke." >&2
  pgrep -af 'runpod_generate_upload_raw_chunked|generate_continuation_baselines' >&2 || true
  exit 1
fi

if [[ ! -f "$SUBSET_JSONL" ]]; then
  "$PYTHON_BIN" "evaluate others/build_full_baseline_manifest.py" \
    --continuation-policy auto \
    --per-source first \
    --scored-repo "" \
    --output-jsonl "$SUBSET_JSONL"
fi

UPLOAD_PRIVATE_ARGS=()
if [[ "$PRIVATE_UPLOAD" == "1" ]]; then
  UPLOAD_PRIVATE_ARGS+=(--private)
fi

prepare_upload_dir() {
  local bundle_dir="$1"
  local upload_dir="$2"
  local model_name="$3"
  local repo_id="$4"
  rm -rf "$upload_dir"
  mkdir -p "$upload_dir/data" "$upload_dir/artifacts"
  ln -f "$bundle_dir/generations_scored.jsonl" "$upload_dir/data/test.jsonl"
  for name in generations.jsonl validation.json aggregate.json bundle_metadata.json; do
    if [[ -f "$bundle_dir/$name" ]]; then
      ln -f "$bundle_dir/$name" "$upload_dir/artifacts/$name"
    fi
  done
  cat > "$upload_dir/README.md" <<README
---
pretty_name: Shaer Yehia vLLM Instruction Smoke - $model_name
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.jsonl
---

# Shaer Yehia vLLM Instruction Smoke - $model_name

This dataset contains a $LIMIT_ROWS-row vLLM smoke generation for \\`$model_name\\`.

The main table is \\`data/test.jsonl\\`; it includes source/prompt/reference fields, model output in \\`generated_text\\`, prompt variant metadata, and meter/count reward columns.

Artifacts such as validation and aggregate metrics are stored under \\`artifacts/\\`.
README
}

run_variant() {
  local variant="$1"
  local model_name="$2"
  local repo_id="$3"
  local run_dir="$OUTPUT_ROOT/$model_name"
  local bundle_dir="$OUTPUT_ROOT/${model_name}_bundle"
  local upload_dir="$OUTPUT_ROOT/${model_name}_hf_dataset"

  echo "============================================================"
  echo "VARIANT=$variant MODEL=$model_name REPO=$repo_id LIMIT_ROWS=$LIMIT_ROWS"
  echo "RUN_DIR=$run_dir"
  echo "============================================================"

  "$PYTHON_BIN" "evaluate others/generate_instruction_vllm.py" \
    --subset-jsonl "$SUBSET_JSONL" \
    --run-dir "$run_dir" \
    --prompt-variant "$variant" \
    --model-name "$model_name" \
    --model-id "$MODEL_ID" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --limit-rows "$LIMIT_ROWS" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    2>&1 | tee -a "$LOG_DIR/${model_name}_generate.log"

  "$PYTHON_BIN" "evaluate others/validate_baseline_generations.py" \
    --input-jsonl "$run_dir/instruction_generations.jsonl" \
    --expected-models "$model_name" \
    --expected-source-rows "$LIMIT_ROWS" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --summary-json "$run_dir/instruction_generations.validation.json" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_validate.log"

  "$PYTHON_BIN" "evaluate others/export_model_results_bundle.py" \
    --input-jsonl "$run_dir/instruction_generations.jsonl" \
    --output-dir "$bundle_dir" \
    --expected-source-rows "$LIMIT_ROWS" \
    --samples-per-row "$SAMPLES_PER_ROW" \
    --dataset-repo-id "$repo_id" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_bundle.log"

  prepare_upload_dir "$bundle_dir" "$upload_dir" "$model_name" "$repo_id"

  "$PYTHON_BIN" "evaluate others/upload_dataset_bundle.py" \
    --folder "$upload_dir" \
    --repo-id "$repo_id" \
    "${UPLOAD_PRIVATE_ARGS[@]}" \
    --commit-message "Upload Yehia vLLM $variant smoke with scoring" \
    2>&1 | tee -a "$LOG_DIR/${model_name}_upload.log"
}

run_variant plain yehia_base_plain_instruction "${REPO_YEHIA_PLAIN:-$HF_OWNER/shaer-eval-instruction-yehia-base-plain}"
run_variant chat yehia_base_chat_template "${REPO_YEHIA_CHAT:-$HF_OWNER/shaer-eval-instruction-yehia-base-chat}"

echo "DONE output_root=$OUTPUT_ROOT"
