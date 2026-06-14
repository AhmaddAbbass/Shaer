#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/preprocess_grpo_pipeline
RUN_TS=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="outputs/preprocess_grpo_pipeline/preprocess_${RUN_TS}"
mkdir -p "$RUN_DIR"

GEN_LOG="$RUN_DIR/generator_detached.log"
JUDGE_LOG="$RUN_DIR/judge_detached.log"
PYTHON="../sft/.venv/bin/python"
JUDGE_WORKERS="${JUDGE_WORKERS:-4}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing python at $PYTHON" >&2
  exit 1
fi

export PATH="$(cd ../sft/.venv/bin && pwd):$PATH"
export SFT_ADAPTER_REPO="${SFT_ADAPTER_REPO:-Shaer-AI/Shaer-adapters}"
export SFT_ADAPTER_MODE="${SFT_ADAPTER_MODE:-fresh_sft/train}"
export GRPO_PREPROCESS_DATASET_ID="${GRPO_PREPROCESS_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
export GRPO_PREPROCESS_OUTPUT_DATASET_ID="${GRPO_PREPROCESS_OUTPUT_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits-grpo-preprocessed-v1}"

setsid "$PYTHON" -u generate_grpo_candidates.py --run-dir "$RUN_DIR" > "$GEN_LOG" 2>&1 < /dev/null &
GEN_PID=$!

JUDGE_PIDS=()
JUDGE_LOGS=()
for ((i=0; i<JUDGE_WORKERS; i++)); do
  WORKER_ID=$(printf "judge%02d" "$i")
  WORKER_LOG="$RUN_DIR/${WORKER_ID}_detached.log"
  EXTRA_ARGS=()
  if [[ "$i" == "0" ]]; then
    EXTRA_ARGS+=(--publish)
  fi
  setsid "$PYTHON" -u judge_grpo_candidates.py --run-dir "$RUN_DIR" --worker-id "$WORKER_ID" "${EXTRA_ARGS[@]}" > "$WORKER_LOG" 2>&1 < /dev/null &
  JUDGE_PIDS+=("$!")
  JUDGE_LOGS+=("$WORKER_LOG")
done

echo "preprocess_grpo_pipeline started"
echo "RUN_DIR=$RUN_DIR"
echo "GENERATOR_PID=$GEN_PID"
echo "GENERATOR_LOG=$GEN_LOG"
for ((i=0; i<${#JUDGE_PIDS[@]}; i++)); do
  echo "JUDGE_PID_${i}=${JUDGE_PIDS[$i]}"
  echo "JUDGE_LOG_${i}=${JUDGE_LOGS[$i]}"
done

echo "$RUN_DIR" > outputs/preprocess_grpo_pipeline/latest.run_dir
echo "$GEN_PID" > outputs/preprocess_grpo_pipeline/latest.generator.pid
printf '%s\n' "${JUDGE_PIDS[@]}" > outputs/preprocess_grpo_pipeline/latest.judge.pids
echo "$GEN_LOG" > outputs/preprocess_grpo_pipeline/latest.generator.log
printf '%s\n' "${JUDGE_LOGS[@]}" > outputs/preprocess_grpo_pipeline/latest.judge.logs
