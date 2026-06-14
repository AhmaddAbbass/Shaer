#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/preprocess_grpo
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LOG="outputs/preprocess_grpo/preprocess_${RUN_TS}.log"
PYTHON="../sft/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing python at $PYTHON" >&2
  exit 1
fi

export PATH="$(cd ../sft/.venv/bin && pwd):$PATH"
export SFT_ADAPTER_REPO="${SFT_ADAPTER_REPO:-Shaer-AI/Shaer-adapters}"
export SFT_ADAPTER_MODE="${SFT_ADAPTER_MODE:-fresh_sft/train}"
export GRPO_PREPROCESS_DATASET_ID="${GRPO_PREPROCESS_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
export GRPO_PREPROCESS_OUTPUT_DATASET_ID="${GRPO_PREPROCESS_OUTPUT_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits-grpo-preprocessed-v1}"

setsid "$PYTHON" -u preprocess_grpo_difficulty.py > "$LOG" 2>&1 < /dev/null &
PID=$!

echo "preprocess_grpo_difficulty started"
echo "PID=$PID"
echo "LOG=$LOG"
echo "$PID" > outputs/preprocess_grpo/latest.pid
echo "$LOG" > outputs/preprocess_grpo/latest.log
