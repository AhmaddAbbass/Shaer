#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?usage: run_full_judge_all.sh RUN_DIR [WORKERS] [JUDGE_MODEL]}"
WORKERS="${2:-4}"
JUDGE_MODEL="${3:-qwen/qwen3-235b-a22b-2507}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAIN_LOG="$RUN_DIR/main.log"

mkdir -p "$RUN_DIR"

set -a
if [ -f "$PROJECT_ROOT/.env" ]; then
  . "$PROJECT_ROOT/.env"
fi
set +a

export JUDGE_REQUEST_SLEEP_SECONDS="${JUDGE_REQUEST_SLEEP_SECONDS:-0.25}"
export JUDGE_MAX_RETRIES="${JUDGE_MAX_RETRIES:-4}"
export JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-120}"

cd "$PROJECT_ROOT"

python evaluation/full_judge_orchestrator.py \
  --run-dir "$RUN_DIR" \
  --workers "$WORKERS" \
  --judge-model "$JUDGE_MODEL" \
  2>&1 | tee -a "$MAIN_LOG"
