#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export SFT_ADAPTER_REPO="${SFT_ADAPTER_REPO:-Shaer-AI/Shaer-adapters}"
export SFT_ADAPTER_MODE="${SFT_ADAPTER_MODE:-fresh_sft/train}"
LOG_PATH="${1:-$ROOT_DIR/outputs/preprocess_grpo_pipeline/preprocess_20260408_174036/generator_detached.log}"
RUN_DIR="${2:-outputs/preprocess_grpo_pipeline/preprocess_20260408_174036}"
nohup ../sft/.venv/bin/python -u generate_grpo_candidates.py --run-dir "$RUN_DIR" > "$LOG_PATH" 2>&1 < /dev/null &
echo $!
