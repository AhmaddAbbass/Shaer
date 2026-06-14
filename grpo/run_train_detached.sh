#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-train}"
if [[ "$MODE" != "train" && "$MODE" != "sanity" ]]; then
  echo "usage: $0 [train|sanity]" >&2
  exit 1
fi

PYTHON="./.venv/bin/python"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

CURATED_ROOT="${GRPO_CURATED_OUTPUT_DIR:-$ROOT_DIR/outputs/curated_meter_count_easyfirst_short_drop_trio}"
CURATED_MANIFEST="$CURATED_ROOT/cap_1000/selected_manifest.csv"
HARD_MANIFEST="$CURATED_ROOT/hard_diagnostic_cap_256/selected_manifest.csv"

if [[ ! -f "$CURATED_MANIFEST" || ! -f "$HARD_MANIFEST" ]]; then
  "$PYTHON" build_easy_first_subset.py \
    --dataset-id Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1 \
    --output-root "$CURATED_ROOT" \
    --cap-per-meter 1000 \
    --allowed-length-buckets 1-3 4-6
fi

RUN_TS="$(date +"%Y%m%d_%H%M%S")"
if [[ "$MODE" == "sanity" ]]; then
  RUN_ROOT="$ROOT_DIR/outputs/sanity_check"
  RUN_PREFIX="sanity"
  export WATCHER_STEP_EMAIL_EVERY="${WATCHER_STEP_EMAIL_EVERY:-1}"
else
  RUN_ROOT="$ROOT_DIR/outputs/train"
  RUN_PREFIX="shaer_grpo"
  export WATCHER_STEP_EMAIL_EVERY="${WATCHER_STEP_EMAIL_EVERY:-50}"
fi

mkdir -p "$RUN_ROOT"
export GRPO_RUN_DIR="$RUN_ROOT/${RUN_PREFIX}_${RUN_TS}"
mkdir -p "$GRPO_RUN_DIR"

TRAIN_STDOUT_LOG="$GRPO_RUN_DIR/train_stdout.log"
WATCHER_LOG="$GRPO_RUN_DIR/watcher.log"
PLOTTER_LOG="$GRPO_RUN_DIR/plotter.log"

export BASE_MODEL_ID="${BASE_MODEL_ID:-Navid-AI/Yehia-7B-preview}"
export SFT_ADAPTER_REPO="${SFT_ADAPTER_REPO:-Shaer-AI/Shaer-adapters}"
export SFT_ADAPTER_MODE="${SFT_ADAPTER_MODE:-fresh_sft/train}"
export GRPO_OUTPUT_REPO="${GRPO_OUTPUT_REPO:-Shaer-AI/Shaer-adapters-grpo-short1k-no-trio-v5}"
export JUDGE_MODEL="${JUDGE_MODEL:-qwen/qwen3.5-35b-a3b}"
export GRPO_RESUME_MODE="${GRPO_RESUME_MODE:-fresh}"
export GRPO_RESUME_PATH="${GRPO_RESUME_PATH:-}"
export GRPO_TRAIN_DATASET_ID="Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1"
export GRPO_SOURCE_DATASET_ID="Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
export GRPO_CURATED_MANIFEST_PATH="$CURATED_MANIFEST"
export GRPO_HARD_DIAGNOSTIC_MANIFEST_PATH="$HARD_MANIFEST"

setsid "$PYTHON" -u train_grpo.py --mode "$MODE" > "$TRAIN_STDOUT_LOG" 2>&1 < /dev/null &
TRAIN_PID=$!

setsid "$PYTHON" -u watcher.py "$GRPO_RUN_DIR" > "$WATCHER_LOG" 2>&1 < /dev/null &
WATCHER_PID=$!

setsid "$PYTHON" -u plot_live_rewards.py --run-dir "$GRPO_RUN_DIR" --watch --follow-chain --train-only > "$PLOTTER_LOG" 2>&1 < /dev/null &
PLOTTER_PID=$!

echo "mode=$MODE"
echo "run_dir=$GRPO_RUN_DIR"
echo "train_pid=$TRAIN_PID"
echo "watcher_pid=$WATCHER_PID"
echo "plotter_pid=$PLOTTER_PID"
echo "train_stdout_log=$TRAIN_STDOUT_LOG"
echo "watcher_log=$WATCHER_LOG"
echo "plotter_log=$PLOTTER_LOG"

printf '%s\n' "$TRAIN_PID" > "$GRPO_RUN_DIR/train.pid"
printf '%s\n' "$WATCHER_PID" > "$GRPO_RUN_DIR/watcher.pid"
printf '%s\n' "$PLOTTER_PID" > "$GRPO_RUN_DIR/plotter.pid"

printf '%s\n' "$GRPO_RUN_DIR" > "$RUN_ROOT/latest.run_dir"
printf '%s\n' "$TRAIN_STDOUT_LOG" > "$RUN_ROOT/latest.train_stdout.log"
printf '%s\n' "$WATCHER_LOG" > "$RUN_ROOT/latest.watcher.log"
printf '%s\n' "$PLOTTER_LOG" > "$RUN_ROOT/latest.plotter.log"
