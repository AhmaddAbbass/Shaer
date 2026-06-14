#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$ROOT_DIR/.." && pwd)"

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$REPO_DIR/.env"
  set +a
fi

RUN_DIR="${1:-}"
SCORE_WORKERS="${2:-${GRPO_MC_SCORE_WORKERS:-4}}"

if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 <run_dir> [score_workers]" >&2
  exit 1
fi

RUN_DIR="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$RUN_DIR")"
cd "$ROOT_DIR"
PYTHON="${GRPO_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing python at $PYTHON; set GRPO_PYTHON or create grpo/.venv" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/status" "$RUN_DIR/pids"

NUM_CANDIDATES="$("$PYTHON" - <<'PY' "$RUN_DIR"
import json, sys
from pathlib import Path
cfg = json.loads((Path(sys.argv[1]) / "run_config.json").read_text(encoding="utf-8"))
print(int(cfg.get("num_candidates") or 4))
PY
)"

stop_after="${GRPO_MC_STOP_AFTER_ROWS_PER_SHARD:-0}"
gen_batch_size="${GRPO_MC_GENERATOR_BATCH_SIZE:-16}"
max_new_tokens="${GRPO_MC_MAX_NEW_TOKENS:-640}"
score_poll="${GRPO_MC_SCORE_POLL_SECONDS:-5}"

start_generator() {
  local shard_id="$1"
  local gpu_id="$2"
  local gen_id
  gen_id=$(printf "gen%02d" "$shard_id")
  local log_path="$RUN_DIR/logs/${gen_id}.log"
  local stop_args=()
  if [[ "$stop_after" != "0" ]]; then
    stop_args+=(--stop-after-generated-rows "$stop_after")
  fi
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  GRPO_MC_BASE_MODEL_ID="${GRPO_MC_BASE_MODEL_ID:-Navid-AI/Yehia-7B-preview}" \
  GRPO_MC_SFT_ADAPTER_REPO="${GRPO_MC_SFT_ADAPTER_REPO:-Shaer-AI/Shaer-adapters}" \
  GRPO_MC_SFT_ADAPTER_MODE="${GRPO_MC_SFT_ADAPTER_MODE:-fresh_sft/train}" \
  setsid "$PYTHON" -u generate_meter_count_candidates.py \
    --run-dir "$RUN_DIR" \
    --shard-id "$shard_id" \
    --generator-id "$gen_id" \
    --num-candidates "$NUM_CANDIDATES" \
    --batch-size "$gen_batch_size" \
    --max-new-tokens "$max_new_tokens" \
    "${stop_args[@]}" \
    > "$log_path" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "$RUN_DIR/pids/${gen_id}.pid"
  echo "started ${gen_id} pid=${pid} gpu=${gpu_id} log=${log_path}"
}

start_score_worker() {
  local idx="$1"
  local worker_id
  worker_id=$(printf "score%02d" "$idx")
  local log_path="$RUN_DIR/logs/${worker_id}.log"
  CUDA_VISIBLE_DEVICES="" \
  METER_DEVICE="${METER_DEVICE:-cpu}" \
  setsid "$PYTHON" -u score_meter_count_candidates.py \
    --run-dir "$RUN_DIR" \
    --worker-id "$worker_id" \
    --poll-seconds "$score_poll" \
    --exit-when-generators-done \
    > "$log_path" 2>&1 < /dev/null &
  local pid="$!"
  echo "$pid" > "$RUN_DIR/pids/${worker_id}.pid"
  echo "started ${worker_id} pid=${pid} log=${log_path}"
}

start_generator 0 0
start_generator 1 1

for ((i=0; i<SCORE_WORKERS; i++)); do
  start_score_worker "$i"
done

{
  echo "RUN_DIR=$RUN_DIR"
  echo "NUM_CANDIDATES=$NUM_CANDIDATES"
  echo "GENERATOR_BATCH_SIZE=$gen_batch_size"
  echo "SCORE_WORKERS=$SCORE_WORKERS"
  echo "STOP_AFTER_ROWS_PER_SHARD=$stop_after"
} > "$RUN_DIR/pids/launcher.env"

echo "meter/count preprocess detached launch complete"
echo "RUN_DIR=$RUN_DIR"
