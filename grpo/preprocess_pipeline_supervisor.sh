#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-}"
JUDGE_WORKERS="${2:-8}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GENERATOR_STALE_MINUTES="${GENERATOR_STALE_MINUTES:-20}"

if [[ -z "$RUN_DIR" ]]; then
  echo "usage: $0 <run_dir> [judge_workers]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
LOG_PATH="$RUN_DIR/preprocess_supervisor.log"

_ts_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
_log() { printf "%s | %s\n" "$(_ts_utc)" "$*" >> "$LOG_PATH"; }
_file_stale_minutes() {
  local path="$1"
  python - "$path" <<'PYSTALE'
import os, sys, time
path = sys.argv[1]
if not os.path.exists(path):
    print(1000000)
else:
    print(max(0.0, (time.time() - os.path.getmtime(path)) / 60.0))
PYSTALE
}
_start_generator() {
  SFT_ADAPTER_REPO=Shaer-AI/Shaer-adapters SFT_ADAPTER_MODE=fresh_sft/train     setsid ../sft/.venv/bin/python -u generate_grpo_candidates.py --run-dir "$RUN_DIR"     >> "$RUN_DIR/generator_detached.log" 2>&1 < /dev/null &
  echo $! > "$RUN_DIR/generator_manual.pid"
  _log "started generator pid=$(cat "$RUN_DIR/generator_manual.pid")"
}
_start_judge() {
  local worker_id="$1"
  local publish_flag="$2"
  local extra=()
  if [[ "$publish_flag" == "1" ]]; then
    extra+=(--publish)
  fi
  setsid ../sft/.venv/bin/python -u judge_grpo_candidates.py --run-dir "$RUN_DIR" --worker-id "$worker_id" "${extra[@]}"     >> "$RUN_DIR/${worker_id}_detached.log" 2>&1 < /dev/null &
  _log "started $worker_id pid=$! publish=$publish_flag"
}

while true; do
  gen_count=$(pgrep -fc "generate_grpo_candidates.py --run-dir $RUN_DIR" || true)
  engine_count=$(pgrep -fc 'VLLM::EngineCore' || true)
  gen_stale=$(_file_stale_minutes "$RUN_DIR/train_generations.jsonl")
  gen_stale_int=${gen_stale%.*}

  if [[ "$gen_count" -eq 0 || ( "$engine_count" -eq 0 && "$gen_stale_int" -ge "$GENERATOR_STALE_MINUTES" ) ]]; then
    pkill -f "generate_grpo_candidates.py --run-dir $RUN_DIR" || true
    pkill -f 'VLLM::EngineCore' || true
    sleep 2
    _start_generator
  fi

  for ((i=0; i<JUDGE_WORKERS; i++)); do
    worker_id=$(printf 'judge%02d' "$i")
    if ! pgrep -f "judge_grpo_candidates.py --run-dir $RUN_DIR --worker-id $worker_id" >/dev/null; then
      publish=0
      if [[ "$i" -eq 0 ]]; then
        publish=1
      fi
      _start_judge "$worker_id" "$publish"
    fi
  done

  sleep "$POLL_SECONDS"
done
