#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="$ROOT_DIR/outputs"
mkdir -p "$OUTPUT_ROOT"

TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="descgen_${TS}"
OUTER_LOG="$OUTPUT_ROOT/${RUN_NAME}.log"
PYTHON="/root/workspace/Shaer/sft/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing virtualenv python at $PYTHON" >&2
  exit 1
fi

HAS_RUN_NAME=0
for arg in "$@"; do
  if [[ "$arg" == "--run-name" ]]; then
    HAS_RUN_NAME=1
    break
  fi
done

CMD=("$PYTHON" -u "$ROOT_DIR/regenerate_descriptions.py")
if [[ $HAS_RUN_NAME -eq 0 ]]; then
  CMD+=(--run-name "$RUN_NAME")
else
  RUN_NAME="custom_$(date -u +%Y%m%d_%H%M%S)"
  OUTER_LOG="$OUTPUT_ROOT/${RUN_NAME}.log"
fi
CMD+=("$@")

setsid "${CMD[@]}" >"$OUTER_LOG" 2>&1 < /dev/null &
PID=$!

echo "$PID" > "$OUTPUT_ROOT/latest.pid"
echo "$OUTER_LOG" > "$OUTPUT_ROOT/latest.log"

echo "description_generation started"
echo "PID=$PID"
echo "LOG=$OUTER_LOG"
