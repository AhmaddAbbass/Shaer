#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="$("./.venv/bin/python" - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("grpo_config.yaml").read_text(encoding="utf-8"))
print(cfg["studies"]["sanity_check_model"]["output_root"])
PY
)"

mkdir -p "$OUTPUT_ROOT"
RUN_TS=$(date +"%Y%m%d_%H%M%S")
LAUNCHER_LOG="$OUTPUT_ROOT/study02_${RUN_TS}.log"
INTERNAL_LOG="$OUTPUT_ROOT/sanity_check_model.log"
PID_FILE="$OUTPUT_ROOT/latest.pid"

echo "$LAUNCHER_LOG" > "$OUTPUT_ROOT/latest.log"
echo "$INTERNAL_LOG" > "$OUTPUT_ROOT/latest_internal.log"
rm -f "$PID_FILE"

setsid -f bash -lc "cd '$ROOT_DIR' && echo \$\$ > '$PID_FILE' && exec ./.venv/bin/python -u studies/08_sanity_check_model_base_only.py > '$LAUNCHER_LOG' 2>&1"
sleep 1
PID="$(cat "$PID_FILE")"

echo "sanity_check_model started"
echo "PID=$PID"
echo "LAUNCHER_LOG=$LAUNCHER_LOG"
echo "INTERNAL_LOG=$INTERNAL_LOG"
