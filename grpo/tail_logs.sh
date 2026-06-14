#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-train}"

get_study02_root() {
  ./.venv/bin/python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("grpo_config.yaml").read_text(encoding="utf-8"))
print(cfg["studies"]["sanity_check_model"]["output_root"])
PY
}

if [[ "$TARGET" == "train" ]]; then
  LOG=$(cat outputs/train/latest.log)
elif [[ "$TARGET" == "sanity" ]]; then
  LOG=$(cat outputs/sanity_check/latest.log)
elif [[ "$TARGET" == "study02" ]]; then
  LOG=$(cat "$(get_study02_root)/latest.log")
elif [[ "$TARGET" == "study02-internal" ]]; then
  LOG=$(cat "$(get_study02_root)/latest_internal.log")
elif [[ "$TARGET" == "meaning-sweep" ]]; then
  LOG=$(cat outputs/meaning_prompt_sweep/latest.log)
elif [[ "$TARGET" == "meaning-sweep-internal" ]]; then
  LOG=$(cat outputs/meaning_prompt_sweep/latest_internal.log)
else
  LOG="$TARGET"
fi

echo "Tailing: $LOG"
tail -f "$LOG"
