#!/usr/bin/env bash
set -euo pipefail
cd /root/workspace/Shaer/grpo
RUN_DIR="outputs/preprocess_grpo_pipeline/smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "$RUN_DIR"
../sft/.venv/bin/python -u judge_grpo_candidates.py --run-dir "$RUN_DIR" > "$RUN_DIR/judge_smoke.log" 2>&1 &
JPID=$!
echo "JPID=$JPID"
SFT_ADAPTER_REPO=Shaer-AI/Shaer-adapters SFT_ADAPTER_MODE=fresh_sft/train ../sft/.venv/bin/python -u generate_grpo_candidates.py --run-dir "$RUN_DIR" --max-rows-per-split 1 > "$RUN_DIR/generator_smoke.log" 2>&1
wait "$JPID"
echo SMOKE_DONE
ls -la "$RUN_DIR"
