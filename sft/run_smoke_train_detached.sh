#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export SFT_MODEL_REPO_ID="${SFT_MODEL_REPO_ID:-Shaer-AI/Shaer-adapters}"
export SFT_DATASET_ID="${SFT_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
export SFT_RESUME_MODE="${SFT_RESUME_MODE:-fresh}"
export SFT_INIT_FROM_ADAPTER="${SFT_INIT_FROM_ADAPTER:-false}"
export SFT_CONTINUATION_NAMESPACE="${SFT_CONTINUATION_NAMESPACE:-fresh_sft_smoke}"
export SFT_MAX_STEPS="${SFT_MAX_STEPS:-120}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-100}"
export SFT_LIVE_EVAL_STEPS="${SFT_LIVE_EVAL_STEPS:-100}"
export SFT_FULL_EVAL_STEPS="${SFT_FULL_EVAL_STEPS:-100}"
export SFT_USE_WEIGHTED_SAMPLER="${SFT_USE_WEIGHTED_SAMPLER:-true}"
export WATCHER_STEP_EMAIL_EVERY="${WATCHER_STEP_EMAIL_EVERY:-50}"

bash run_train_detached.sh
