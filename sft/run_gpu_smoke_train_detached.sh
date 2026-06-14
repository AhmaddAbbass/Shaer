#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export SFT_MODEL_REPO_ID="${SFT_MODEL_REPO_ID:-Shaer-AI/Shaer-adapters}"
export SFT_DATASET_ID="${SFT_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
export SFT_RESUME_MODE="${SFT_RESUME_MODE:-fresh}"
export SFT_INIT_FROM_ADAPTER="${SFT_INIT_FROM_ADAPTER:-false}"
export SFT_CONTINUATION_NAMESPACE="${SFT_CONTINUATION_NAMESPACE:-fresh_sft_gpu_smoke}"

# Keep the real model/QLoRA path, but cap rows and steps so the smoke completes quickly.
export SFT_LIMIT_TRAIN_ROWS="${SFT_LIMIT_TRAIN_ROWS:-256}"
export SFT_LIMIT_EVAL_ROWS="${SFT_LIMIT_EVAL_ROWS:-64}"
export SFT_LIMIT_TEST_ROWS="${SFT_LIMIT_TEST_ROWS:-64}"
export SFT_MAX_STEPS="${SFT_MAX_STEPS:-12}"
export SFT_LOGGING_STEPS="${SFT_LOGGING_STEPS:-1}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-10}"
export SFT_LIVE_EVAL_STEPS="${SFT_LIVE_EVAL_STEPS:-10}"
export SFT_FULL_EVAL_STEPS="${SFT_FULL_EVAL_STEPS:-10}"
export SFT_USE_WEIGHTED_SAMPLER="${SFT_USE_WEIGHTED_SAMPLER:-true}"
export SFT_PROBE_BATCH_SIZE="${SFT_PROBE_BATCH_SIZE:-2}"
export SFT_PROBE_MAX_NEW_TOKENS="${SFT_PROBE_MAX_NEW_TOKENS:-96}"
export WATCHER_STEP_EMAIL_EVERY="${WATCHER_STEP_EMAIL_EVERY:-5}"

bash run_train_detached.sh
