#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export SFT_MODEL_REPO_ID="${SFT_MODEL_REPO_ID:-Shaer-AI/shaer-adapters-v2}"
export SFT_DATASET_ID="${SFT_DATASET_ID:-Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits}"
export SFT_RESUME_MODE="${SFT_RESUME_MODE:-fresh}"
export SFT_INIT_FROM_ADAPTER="${SFT_INIT_FROM_ADAPTER:-false}"
export SFT_INIT_ADAPTER_PATH="${SFT_INIT_ADAPTER_PATH:-/root/workspace/Shaer/sft/outputs/train/train_20260407_231929/checkpoint-3000}"
export SFT_CONTINUATION_NAMESPACE="${SFT_CONTINUATION_NAMESPACE:-meter_aux_from_best_eval_v1}"

export SFT_ENABLE_METER_AUX="${SFT_ENABLE_METER_AUX:-true}"
export SFT_METER_AUX_LAMBDA="${SFT_METER_AUX_LAMBDA:-0.1}"
export SFT_LEARNING_RATE="${SFT_LEARNING_RATE:-3e-5}"
export SFT_MAX_STEPS="${SFT_MAX_STEPS:-1200}"
export SFT_SAVE_STEPS="${SFT_SAVE_STEPS:-200}"
export SFT_LIVE_EVAL_STEPS="${SFT_LIVE_EVAL_STEPS:-200}"

bash run_train_detached.sh
