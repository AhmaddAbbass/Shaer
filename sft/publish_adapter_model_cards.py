#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_latest_jsonl_row(path: Path, mode: str | None = None) -> dict | None:
    if not path.exists():
        return None
    latest = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if mode is None or row.get("mode") == mode:
                latest = row
    return latest


def collect_v2_status(v2_run_dir: Path) -> dict:
    metrics_path = v2_run_dir / "metrics.jsonl"
    probe_path = v2_run_dir / "probe_metrics.jsonl"
    latest_eval = read_latest_jsonl_row(metrics_path, "eval") or {}
    latest_eval_aux = read_latest_jsonl_row(metrics_path, "eval_aux") or {}
    latest_probe = read_latest_jsonl_row(probe_path) or {}
    return {
        "run_name": v2_run_dir.name,
        "run_dir": str(v2_run_dir),
        "latest_eval": latest_eval,
        "latest_eval_aux": latest_eval_aux,
        "latest_probe": latest_probe,
        "stopped_summary": "/root/workspace/Shaer/artifacts/sft_continuation/train_20260408_133801_stopped_summary.md",
    }


def build_baseline_card(main_summary: dict) -> str:
    return f"""---
language:
- ar
license: apache-2.0
base_model: Navid-AI/Yehia-7B-preview
tags:
- peft
- qlora
- arabic
- poetry
- classical-arabic-poetry
- meter-conditioned-generation
pipeline_tag: text-generation
---

# Shaer Main SFT Adapters

This repository stores the completed main SFT baseline for the Shaer classical Arabic poetry project.

## Baseline Summary

- Base model: `Navid-AI/Yehia-7B-preview`
- Dataset: `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`
- Split policy: deterministic `94 / 3 / 3`, stratified by `base_meter||form||length_bucket`
- Train sampler: weighted train-only sampler on `base_meter||form||length_bucket`
- LoRA: `all-linear`, `r=64`, `alpha=128`, `dropout=0.05`, `use_rslora=true`
- Run name: `{main_summary['run_name']}`
- Best eval checkpoint: `/root/workspace/Shaer/sft/outputs/train/{main_summary['run_name']}/checkpoint-3000`
- Best eval loss: `{main_summary['final_eval']['eval_loss']}`
- Final test loss: `{main_summary['final_test_eval']['test_loss']}`
- Best probe meter mean: `0.6042156156147234` at step `2800`
- Final probe meter mean: `{main_summary['final_probe']['probe_meter_mean']}`
- Final probe count adherence mean: `{main_summary['final_probe']['probe_count_adherence_mean']}`

## Important Paths In This Repo

- Latest adapter export:
  - `adapters/fresh_sft/train/latest`
- Best adapter export:
  - `adapters/fresh_sft/train/best`
- Finished run report bundle:
  - `reports/fresh_sft/train_20260407_231929`

## Current Comparison Context

- The short meter-loss continuation in `Shaer-AI/shaer-adapters-v2` was stopped early after the auxiliary meter head improved but CE and probe meter drifted worse than this baseline.
- A later fresh-from-start `v3` run was deleted after confirming the current meter-loss path was invalid because it could read the requested meter from prompt-conditioned hidden states.
- This baseline remains the strongest known safe reference until a corrected challenger clearly beats it.
"""


def build_v2_card(main_summary: dict, v2_status: dict) -> str:
    eval_aux = v2_status["latest_eval_aux"]
    probe = v2_status["latest_probe"]
    return f"""---
language:
- ar
license: apache-2.0
base_model: Navid-AI/Yehia-7B-preview
tags:
- peft
- qlora
- arabic
- poetry
- classical-arabic-poetry
- auxiliary-meter-loss
- stopped-run
pipeline_tag: text-generation
---

# Shaer Meter-Loss Continuation V2

This repository stores the first continuation-from-best-eval meter-loss experiment.

## Experiment Identity

- Source baseline run: `{main_summary['run_name']}`
- Source checkpoint: `/root/workspace/Shaer/sft/outputs/train/{main_summary['run_name']}/checkpoint-3000`
- Target repo: `Shaer-AI/shaer-adapters-v2`
- Namespace: `meter_aux_from_best_eval_v1`
- Run dir: `{v2_status['run_dir']}`

## What Was Tried

- Start from the finished baseline best checkpoint by eval loss
- Keep the same dataset, splits, weighted sampler, tokenizer, prompt format, and all-linear QLoRA setup
- Add a train-time differentiable auxiliary meter loss on top of CE

## Why This Run Was Stopped

- The auxiliary meter head improved quickly on held-out eval
- But held-out CE became worse than the starting baseline checkpoint
- And generation-side probe meter also became worse than the starting baseline checkpoint

## Latest Useful Stopped Snapshot

- Step: `{eval_aux.get('global_step')}`
- Eval CE loss: `{eval_aux.get('eval_ce_loss')}`
- Eval meter loss: `{eval_aux.get('eval_meter_loss')}`
- Eval total loss: `{eval_aux.get('eval_total_loss')}`
- Eval meter accuracy: `{eval_aux.get('eval_meter_accuracy')}`
- Probe meter mean: `{probe.get('probe_meter_mean')}`
- Probe count adherence mean: `{probe.get('probe_count_adherence_mean')}`

## Interpretation

This run is preserved as a useful cautionary / negative result. The evidence points more toward objective mismatch or harmful drift than toward a convincing model improvement. A later audit also showed that the current auxiliary meter-loss design is flawed because it can recover the requested meter from prompt-conditioned hidden states.

## Useful Paths

- Stopped-run summary:
  - `reports/meter_aux_from_best_eval_v1/train_20260408_133801_stopped_summary.md`
- Checkpoints:
  - `checkpoints/meter_aux_from_best_eval_v1/train/train_20260408_133801/`
"""


def upload_text(api: HfApi, repo_id: str, text: str) -> None:
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )


def upload_optional_report(api: HfApi, repo_id: str, local_path: Path, repo_path: str) -> None:
    if not local_path.exists():
        return
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="model",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run-summary", required=True)
    parser.add_argument("--v2-run-dir", required=True)
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=False)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")

    api = HfApi(token=hf_token)
    main_summary = read_json(Path(args.main_run_summary))
    v2_status = collect_v2_status(Path(args.v2_run_dir))

    upload_text(api, "Shaer-AI/Shaer-adapters", build_baseline_card(main_summary))
    upload_text(api, "Shaer-AI/shaer-adapters-v2", build_v2_card(main_summary, v2_status))

    upload_optional_report(
        api,
        "Shaer-AI/shaer-adapters-v2",
        Path(v2_status["stopped_summary"]),
        "reports/meter_aux_from_best_eval_v1/train_20260408_133801_stopped_summary.md",
    )

    print("MODEL_CARDS_OK")


if __name__ == "__main__":
    main()
