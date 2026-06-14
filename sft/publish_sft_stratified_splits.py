#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

sys.path.insert(0, str(PROJECT_ROOT))

from split_utils import MIN_SAFE_STRATIFY_GROUP_SIZE, split_dataset  # noqa: E402
from train_sft_qlora import load_dataset_resilient  # noqa: E402


DEFAULT_SOURCE_DATASET = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500"
DEFAULT_TARGET_DATASET = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_readme(source_dataset: str, target_dataset: str, summary: dict[str, Any]) -> str:
    split_counts = summary["split_counts"]
    levels = summary["stratify_group_level_counts"]
    return f"""---
language:
- ar
license: apache-2.0
pretty_name: Ashaar Enhanced Description SFT Stratified Splits
task_categories:
- text-generation
size_categories:
- 100K<n<1M
---

# Ashaar Enhanced Description SFT Stratified Splits

Source dataset:
- `{source_dataset}`

Target dataset:
- `{target_dataset}`

This dataset publishes deterministic `train / eval / test` splits with a `94 / 3 / 3` policy.

## Split policy

Primary stratification key:
- `base_meter`
- `form`
- `length_bucket`

Length buckets:
- `1-3`
- `4-6`
- `7-10`
- `11-20`

Small groups fall back gracefully to coarser stratification levels when needed.

## Counts

- train: **{split_counts['train']}**
- eval: **{split_counts['eval']}**
- test: **{split_counts['test']}**

## Stratification fallback levels used

- `base_meter_form_length_bucket`: **{levels.get('base_meter_form_length_bucket', 0)}**
- `base_meter_form`: **{levels.get('base_meter_form', 0)}**
- `base_meter`: **{levels.get('base_meter', 0)}**
- `global`: **{levels.get('global', 0)}**

## Notes

- `sampler_group` keeps the fine-grained joint group `base_meter||form||length_bucket`
- `split_group` is the actual group used to allocate split quotas after fallback
- weighted sampling should be applied only on the `train` split
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish deterministic stratified SFT splits to Hugging Face")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET_DATASET)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-group-size", type=int, default=MIN_SAFE_STRATIFY_GROUP_SIZE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_dotenv(ENV_PATH, override=False)
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")

    run_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(
        Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "sft" / "outputs" / "split_publish" / f"split_publish_{run_slug}")
    )

    class SimpleLogger:
        def info(self, message: str, *args: Any) -> None:
            print(message % args if args else message, flush=True)

        def warning(self, message: str, *args: Any) -> None:
            print(message % args if args else message, flush=True)

    logger = SimpleLogger()
    logger.info("Loading source dataset: %s", args.source_dataset)
    ds = load_dataset_resilient(args.source_dataset, hf_token, logger=logger, cache_dir=output_dir / ".cache")
    logger.info("Loaded %d rows; building deterministic splits", len(ds))

    dataset_dict, summary = split_dataset(ds, seed=args.seed, min_group_size=args.min_group_size)
    summary = {
        "timestamp_utc": utc_now_iso(),
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        **summary,
    }

    summary_path = output_dir / "split_summary.json"
    dump_json(summary_path, summary)

    readme_text = build_readme(args.source_dataset, args.target_dataset, summary)
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")

    if not args.no_push:
        api = HfApi(token=hf_token)
        create_repo(repo_id=args.target_dataset, repo_type="dataset", token=hf_token, exist_ok=True)
        dataset_dict.push_to_hub(args.target_dataset, token=hf_token)
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo="split_summary.json",
            repo_id=args.target_dataset,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=args.target_dataset,
            repo_type="dataset",
        )
        logger.info("Pushed split dataset to %s", args.target_dataset)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
