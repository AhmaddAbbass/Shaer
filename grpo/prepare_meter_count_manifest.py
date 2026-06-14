#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from preprocess_meter_count_common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_DATASET_ID,
    DEFAULT_OUTPUT_DATASET_ID,
    DEFAULT_RUN_ROOT,
    DEFAULT_SFT_ADAPTER_MODE,
    DEFAULT_SFT_ADAPTER_REPO,
    atomic_write_json,
    balanced_subset_indices,
    ensure_dir,
    load_dotenv_if_present,
    manifest_path,
    manifest_row_from_source,
    meter_round_robin_indices,
    run_config_path,
    save_json,
    shard_manifest_path,
    write_jsonl,
)


def load_dataset_split(dataset_id: str, split: str, cache_dir: str | None):
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("The datasets package is required. Install the GRPO environment first.") from exc
    token = os.getenv("HF_TOKEN", "").strip() or None
    return load_dataset(dataset_id, split=split, token=token, cache_dir=cache_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a train-only meter/count preprocess manifest.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--source-split", "--split", default="train")
    parser.add_argument("--output-dataset-id", default=DEFAULT_OUTPUT_DATASET_ID)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--num-generator-shards", "--num-shards", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, choices=[1, 2, 4, 6, 8], default=4)
    parser.add_argument("--balanced-rows-per-base-meter", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    if args.num_generator_shards < 1:
        raise ValueError("--num-generator-shards must be >= 1")
    if args.source_split != "train":
        raise ValueError("This preprocess path is intentionally train-split-only for now.")

    if args.run_dir:
        run_dir = ensure_dir(args.run_dir)
    else:
        slug = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = ensure_dir(Path(DEFAULT_RUN_ROOT) / f"meter_count_{slug}_k{args.num_candidates}")

    ds = load_dataset_split(args.dataset_id, args.source_split, args.cache_dir or None)
    source_rows = [dict(row) for row in ds]
    selected = balanced_subset_indices(source_rows, args.balanced_rows_per_base_meter)
    if args.max_rows > 0:
        selected = selected[: args.max_rows]
    selected_source_rows = [source_rows[idx] for idx in selected]

    ordered_local_indices = meter_round_robin_indices(selected_source_rows)
    manifest_rows = []
    for manifest_order, local_idx in enumerate(ordered_local_indices):
        train_row_index = selected[local_idx]
        generator_shard_id = manifest_order % args.num_generator_shards
        manifest_row = manifest_row_from_source(
            selected_source_rows[local_idx],
            train_row_index=train_row_index,
            source_split=args.source_split,
            manifest_order=manifest_order,
            generator_shard_id=generator_shard_id,
        )
        manifest_row["source_dataset_id"] = args.dataset_id
        manifest_rows.append(manifest_row)

    base_model_id = os.getenv("GRPO_MC_BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID).strip() or DEFAULT_BASE_MODEL_ID
    adapter_repo = os.getenv("GRPO_MC_SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO).strip() or DEFAULT_SFT_ADAPTER_REPO
    adapter_mode = os.getenv("GRPO_MC_SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE).strip() or DEFAULT_SFT_ADAPTER_MODE

    run_config = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_id": args.dataset_id,
        "source_split": args.source_split,
        "output_dataset_id": args.output_dataset_id,
        "base_model_id": base_model_id,
        "sft_adapter_repo": adapter_repo,
        "sft_adapter_mode": adapter_mode,
        "num_generator_shards": int(args.num_generator_shards),
        "num_candidates": int(args.num_candidates),
        "seed": int(args.seed),
        "source_rows_available": len(source_rows),
        "manifest_rows": len(manifest_rows),
        "balanced_rows_per_base_meter": int(args.balanced_rows_per_base_meter),
        "max_rows": int(args.max_rows),
        "preprocess_design": "meter_count_train_only_v1",
    }
    ensure_dir(run_dir / "manifest")
    atomic_write_json(run_config_path(run_dir), run_config)
    write_jsonl(manifest_path(run_dir), manifest_rows)

    for shard_id in range(args.num_generator_shards):
        shard_rows = [row for row in manifest_rows if int(row["generator_shard_id"]) == shard_id]
        write_jsonl(shard_manifest_path(run_dir, shard_id), shard_rows)

    meter_counts = Counter(str(row["base_meter"]) for row in manifest_rows)
    shard_counts = Counter(int(row["generator_shard_id"]) for row in manifest_rows)
    summary = {
        **run_config,
        "run_dir": str(run_dir),
        "base_meter_counts": dict(sorted(meter_counts.items())),
        "shard_counts": {str(k): int(v) for k, v in sorted(shard_counts.items())},
        "manifest_path": str(manifest_path(run_dir)),
    }
    save_json(run_dir / "manifest" / "manifest_summary.json", summary)
    print(f"RUN_DIR={run_dir}")
    print(f"MANIFEST_ROWS={len(manifest_rows)}")
    print(f"NUM_CANDIDATES={args.num_candidates}")
    print(f"SHARD_COUNTS={dict(sorted(shard_counts.items()))}")
    print(f"BASE_METER_COUNTS={dict(sorted(meter_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
