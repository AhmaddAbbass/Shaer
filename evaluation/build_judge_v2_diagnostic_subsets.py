#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from materialize_judge_dataset import main as materialize_main  # type: ignore
from judge_common import read_jsonl
from judge_dataset_registry import DATASET_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed-size diagnostic subsets for strict v2 judge reruns.")
    parser.add_argument("--rows-per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="evaluation/outputs/judge_v2_strict_diagnostic/subsets")
    parser.add_argument("--cache-root", default="C:/sjc_v2_diagnostic")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    for index, dataset_key in enumerate(DATASET_ORDER):
        dataset_dir = output_root / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        full_input_jsonl = dataset_dir / "full_input_rows.jsonl"
        full_summary_json = dataset_dir / "full_input_rows.summary.json"
        subset_input_jsonl = dataset_dir / "input_rows.jsonl"
        subset_summary_json = dataset_dir / "input_rows.summary.json"

        materialize_dataset(
            dataset_key=dataset_key,
            output_jsonl=full_input_jsonl,
            summary_json=full_summary_json,
            cache_dir=cache_root / dataset_key,
        )
        rows = read_jsonl(full_input_jsonl)
        subset = sample_rows(rows, rows_per_dataset=int(args.rows_per_dataset), seed=int(args.seed) + index)

        with subset_input_jsonl.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary = {
            "dataset_key": dataset_key,
            "rows": len(subset),
            "rows_per_dataset": int(args.rows_per_dataset),
            "seed": int(args.seed) + index,
            "source_input_jsonl": str(full_input_jsonl),
            "output_jsonl": str(subset_input_jsonl),
        }
        subset_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"dataset={dataset_key} subset_rows={len(subset)} output={subset_input_jsonl}")
    return 0


def sample_rows(rows: list[dict], *, rows_per_dataset: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(rows) <= rows_per_dataset:
        return rows
    sampled = rng.sample(rows, rows_per_dataset)
    sampled.sort(key=lambda row: int(row.get("judge_row_index") or 0))
    return sampled


def materialize_dataset(*, dataset_key: str, output_jsonl: Path, summary_json: Path, cache_dir: Path) -> None:
    import sys

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "materialize_judge_dataset.py",
            "--dataset-key",
            dataset_key,
            "--output-jsonl",
            str(output_jsonl),
            "--summary-json",
            str(summary_json),
            "--cache-dir",
            str(cache_dir),
        ]
        raise SystemExit(materialize_main())
    except SystemExit as exc:
        if int(exc.code or 0) != 0:
            raise
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
