#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from judge_common import load_env


DEFAULT_DATASET_ID = "Shaer-AI/shaer-sft-test"
DEFAULT_LOCAL_PARQUET = "evaluation/outputs/clean_shaer_sft_test/shaer_sft_test.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small reference-poem calibration bank for judge prompts.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-parquet", default=DEFAULT_LOCAL_PARQUET)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-jsonl", default="evaluation/outputs/judge_reference_calibration/reference_rows.jsonl")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    ds = load_rows(args)
    rows = [dict(row) for row in ds]
    selected = stratified_reference_sample(rows, total_rows=int(args.rows), seed=int(args.seed))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(selected, start=1):
            payload = {
                "calibration_id": f"reference_calibration_{idx:03d}",
                "source_dataset_id": args.dataset_id,
                "source_split": args.split,
                "base_meter": str(row.get("base_meter") or ""),
                "form": str(row.get("form") or ""),
                "requested_bayts": int(row.get("requested_bayts") or 0),
                "requested_num_lines": int(row.get("requested_num_lines") or 0),
                "description": str(row.get("description") or ""),
                "enhanced_description": str(row.get("enhanced_description") or ""),
                "reference_poem": str(row.get("reference_completion") or ""),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(selected),
        "dataset_id": args.dataset_id,
        "input_parquet": str(args.input_parquet or ""),
        "split": args.split,
        "seed": int(args.seed),
        "meters": meter_counts(selected),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written_rows={len(selected)} output={output_path}")
    return 0


def stratified_reference_sample(rows: list[dict[str, Any]], total_rows: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_meter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reference_poem = str(row.get("reference_completion") or "").strip()
        if not reference_poem:
            continue
        by_meter[str(row.get("base_meter") or "").strip()].append(row)

    meters = sorted([meter for meter, items in by_meter.items() if meter and items], key=lambda meter: (-len(by_meter[meter]), meter))
    if not meters:
        return []

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    while len(selected) < total_rows:
        progressed = False
        for meter in meters:
            candidates = [row for row in by_meter[meter] if row_key(row) not in used_ids]
            if not candidates:
                continue
            row = rng.choice(candidates)
            selected.append(row)
            used_ids.add(row_key(row))
            progressed = True
            if len(selected) >= total_rows:
                break
        if not progressed:
            break
    return selected


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("source_row_index") or row.get("reference_completion") or "")


def meter_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("base_meter") or "").strip()] += 1
    return dict(sorted(counts.items()))


def load_rows(args: argparse.Namespace):
    local_path = Path(str(args.input_parquet or "")).expanduser()
    if local_path.exists():
        return Dataset.from_parquet(str(local_path))
    return load_dataset(args.dataset_id, split=args.split)


if __name__ == "__main__":
    raise SystemExit(main())
