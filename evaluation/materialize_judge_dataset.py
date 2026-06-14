#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from judge_common import load_env
from judge_dataset_registry import DATASET_ORDER, DATASET_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize one evaluation dataset into a local JSONL snapshot.")
    parser.add_argument("--dataset-key", required=True, choices=DATASET_ORDER)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--cache-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    spec = DATASET_SPECS[args.dataset_key]
    load_kwargs: dict[str, Any] = {}
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["cache_dir"] = str(cache_dir)
    dataset_dict = load_dataset(spec["dataset_id"], **load_kwargs)
    split = spec["split"] if spec["split"] in dataset_dict else next(iter(dataset_dict.keys()))
    ds = dataset_dict[split]

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(ds):
            normalized = normalize_row(args.dataset_key, row, idx)
            if not normalized:
                continue
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "dataset_key": args.dataset_key,
        "dataset_id": spec["dataset_id"],
        "split": split,
        "rows": written,
        "poem_field": spec["poem_field"],
        "description_field": spec["description_field"],
        "output_jsonl": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"dataset_key={args.dataset_key} rows={written} output={output_path}")
    return 0


def normalize_row(dataset_key: str, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
    spec = DATASET_SPECS[dataset_key]
    poem = str(row.get(spec["poem_field"]) or "").strip()
    description = str(row.get(spec["description_field"]) or row.get("description") or "").strip()
    if not poem:
        return None
    source_id = first_existing_value(row, spec["id_field_candidates"]) or f"{dataset_key}_{idx:05d}"
    normalized = {key: to_jsonable(value) for key, value in row.items()}
    normalized["judge_dataset_key"] = dataset_key
    normalized["judge_dataset_id"] = spec["dataset_id"]
    normalized["judge_split"] = spec["split"]
    normalized["judge_row_index"] = idx
    normalized["judge_row_id"] = f"{dataset_key}::{source_id}"
    normalized["judge_source_id"] = str(source_id)
    normalized["judge_poem"] = poem
    normalized["judge_description"] = description
    return normalized


def first_existing_value(row: dict[str, Any], candidates: list[str]) -> str:
    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
