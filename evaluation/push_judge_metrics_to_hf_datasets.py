#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values
from huggingface_hub import HfApi


METRIC_COLUMNS = [
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


DATASET_SPECS = {
    "shaer": {
        "dataset_id": "Shaer-AI/shaer-sft-test",
        "repo_data_path": "data/test-00000-of-00001.parquet",
        "format": "parquet",
        "id_field_candidates": ["id"],
        "judge_metrics": METRIC_COLUMNS,
    },
    "ashaar": {
        "dataset_id": "Shaer-AI/shaer-eval-ashaar-native-controls",
        "repo_data_path": "data/test.jsonl",
        "format": "jsonl",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": ["meaning", "fluency", "coherence", "poeticness"],
    },
    "yehia": {
        "dataset_id": "Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template",
        "repo_data_path": "data/test.jsonl",
        "format": "jsonl",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": METRIC_COLUMNS,
    },
    "fanar": {
        "dataset_id": "Shaer-AI/fanar-eval-native-prompt",
        "repo_data_path": "data/train-00000-of-00001.parquet",
        "format": "parquet",
        "id_field_candidates": ["generation_id", "id", "input_row_id"],
        "judge_metrics": ["meaning", "fluency", "coherence", "poeticness"],
    },
}

DATASET_ORDER = ["shaer", "ashaar", "yehia", "fanar"]


def load_hf_token(project_root: Path) -> str:
    env = dotenv_values(project_root / ".env")
    token = (env.get("HF_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing from .env")
    return token


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def detect_id_field(rows: list[dict[str, Any]], candidates: list[str]) -> str:
    if not rows:
        raise RuntimeError("cannot detect id field from empty rows")
    sample = rows[0]
    for field in candidates:
        if field in sample:
            return field
    raise RuntimeError(f"none of the id candidates exist: {candidates}")


def strip_judge_helper_fields(row: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("judge_"):
            continue
        if key in METRIC_COLUMNS:
            continue
        clean[key] = value
    return clean


def build_updated_rows(dataset_key: str, run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = DATASET_SPECS[dataset_key]
    dataset_dir = run_root / dataset_key
    input_rows_path = dataset_dir / "input_rows.jsonl"
    scored_rows_path = dataset_dir / "scored_rows.jsonl"

    input_rows = read_jsonl(input_rows_path)
    scored_rows = read_jsonl(scored_rows_path)

    source_id_field = detect_id_field(input_rows, spec["id_field_candidates"])
    scored_id_field = detect_id_field(scored_rows, spec["id_field_candidates"])

    source_rows = [strip_judge_helper_fields(row) for row in input_rows]

    score_map: dict[Any, dict[str, Any]] = {}
    for row in scored_rows:
        row_id = row.get(scored_id_field)
        if row_id in score_map:
            raise RuntimeError(f"duplicate scored id in {dataset_key}: {row_id}")
        score_map[row_id] = {metric: row.get(metric) for metric in METRIC_COLUMNS}

    updated_rows: list[dict[str, Any]] = []
    applicable_metrics = spec["judge_metrics"]
    missing_scores = 0
    for row in source_rows:
        row_id = row.get(source_id_field)
        scores = score_map.get(row_id)
        if scores is None:
            raise RuntimeError(f"missing scored row for {dataset_key} id={row_id}")
        merged = dict(row)
        for metric in METRIC_COLUMNS:
            merged[metric] = scores.get(metric)
        if any(merged.get(metric) is None for metric in applicable_metrics):
            missing_scores += 1
        updated_rows.append(merged)

    summary = {
        "dataset_key": dataset_key,
        "dataset_id": spec["dataset_id"],
        "rows": len(updated_rows),
        "source_id_field": source_id_field,
        "scored_id_field": scored_id_field,
        "judge_metrics": applicable_metrics,
        "missing_metric_rows": missing_scores,
        "repo_data_path": spec["repo_data_path"],
        "format": spec["format"],
    }
    return updated_rows, summary


def write_dataset_file(dataset_key: str, rows: list[dict[str, Any]], output_root: Path) -> Path:
    spec = DATASET_SPECS[dataset_key]
    out_path = output_root / dataset_key / Path(spec["repo_data_path"]).name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if spec["format"] == "jsonl":
        write_jsonl(out_path, rows)
    elif spec["format"] == "parquet":
        pd.DataFrame(rows).to_parquet(out_path, index=False)
    else:
        raise RuntimeError(f"unsupported format: {spec['format']}")
    return out_path


def upload_dataset_file(api: HfApi, dataset_key: str, local_file: Path) -> None:
    spec = DATASET_SPECS[dataset_key]
    api.upload_file(
        path_or_fileobj=str(local_file),
        path_in_repo=spec["repo_data_path"],
        repo_id=spec["dataset_id"],
        repo_type="dataset",
        commit_message="Add strict v2 judge metric columns to dataset split file",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append strict v2 judge metrics to the four evaluation datasets and optionally push only the split data files."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(r"C:/sjrun/judge_v2_strict_full_20260606_0031"),
        help="Finished strict v2 full run root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/outputs/push_judge_metrics_v2"),
        help="Local staging directory for updated split files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Which datasets to stage/push",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Upload only the split data files to the existing HF dataset repos",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    api = HfApi(token=load_hf_token(project_root)) if args.push else None

    all_summaries: list[dict[str, Any]] = []
    for dataset_key in args.datasets:
        rows, summary = build_updated_rows(dataset_key, args.run_root)
        local_file = write_dataset_file(dataset_key, rows, args.output_root)
        summary["local_file"] = str(local_file)
        if api is not None:
            upload_dataset_file(api, dataset_key, local_file)
            summary["pushed"] = True
        else:
            summary["pushed"] = False
        all_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
