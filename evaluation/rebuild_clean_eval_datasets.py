#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from dotenv import dotenv_values
from huggingface_hub import HfApi


DATASET_SPECS = {
    "shaer": {
        "repo_id": "Shaer-AI/shaer-sft-test",
        "source_split": "test",
        "target_filename": "test-00000-of-00001.parquet",
        "format": "parquet",
        "description_adherence_applicable": True,
        "count_adherence_applicable": True,
    },
    "yehia": {
        "repo_id": "Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template",
        "source_split": "test",
        "target_filename": "test.jsonl",
        "format": "jsonl",
        "description_adherence_applicable": True,
        "count_adherence_applicable": True,
    },
    "ashaar": {
        "repo_id": "Shaer-AI/shaer-eval-ashaar-native-controls",
        "source_split": "test",
        "target_filename": "test.jsonl",
        "format": "jsonl",
        "description_adherence_applicable": False,
        "count_adherence_applicable": False,
    },
    "fanar": {
        "repo_id": "Shaer-AI/fanar-eval-native-prompt",
        "source_split": "train",
        "target_filename": "train-00000-of-00001.parquet",
        "format": "parquet",
        "description_adherence_applicable": False,
        "count_adherence_applicable": False,
    },
}

DATASET_ORDER = ["shaer", "ashaar", "yehia", "fanar"]

CLEAN_COLUMNS = [
    "id",
    "base_meter",
    "form",
    "requested_bayts",
    "requested_num_lines",
    "description",
    "enhanced_description",
    "reference_completion",
    "generated_text",
    "meter",
    "count_adherence",
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


def find_hf_token(project_root: Path) -> str:
    token = (Path.cwd().resolve(),)
    env_paths = [project_root / ".env", project_root.parent / ".env", project_root.parent.parent / ".env"]
    for env_path in env_paths:
        if env_path.exists():
            env = dotenv_values(env_path)
            val = (env.get("HF_TOKEN") or "").strip()
            if val:
                return val
    raise RuntimeError("HF_TOKEN not found in nearby .env files")


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def normalize_row(dataset_key: str, row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if dataset_key == "shaer":
        out = {
            "id": row["id"],
            "base_meter": row.get("base_meter"),
            "form": row.get("form"),
            "requested_bayts": row.get("requested_bayts"),
            "requested_num_lines": row.get("requested_num_lines"),
            "description": row.get("description"),
            "enhanced_description": row.get("enhanced_description"),
            "reference_completion": row.get("reference_completion"),
            "generated_text": row.get("generated_text"),
            "meter": row.get("meter"),
            "count_adherence": row.get("count_adherence"),
            "description_adherence": row.get("description_adherence"),
            "meaning": row.get("meaning"),
            "fluency": row.get("fluency"),
            "coherence": row.get("coherence"),
            "poeticness": row.get("poeticness"),
        }
    elif dataset_key == "yehia":
        out = {
            "id": row["generation_id"],
            "base_meter": row.get("base_meter"),
            "form": row.get("form"),
            "requested_bayts": row.get("requested_bayts"),
            "requested_num_lines": row.get("requested_num_lines") or row.get("sft_num_lines"),
            "description": row.get("description"),
            "enhanced_description": row.get("enhanced_description"),
            "reference_completion": row.get("reference_completion"),
            "generated_text": row.get("generated_text"),
            "meter": row.get("meter"),
            "count_adherence": row.get("count_adherence"),
            "description_adherence": row.get("description_adherence"),
            "meaning": row.get("meaning"),
            "fluency": row.get("fluency"),
            "coherence": row.get("coherence"),
            "poeticness": row.get("poeticness"),
        }
    elif dataset_key == "ashaar":
        out = {
            "id": row["generation_id"],
            "base_meter": row.get("base_meter"),
            "form": row.get("form"),
            "requested_bayts": row.get("requested_bayts"),
            "requested_num_lines": row.get("requested_num_lines"),
            "description": row.get("description"),
            "enhanced_description": row.get("enhanced_description"),
            "reference_completion": row.get("reference_completion"),
            "generated_text": row.get("generated_text"),
            "meter": row.get("meter"),
            "count_adherence": None,
            "description_adherence": None,
            "meaning": row.get("meaning"),
            "fluency": row.get("fluency"),
            "coherence": row.get("coherence"),
            "poeticness": row.get("poeticness"),
        }
    elif dataset_key == "fanar":
        out = {
            "id": row["generation_id"],
            "base_meter": row.get("base_meter"),
            "form": row.get("form"),
            "requested_bayts": row.get("requested_bayts"),
            "requested_num_lines": row.get("sft_num_lines"),
            "description": row.get("description"),
            "enhanced_description": row.get("enhanced_description"),
            "reference_completion": row.get("reference_completion"),
            "generated_text": row.get("generated_text"),
            "meter": row.get("selected_shaer_meter"),
            "count_adherence": None,
            "description_adherence": None,
            "meaning": row.get("meaning"),
            "fluency": row.get("fluency"),
            "coherence": row.get("coherence"),
            "poeticness": row.get("poeticness"),
        }
    else:
        raise RuntimeError(f"unknown dataset key: {dataset_key}")

    return {key: clean_value(out.get(key)) for key in CLEAN_COLUMNS}


def read_source_dataset(repo_id: str, split: str, token: str, cache_dir: Path):
    return load_dataset(repo_id, split=split, token=token, cache_dir=str(cache_dir))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset_card(dataset_key: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    desc_note = (
        "This dataset includes `description_adherence` because the model was evaluated under the description-conditioned instruction setup."
        if spec["description_adherence_applicable"]
        else "This dataset sets `description_adherence` to null because the baseline was evaluated in its native setup rather than the description-conditioned instruction setup."
    )
    count_note = (
        "This dataset includes `count_adherence` because the model was evaluated with an explicit requested hemistich count."
        if spec["count_adherence_applicable"]
        else "This dataset sets `count_adherence` to null in the clean public export because the native benchmark interface does not expose the same explicit hemistich-count control used for Shaer and Yehia."
    )
    fields = "\n".join(f"- `{c}`" for c in CLEAN_COLUMNS)
    return f"""# {spec['repo_id'].split('/')[-1]}

Clean paper-facing evaluation dataset for the **Shaer** benchmark.

Rows: `{len(rows)}`
Split: `{spec['source_split']}`

## Schema

{fields}

## Notes

- `meter` is the row-level metrical conformity score used in the paper.
- {count_note}
- {desc_note}
- This clean export intentionally removes provenance columns related to internal manifesting, sampling, and selected-generation bookkeeping.
"""


def stage_dataset(dataset_key: str, rows: list[dict[str, Any]], spec: dict[str, Any], output_root: Path) -> Path:
    repo_dir = output_root / dataset_key
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    (repo_dir / "data").mkdir(parents=True, exist_ok=True)
    target_path = repo_dir / "data" / spec["target_filename"]
    if spec["format"] == "parquet":
        pd.DataFrame(rows)[CLEAN_COLUMNS].to_parquet(target_path, index=False)
    elif spec["format"] == "jsonl":
        write_jsonl(target_path, rows)
    else:
        raise RuntimeError(f"unsupported format: {spec['format']}")
    (repo_dir / "README.md").write_text(build_dataset_card(dataset_key, spec, rows), encoding="utf-8")
    return repo_dir


def summarize_rows(dataset_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_for(field: str) -> float | None:
        vals = [float(r[field]) for r in rows if r.get(field) is not None]
        return (sum(vals) / len(vals)) if vals else None

    return {
        "dataset_key": dataset_key,
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "meter_mean": mean_for("meter"),
        "count_adherence_mean": mean_for("count_adherence"),
        "description_adherence_mean": mean_for("description_adherence"),
        "meaning_mean": mean_for("meaning"),
        "fluency_mean": mean_for("fluency"),
        "coherence_mean": mean_for("coherence"),
        "poeticness_mean": mean_for("poeticness"),
    }


def republish_dataset(api: HfApi, spec: dict[str, Any], repo_dir: Path) -> None:
    repo_id = spec["repo_id"]
    api.delete_repo(repo_id=repo_id, repo_type="dataset")
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
    api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Republish clean evaluation dataset without provenance columns",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild and republish clean evaluation datasets without provenance columns.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("evaluation/outputs/clean_eval_dataset_republish"),
        help="Local staging directory for clean dataset repos",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"C:\hfds"),
        help="Short local HF cache directory",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_ORDER,
        default=DATASET_ORDER,
        help="Which datasets to rebuild and republish",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Delete the existing HF dataset repos, recreate them, and upload the clean exports",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    token = find_hf_token(project_root)
    api = HfApi(token=token)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for dataset_key in args.datasets:
        spec = DATASET_SPECS[dataset_key]
        ds = read_source_dataset(spec["repo_id"], spec["source_split"], token, args.cache_dir)
        rows = [normalize_row(dataset_key, dict(row), spec) for row in ds]
        repo_dir = stage_dataset(dataset_key, rows, spec, args.output_root)
        summary = summarize_rows(dataset_key, rows)
        summary["repo_id"] = spec["repo_id"]
        summary["staged_repo_dir"] = str(repo_dir)
        if args.publish:
            republish_dataset(api, spec, repo_dir)
            summary["published"] = True
        else:
            summary["published"] = False
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
