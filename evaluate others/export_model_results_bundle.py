#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import atomic_write_json, ensure_dir, json_ready, read_jsonl, utc_now_iso
from metrics import score_generation_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable per-model results bundle with JSONL, CSV, scores, and summary files.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-source-rows", type=int, default=0)
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--dataset-repo-id", default="")
    parser.add_argument("--source-dataset", default="Shaer-AI/shaer-sft-test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-score", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(Path(args.input_jsonl))
    if args.limit:
        rows = rows[: int(args.limit)]
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_jsonl}")

    output_dir = ensure_dir(Path(args.output_dir))
    raw_jsonl = output_dir / "generations.jsonl"
    raw_csv = output_dir / "generations.csv"
    scored_jsonl = output_dir / "generations_scored.jsonl"
    scored_csv = output_dir / "generations_scored.csv"
    validation_json = output_dir / "validation.json"
    aggregate_json = output_dir / "aggregate.json"
    metadata_json = output_dir / "bundle_metadata.json"
    readme_path = output_dir / "README.md"

    write_jsonl(raw_jsonl, rows)
    write_csv(raw_csv, rows)

    scored_rows = score_rows(rows, skip_score=bool(args.skip_score))
    if scored_rows:
        write_jsonl(scored_jsonl, scored_rows)
        write_csv(scored_csv, scored_rows)

    validation = validate_rows(rows, int(args.expected_source_rows), int(args.samples_per_row))
    aggregate = aggregate_rows(scored_rows or rows)
    model_names = sorted({str(row.get("model_name") or "").strip() for row in rows if str(row.get("model_name") or "").strip()})

    metadata = {
        "created_at_utc": utc_now_iso(),
        "input_jsonl": str(Path(args.input_jsonl)),
        "output_dir": str(output_dir),
        "dataset_repo_id": str(args.dataset_repo_id or ""),
        "source_dataset": str(args.source_dataset),
        "row_count": len(rows),
        "model_names": model_names,
        "expected_source_rows": int(args.expected_source_rows),
        "samples_per_row": int(args.samples_per_row),
        "files": {
            "generations_jsonl": raw_jsonl.name,
            "generations_csv": raw_csv.name,
            "generations_scored_jsonl": scored_jsonl.name if scored_rows else "",
            "generations_scored_csv": scored_csv.name if scored_rows else "",
            "validation_json": validation_json.name,
            "aggregate_json": aggregate_json.name,
            "readme": readme_path.name,
        },
        "scored_rows_written": bool(scored_rows),
    }

    atomic_write_json(validation_json, validation)
    atomic_write_json(aggregate_json, aggregate)
    atomic_write_json(metadata_json, metadata)
    readme_path.write_text(build_dataset_card(metadata, validation, aggregate), encoding="utf-8")

    print(f"bundle_dir={output_dir} rows={len(rows)} models={','.join(model_names)}")
    return 0


def score_rows(rows: list[dict[str, Any]], skip_score: bool) -> list[dict[str, Any]]:
    if skip_score:
        return []
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        scored = dict(row)
        try:
            scored.update(score_generation_row(row))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Scoring dependencies are missing. Ensure the repo includes grpo/rewards or rerun with --skip-score."
            ) from exc
        scored_rows.append(scored)
    return scored_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify_csv_value(row.get(key)) for key in fieldnames})


def stringify_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True)


def validate_rows(rows: list[dict[str, Any]], expected_source_rows: int, samples_per_row: int) -> dict[str, Any]:
    key_counts = Counter((row.get("model_name"), input_row_id(row), row.get("sample_index")) for row in rows)
    duplicates = [key for key, count in key_counts.items() if count > 1]
    bad_rows = [
        row
        for row in rows
        if row.get("generation_status") != "ok"
        or row.get("health_status") != "ok"
        or not str(row.get("generated_text") or "").strip()
    ]
    per_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_model[str(row.get("model_name") or "")].append(row)

    models_summary: dict[str, dict[str, int | bool]] = {}
    overall_valid = not duplicates and not bad_rows
    for model_name, model_rows in sorted(per_model.items()):
        unique_pairs = {
            (input_row_id(row), row.get("sample_index"))
            for row in model_rows
            if row.get("generation_status") == "ok" and row.get("health_status") == "ok"
        }
        expected_rows = expected_source_rows * samples_per_row if expected_source_rows else 0
        model_valid = len(unique_pairs) == expected_rows if expected_rows else True
        overall_valid = overall_valid and model_valid
        models_summary[model_name] = {
            "rows": len(model_rows),
            "valid_pairs": len(unique_pairs),
            "expected_rows": expected_rows,
            "valid": bool(model_valid),
        }
    return {
        "created_at_utc": utc_now_iso(),
        "row_count": len(rows),
        "duplicate_key_count": len(duplicates),
        "bad_row_count": len(bad_rows),
        "model_count": len(per_model),
        "models": models_summary,
        "valid": bool(overall_valid),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[f"model:{row.get('model_name', '')}"].append(row)
        groups[f"group:{row.get('model_group', '')}"].append(row)
        groups[f"meter:{row.get('base_meter', '')}"].append(row)
        groups[f"form:{row.get('form', '')}"].append(row)
        groups[f"model_meter:{row.get('model_name', '')}|{row.get('base_meter', '')}"].append(row)
    return {group_name: summarize(group_rows) for group_name, group_rows in groups.items()}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    meter_vals = [float(row.get("meter", 0.0) or 0.0) for row in rows]
    count_vals = [float(row.get("count_adherence", 0.0) or 0.0) for row in rows]
    exact_count = [
        int(int(row.get("parsed_num_lines") or 0) == int(row.get("requested_num_lines") or -1))
        for row in rows
        if row.get("requested_num_lines") is not None
    ]
    return {
        "rows": len(rows),
        "meter_mean": avg(meter_vals),
        "count_adherence_mean": avg(count_vals),
        "exact_count_rate": avg(exact_count),
    }


def avg(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def input_row_id(row: dict[str, Any]) -> str:
    for key in ("input_row_id", "manifest_row_id", "selected_shaer_generation_id", "generation_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return str(row.get("source_row_index") or "")


def build_dataset_card(metadata: dict[str, Any], validation: dict[str, Any], aggregate: dict[str, Any]) -> str:
    repo_id = str(metadata.get("dataset_repo_id") or "").strip()
    model_names = ", ".join(metadata.get("model_names", [])) or "unknown_model"
    all_summary = aggregate.get("all", {})
    lines = [
        "---",
        "pretty_name: Shaer Evaluation Results",
        f"dataset_info:\n- config_name: default",
        "---",
        "",
        "# Shaer Evaluation Results",
        "",
        f"- Models: `{model_names}`",
        f"- Source dataset: `{metadata.get('source_dataset', '')}`",
        f"- Rows: `{metadata.get('row_count', 0)}`",
        f"- Validation passed: `{validation.get('valid', False)}`",
        f"- Scored rows included: `{metadata.get('scored_rows_written', False)}`",
    ]
    if repo_id:
        lines.append(f"- Dataset repo: `{repo_id}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{metadata['files']['generations_jsonl']}`: raw generation rows",
            f"- `{metadata['files']['generations_csv']}`: raw generation rows in CSV",
            f"- `{metadata['files']['validation_json']}`: validation summary",
            f"- `{metadata['files']['aggregate_json']}`: aggregate metrics",
            "",
            "## Aggregate",
            "",
            f"- Meter mean: `{all_summary.get('meter_mean', 0.0):.6f}`",
            f"- Count adherence mean: `{all_summary.get('count_adherence_mean', 0.0):.6f}`",
            f"- Exact count rate: `{all_summary.get('exact_count_rate', 0.0):.6f}`",
        ]
    )
    if metadata["files"]["generations_scored_jsonl"]:
        lines.insert(lines.index(f"- `{metadata['files']['validation_json']}`: validation summary"), f"- `{metadata['files']['generations_scored_jsonl']}`: raw rows plus meter/count evaluation")
        lines.insert(lines.index(f"- `{metadata['files']['validation_json']}`: validation summary") + 1, f"- `{metadata['files']['generations_scored_csv']}`: scored rows in CSV")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
