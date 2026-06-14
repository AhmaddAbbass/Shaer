#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

from final_sft_dataset import DEFAULT_MODEL_ID, DEFAULT_FINAL_DATASET_REPO
from prompt_contracts import build_sft_completion, build_sft_full_text, build_sft_prompt


TATWEEL = "ـ"
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ENV_PATH = PROJECT_ROOT / ".env"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_api(push: bool) -> HfApi | None:
    if not push:
        return None
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required when push_to_hub is enabled")
    return HfApi(token=hf_token)


def count_leading_tatweel(text: str) -> int:
    count = 0
    for char in text:
        if char != TATWEEL:
            break
        count += 1
    return count


def count_trailing_tatweel(text: str) -> int:
    count = 0
    for char in reversed(text):
        if char != TATWEEL:
            break
        count += 1
    return count


def strip_all_tatweel(text: str) -> str:
    return text.replace(TATWEEL, "")


def clean_poem_verse(text: str) -> str:
    prefix_len = count_leading_tatweel(text)
    suffix_len = count_trailing_tatweel(text)
    middle_end = len(text) - suffix_len if suffix_len else len(text)
    middle = text[prefix_len:middle_end]
    cleaned_middle = strip_all_tatweel(middle)
    prefix = text[:prefix_len]
    suffix = text[middle_end:]
    return f"{prefix}{cleaned_middle}{suffix}"


def verse_has_internal_tatweel(text: str) -> bool:
    prefix_len = count_leading_tatweel(text)
    suffix_len = count_trailing_tatweel(text)
    middle_end = len(text) - suffix_len if suffix_len else len(text)
    return TATWEEL in text[prefix_len:middle_end]


def row_has_internal_poem_tatweel(verses: list[str]) -> bool:
    return any(verse_has_internal_tatweel(verse) for verse in verses)


def row_has_edge_poem_tatweel(verses: list[str]) -> bool:
    return any(verse.startswith(TATWEEL) or verse.endswith(TATWEEL) for verse in verses)


def validate_verse_preservation(original: str, cleaned: str) -> dict[str, Any]:
    original_prefix = count_leading_tatweel(original)
    cleaned_prefix = count_leading_tatweel(cleaned)
    original_suffix = count_trailing_tatweel(original)
    cleaned_suffix = count_trailing_tatweel(cleaned)
    cleaned_middle_end = len(cleaned) - cleaned_suffix if cleaned_suffix else len(cleaned)
    cleaned_middle = cleaned[cleaned_prefix:cleaned_middle_end]

    return {
        "same_without_tatweel": strip_all_tatweel(original) == strip_all_tatweel(cleaned),
        "prefix_preserved": original[:original_prefix] == cleaned[:cleaned_prefix],
        "suffix_preserved": (
            original[len(original) - original_suffix :] if original_suffix else ""
        ) == (cleaned[len(cleaned) - cleaned_suffix :] if cleaned_suffix else ""),
        "internal_removed": TATWEEL not in cleaned_middle,
        "original_prefix_len": original_prefix,
        "cleaned_prefix_len": cleaned_prefix,
        "original_suffix_len": original_suffix,
        "cleaned_suffix_len": cleaned_suffix,
    }


def clean_row(row: dict[str, Any], tokenizer: AutoTokenizer) -> tuple[dict[str, Any], dict[str, Any]]:
    cleaned = dict(row)
    original_verses = [str(item) for item in row.get("poem verses", [])]
    cleaned_verses = [clean_poem_verse(verse) for verse in original_verses]

    cleaned["poem verses"] = cleaned_verses
    if row.get("description") is not None:
        cleaned["description"] = strip_all_tatweel(str(row["description"]))
    cleaned["enhanced_description"] = strip_all_tatweel(str(row["enhanced_description"]))
    cleaned["sft_prompt"] = build_sft_prompt(
        base_meter=str(row["base_meter"]),
        form=str(row["form"]),
        description=str(cleaned["enhanced_description"]),
        num_lines=len(cleaned_verses),
    )
    cleaned["sft_completion"] = build_sft_completion(cleaned_verses)
    cleaned["sft_full_text"] = build_sft_full_text(cleaned["sft_prompt"], cleaned["sft_completion"])
    cleaned["sft_num_lines"] = len(cleaned_verses)
    cleaned["sft_total_tokens"] = len(
        tokenizer(cleaned["sft_full_text"], add_special_tokens=False)["input_ids"]
    )

    verse_checks = [
        validate_verse_preservation(original, new)
        for original, new in zip(original_verses, cleaned_verses, strict=True)
    ]
    validation = {
        "row_id": row.get("id"),
        "source_index": row.get("source_index"),
        "verse_count_same": len(original_verses) == len(cleaned_verses),
        "all_verses_same_without_tatweel": all(check["same_without_tatweel"] for check in verse_checks),
        "all_edge_prefix_preserved": all(
            check["prefix_preserved"] and check["original_prefix_len"] == check["cleaned_prefix_len"]
            for check in verse_checks
        ),
        "all_edge_suffix_preserved": all(
            check["suffix_preserved"] and check["original_suffix_len"] == check["cleaned_suffix_len"]
            for check in verse_checks
        ),
        "no_internal_tatweel_remaining_in_poem": not row_has_internal_poem_tatweel(cleaned_verses),
        "enhanced_description_has_tatweel": TATWEEL in str(cleaned["enhanced_description"]),
        "description_has_tatweel": TATWEEL in str(cleaned.get("description") or ""),
        "sft_prompt_has_tatweel": TATWEEL in str(cleaned["sft_prompt"]),
        "sft_completion_has_tatweel": row_has_internal_poem_tatweel(cleaned_verses),
        "sft_full_text_has_non_boundary_tatweel": row_has_internal_poem_tatweel(cleaned_verses),
        "original_row_poem_had_internal_tatweel": row_has_internal_poem_tatweel(original_verses),
        "original_row_poem_had_edge_tatweel": row_has_edge_poem_tatweel(original_verses),
        "cleaned_row_poem_has_edge_tatweel": row_has_edge_poem_tatweel(cleaned_verses),
        "verse_checks": verse_checks,
    }
    return cleaned, validation


def build_readme(base_readme_path: Path, summary: dict[str, Any]) -> str:
    base_readme = base_readme_path.read_text(encoding="utf-8").rstrip()
    return (
        f"{base_readme}\n\n"
        "## Tatweel Normalization\n\n"
        "This dataset version preserves edge tatweel in `poem verses` when it appears at the start or end of a shatr,\n"
        "and removes tatweel everywhere else.\n\n"
        f"- Rows processed: **{summary['rows_processed']}**\n"
        f"- Rows with poem tatweel before cleanup: **{summary['rows_with_poem_tatweel_before']}**\n"
        f"- Rows with poem tatweel after cleanup: **{summary['rows_with_poem_tatweel_after']}**\n"
        f"- Rows with internal poem tatweel before cleanup: **{summary['rows_with_internal_poem_tatweel_before']}**\n"
        f"- Rows with internal poem tatweel after cleanup: **{summary.get('rows_with_internal_poem_tatweel_after', 0)}**\n"
        f"- Rows with edge poem tatweel preserved after cleanup: **{summary['rows_with_edge_poem_tatweel_after']}**\n"
        f"- Rows with `enhanced_description` tatweel after cleanup: **{summary.get('rows_with_enhanced_description_tatweel_after', 0)}**\n"
        f"- Rows with `description` tatweel after cleanup: **{summary.get('rows_with_description_tatweel_after', 0)}**\n"
        f"- Validation failures: **{summary['validation_failures']}**\n"
    )


def cmd_clean_publish(args: argparse.Namespace) -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    input_jsonl = Path(args.input_jsonl)
    if not input_jsonl.exists():
        raise FileNotFoundError(f"input final rows file not found: {input_jsonl}")

    final_dir = input_jsonl.parent
    run_root = final_dir.parent
    output_dir = ensure_dir(Path(args.output_dir) if args.output_dir else run_root / "final_tatweel_clean")
    cleaned_rows_path = output_dir / "final_rows_tatweel_clean.jsonl"
    validation_report_path = output_dir / "tatweel_validation_report.json"
    cleanup_summary_path = output_dir / "tatweel_cleanup_summary.json"
    readme_path = output_dir / "README.md"

    if cleaned_rows_path.exists():
        cleaned_rows_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)
    api = resolve_api(push=args.push_to_hub)
    repo_id = args.repo_id or DEFAULT_FINAL_DATASET_REPO
    if api is not None:
        create_repo(repo_id=repo_id, repo_type="dataset", token=api.token, exist_ok=True)

    summary = Counter()
    validation_failures: list[dict[str, Any]] = []
    changed_examples: list[dict[str, Any]] = []

    for row in iter_jsonl(input_jsonl):
        original_verses = [str(item) for item in row.get("poem verses", [])]
        summary["rows_processed"] += 1
        if any(TATWEEL in verse for verse in original_verses):
            summary["rows_with_poem_tatweel_before"] += 1
        if row_has_internal_poem_tatweel(original_verses):
            summary["rows_with_internal_poem_tatweel_before"] += 1
        if row_has_edge_poem_tatweel(original_verses):
            summary["rows_with_edge_poem_tatweel_before"] += 1
        if TATWEEL in str(row.get("enhanced_description") or ""):
            summary["rows_with_enhanced_description_tatweel_before"] += 1
        if TATWEEL in str(row.get("description") or ""):
            summary["rows_with_description_tatweel_before"] += 1

        cleaned_row, validation = clean_row(row, tokenizer)
        cleaned_verses = [str(item) for item in cleaned_row.get("poem verses", [])]

        if any(TATWEEL in verse for verse in cleaned_verses):
            summary["rows_with_poem_tatweel_after"] += 1
        if row_has_internal_poem_tatweel(cleaned_verses):
            summary["rows_with_internal_poem_tatweel_after"] += 1
        if row_has_edge_poem_tatweel(cleaned_verses):
            summary["rows_with_edge_poem_tatweel_after"] += 1
        if TATWEEL in str(cleaned_row.get("enhanced_description") or ""):
            summary["rows_with_enhanced_description_tatweel_after"] += 1
        if TATWEEL in str(cleaned_row.get("description") or ""):
            summary["rows_with_description_tatweel_after"] += 1

        original_tatweel_chars = (
            sum(verse.count(TATWEEL) for verse in original_verses)
            + str(row.get("description") or "").count(TATWEEL)
            + str(row.get("enhanced_description") or "").count(TATWEEL)
        )
        cleaned_tatweel_chars = (
            sum(verse.count(TATWEEL) for verse in cleaned_verses)
            + str(cleaned_row.get("description") or "").count(TATWEEL)
            + str(cleaned_row.get("enhanced_description") or "").count(TATWEEL)
        )
        summary["tatweel_chars_removed"] += original_tatweel_chars - cleaned_tatweel_chars

        if any(original != cleaned for original, cleaned in zip(original_verses, cleaned_verses, strict=True)):
            summary["rows_changed"] += 1
            if len(changed_examples) < 8:
                changed_examples.append(
                    {
                        "id": row.get("id"),
                        "source_index": row.get("source_index"),
                        "before": original_verses,
                        "after": cleaned_verses,
                    }
                )

        checks_ok = (
            validation["verse_count_same"]
            and validation["all_verses_same_without_tatweel"]
            and validation["all_edge_prefix_preserved"]
            and validation["all_edge_suffix_preserved"]
            and validation["no_internal_tatweel_remaining_in_poem"]
            and not validation["enhanced_description_has_tatweel"]
            and not validation["description_has_tatweel"]
            and not validation["sft_prompt_has_tatweel"]
            and not validation["sft_completion_has_tatweel"]
            and not validation["sft_full_text_has_non_boundary_tatweel"]
        )
        if not checks_ok:
            validation_failures.append(validation)
        append_jsonl(cleaned_rows_path, cleaned_row)

    summary["validation_failures"] = len(validation_failures)
    cleanup_summary = {
        "timestamp_utc": utc_now_iso(),
        "repo_id": repo_id,
        **dict(summary),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(cleaned_rows_path),
        "changed_examples": changed_examples,
    }
    validation_report = {
        "timestamp_utc": utc_now_iso(),
        "repo_id": repo_id,
        "failures": validation_failures[:20],
        "failure_count": len(validation_failures),
    }

    dump_json(cleanup_summary_path, cleanup_summary)
    dump_json(validation_report_path, validation_report)

    if validation_failures:
        raise RuntimeError(
            f"tatweel cleanup validation failed for {len(validation_failures)} rows; see {validation_report_path}"
        )

    base_readme_path = final_dir / "README.md"
    readme_text = build_readme(base_readme_path, cleanup_summary)
    readme_path.write_text(readme_text, encoding="utf-8")

    ds = load_dataset("json", data_files=str(cleaned_rows_path), split="train")

    if args.push_to_hub:
        ds.push_to_hub(repo_id=repo_id, token=api.token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        progress_prefix = args.progress_prefix or f"progress/{run_root.name}/tatweel_cleanup"
        api.upload_file(
            path_or_fileobj=str(cleanup_summary_path),
            path_in_repo=f"{progress_prefix}/tatweel_cleanup_summary.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(validation_report_path),
            path_in_repo=f"{progress_prefix}/tatweel_validation_report.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(json.dumps(cleanup_summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strip internal tatweel while preserving shatr-boundary tatweel")
    parser.add_argument(
        "--input-jsonl",
        default="/root/workspace/Shaer/description_generation/outputs/final_sft_regen_v1_20260407_w24/final/final_rows.jsonl",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--repo-id", default=DEFAULT_FINAL_DATASET_REPO)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--progress-prefix", default="")
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cmd_clean_publish(args)


if __name__ == "__main__":
    main()
