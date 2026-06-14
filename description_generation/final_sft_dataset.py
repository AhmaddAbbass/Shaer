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
import pyarrow.ipc as ipc
from transformers import AutoTokenizer

from prompt_contracts import (
    DESCRIPTION_PROMPT_VERSION,
    DESCRIPTION_SYSTEM_PROMPT,
    FINAL_SFT_PROMPT_VERSION,
    FINAL_SFT_SYSTEM_PROMPT,
    FINAL_SFT_USER_TEMPLATE,
    build_description_user_prompt,
    build_sft_completion,
    build_sft_full_text,
    build_sft_prompt,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_SOURCE_DATASET = "Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed"
DEFAULT_FINAL_DATASET_REPO = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500"
DEFAULT_MODEL_ID = "Navid-AI/Yehia-7B-preview"
DEFAULT_NUM_WORKERS = 10
DEFAULT_MAX_BAYTS = 20
DEFAULT_MIN_METER_COUNT = 500
DEFAULT_MAX_TOKENS = 2048
DATASET_CACHE_ROOT = Path("/root/.cache/huggingface/datasets")
DATASET_CACHE_GLOB = (
    "Shaer-AI___ashaar-with-descriptions-baseform-final-trimmed/default/0.0.0/*/"
    "ashaar-with-descriptions-baseform-final-trimmed-train-*.arrow"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_poem_verses(verses: Any) -> list[str]:
    if not isinstance(verses, list):
        return []
    out: list[str] = []
    for item in verses:
        text = str(item).strip()
        if not text:
            return []
        out.append(text)
    return out


def requested_bayts_from_verses(verses: list[str]) -> int | None:
    if not verses or len(verses) % 2 != 0:
        return None
    return len(verses) // 2


def build_prepare_summary(
    source_dataset: str,
    repo_id: str,
    source_rows: int,
    after_valid_rows: int,
    after_bayt_filter_rows: int,
    final_rows: int,
    num_workers: int,
    max_bayts: int,
    min_meter_count: int,
    dropped_meters: list[str],
    invalid_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "timestamp_utc": utc_now_iso(),
        "source_dataset": source_dataset,
        "repo_id": repo_id,
        "source_rows": source_rows,
        "after_valid_rows": after_valid_rows,
        "after_bayt_filter_rows": after_bayt_filter_rows,
        "final_rows": final_rows,
        "num_workers": num_workers,
        "max_bayts": max_bayts,
        "min_meter_count": min_meter_count,
        "dropped_meters": dropped_meters,
        "invalid_counts": invalid_counts,
        "description_prompt_version": DESCRIPTION_PROMPT_VERSION,
        "final_sft_prompt_version": FINAL_SFT_PROMPT_VERSION,
    }


def prepare_repo(api: HfApi | None, repo_id: str) -> None:
    if api is None:
        return
    create_repo(repo_id=repo_id, repo_type="dataset", token=api.token, exist_ok=True)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_api(push: bool) -> HfApi | None:
    if not push:
        return None
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required when push_to_hub is enabled")
    return HfApi(token=hf_token)


def find_cached_source_arrow_paths() -> list[Path]:
    paths = sorted(DATASET_CACHE_ROOT.glob(DATASET_CACHE_GLOB))
    if not paths:
        raise FileNotFoundError("could not find cached arrow files for the source dataset")
    return paths


def iter_source_rows(source_dataset: str) -> Any:
    if source_dataset == DEFAULT_SOURCE_DATASET:
        for path in find_cached_source_arrow_paths():
            with ipc.open_stream(path) as reader:
                table = reader.read_all()
            for row in table.to_pylist():
                yield row
        return

    ds = load_dataset(source_dataset, split="train")
    for row in ds:
        yield row


def upload_prepare_artifacts(api: HfApi | None, repo_id: str, progress_prefix: str, files: list[Path]) -> None:
    if api is None:
        return
    for file_path in files:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"{progress_prefix}/{file_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )


def build_final_readme(
    source_dataset: str,
    repo_id: str,
    summary: dict[str, Any],
    publish_summary: dict[str, Any],
) -> str:
    return f"""---
language:
- ar
license: apache-2.0
pretty_name: Ashaar Final SFT Dataset with Enhanced Descriptions ({datetime.now(timezone.utc).date().isoformat()})
task_categories:
- text-generation
size_categories:
- 100K<n<1M
---

# Ashaar Final SFT Dataset with Enhanced Descriptions

This dataset is derived from `{source_dataset}` and is intended to be the final SFT-ready dataset we continue working with.

We got here the hard way. GRPO did not deliver a convincing improvement. Continuation SFT degraded. A fresh-from-zero SFT direction still exposed a deeper data problem. After inspecting the conditioning text, we concluded that many of the old descriptions were weak or noisy enough to hurt the prompt quality, so this dataset regenerates them into a new `enhanced_description` field and rebuilds the SFT prompt from that improved field.

## Source Lineage

- Source dataset: `{source_dataset}`
- Final dataset repo: `{repo_id}`
- Description prompt version: `{DESCRIPTION_PROMPT_VERSION}`
- Final SFT prompt version: `{FINAL_SFT_PROMPT_VERSION}`

## Filtering

- `requested_bayts = len(poem verses) // 2`
- keep only rows with valid even-length `poem verses`
- keep only rows with `requested_bayts <= {summary["max_bayts"]}`
- drop base meters with post-filter support `< {summary["min_meter_count"]}`
- dropped meters: `{", ".join(summary["dropped_meters"]) if summary["dropped_meters"] else "none"}`

## Counts

- Source rows: **{summary["source_rows"]}**
- After valid-row filtering: **{summary["after_valid_rows"]}**
- After `<= {summary["max_bayts"]}` bayts filtering: **{summary["after_bayt_filter_rows"]}**
- Final staged rows before regeneration: **{summary["final_rows"]}**
- Regenerated rows merged successfully: **{publish_summary["merged_rows"]}**
- Removed by final token cap (`> {publish_summary["max_tokens"]}`): **{publish_summary["rows_over_token_cap"]}**
- Final published rows: **{publish_summary["published_rows"]}**

## Description Regeneration Prompt Sent to Qwen

### SYSTEM_PROMPT
```text
{DESCRIPTION_SYSTEM_PROMPT}
```

### USER_PROMPT_TEMPLATE
```text
{build_description_user_prompt(["{poem_shatr_1}", "{poem_shatr_2}"])}
```

## Final SFT Prompt Contract

### SYSTEM_PROMPT
```text
{FINAL_SFT_SYSTEM_PROMPT}
```

### USER_TEMPLATE
```text
{FINAL_SFT_USER_TEMPLATE}
```

## Columns

- original source columns are preserved
- `description` is preserved as the old field
- `enhanced_description` is the regenerated field
- `sft_prompt` is rebuilt from `enhanced_description`
- `sft_completion`, `sft_full_text`, `sft_num_lines`, and `sft_total_tokens` are rebuilt from the new prompt contract

## Notes

- This repo keeps the old description for comparison, but the rebuilt training prompt uses `enhanced_description`.
- The purpose of the stronger prompt contract is to make the generation structure respond more strongly to the required meter, not merely mention it once as metadata.
"""


def cmd_prepare(args: argparse.Namespace) -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    run_name = args.run_name or f"final_sft_regen_{timestamp_slug()}"
    run_root = ensure_dir(OUTPUT_ROOT / run_name)
    shards_dir = ensure_dir(run_root / "shards")
    workers_dir = ensure_dir(run_root / "workers")

    api = resolve_api(push=args.push_progress)
    prepare_repo(api, args.repo_id)

    staged_rows: list[dict[str, Any]] = []
    invalid_counts = Counter()
    meter_counts_before = Counter()
    source_rows = 0

    for source_index, row in enumerate(iter_source_rows(args.source_dataset)):
        source_rows += 1
        verses = normalize_poem_verses(row.get("poem verses"))
        if not verses:
            invalid_counts["invalid_or_empty_poem_verses"] += 1
            continue
        if len(verses) % 2 != 0:
            invalid_counts["odd_shatr_rows"] += 1
            continue
        requested_bayts = requested_bayts_from_verses(verses)
        if requested_bayts is None:
            invalid_counts["missing_requested_bayts"] += 1
            continue
        if requested_bayts > args.max_bayts:
            invalid_counts["over_max_bayts"] += 1
            continue
        prepared = dict(row)
        prepared["poem verses"] = verses
        prepared["source_index"] = int(source_index)
        prepared["requested_bayts"] = int(requested_bayts)
        staged_rows.append(prepared)
        meter_counts_before[str(prepared["base_meter"]).strip()] += 1

    after_valid_rows = source_rows - invalid_counts["invalid_or_empty_poem_verses"] - invalid_counts["odd_shatr_rows"]
    after_bayt_filter_rows = len(staged_rows)

    dropped_meters = sorted(
        meter for meter, count in meter_counts_before.items() if count < args.min_meter_count
    )
    staged_rows = [
        row for row in staged_rows if str(row["base_meter"]).strip() not in dropped_meters
    ]
    meter_counts_after = Counter(str(row["base_meter"]).strip() for row in staged_rows)

    staged_rows_path = run_root / "staged_rows.jsonl"
    if staged_rows_path.exists():
        staged_rows_path.unlink()
    for row in staged_rows:
        append_jsonl(staged_rows_path, row)

    per_worker_base, remainder = divmod(len(staged_rows), args.num_workers)
    worker_shards: list[dict[str, Any]] = []
    seen_source_indices: set[int] = set()
    cursor = 0
    for worker_id in range(args.num_workers):
        shard_size = per_worker_base + (1 if worker_id < remainder else 0)
        shard_rows = staged_rows[cursor : cursor + shard_size]
        cursor += shard_size
        shard_path = shards_dir / f"worker_{worker_id:02d}.jsonl"
        if shard_path.exists():
            shard_path.unlink()
        for row in shard_rows:
            source_index = int(row["source_index"])
            if source_index in seen_source_indices:
                raise RuntimeError(f"duplicate source_index detected in shard building: {source_index}")
            seen_source_indices.add(source_index)
            append_jsonl(shard_path, row)
        source_indices = [int(row["source_index"]) for row in shard_rows]
        worker_shards.append(
            {
                "worker_id": worker_id,
                "shard_path": str(shard_path),
                "output_dir": str(workers_dir / f"worker_{worker_id:02d}"),
                "row_count": len(shard_rows),
                "source_index_min": min(source_indices) if source_indices else None,
                "source_index_max": max(source_indices) if source_indices else None,
            }
        )

    if len(seen_source_indices) != len(staged_rows):
        raise RuntimeError("worker shard union does not match staged rows")

    progress_prefix = f"progress/{run_name}"
    manifest = {
        "manifest_version": 1,
        "timestamp_utc": utc_now_iso(),
        "run_name": run_name,
        "source_dataset": args.source_dataset,
        "repo_id": args.repo_id,
        "max_bayts": args.max_bayts,
        "min_meter_count": args.min_meter_count,
        "dropped_meters": dropped_meters,
        "num_workers": args.num_workers,
        "stage_root": str(run_root),
        "staged_rows_path": str(staged_rows_path),
        "workers_dir": str(workers_dir),
        "progress_prefix": progress_prefix,
        "description_prompt_version": DESCRIPTION_PROMPT_VERSION,
        "final_sft_prompt_version": FINAL_SFT_PROMPT_VERSION,
        "worker_shards": worker_shards,
    }
    manifest_path = run_root / "workers_manifest.json"
    dump_json(manifest_path, manifest)
    meter_counts_before_path = run_root / "meter_counts_before.json"
    meter_counts_after_path = run_root / "meter_counts_after.json"
    dump_json(meter_counts_before_path, dict(sorted(meter_counts_before.items())))
    dump_json(meter_counts_after_path, dict(sorted(meter_counts_after.items())))

    summary = build_prepare_summary(
        source_dataset=args.source_dataset,
        repo_id=args.repo_id,
        source_rows=source_rows,
        after_valid_rows=after_valid_rows,
        after_bayt_filter_rows=after_bayt_filter_rows,
        final_rows=len(staged_rows),
        num_workers=args.num_workers,
        max_bayts=args.max_bayts,
        min_meter_count=args.min_meter_count,
        dropped_meters=dropped_meters,
        invalid_counts=dict(invalid_counts),
    )
    summary_path = run_root / "preprocess_summary.json"
    dump_json(summary_path, summary)

    upload_prepare_artifacts(
        api,
        args.repo_id,
        progress_prefix,
        [manifest_path, meter_counts_before_path, meter_counts_after_path, summary_path],
    )

    print(json.dumps({"run_root": str(run_root), "manifest_path": str(manifest_path), **summary}, ensure_ascii=False, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    manifest = load_manifest(Path(args.workers_manifest))
    source_index_to_worker: dict[int, int] = {}
    total_rows = 0
    min_size = None
    max_size = None
    for shard in manifest["worker_shards"]:
        shard_rows = list(iter_jsonl(Path(shard["shard_path"])))
        row_count = len(shard_rows)
        total_rows += row_count
        min_size = row_count if min_size is None else min(min_size, row_count)
        max_size = row_count if max_size is None else max(max_size, row_count)
        for row in shard_rows:
            source_index = int(row["source_index"])
            if source_index in source_index_to_worker:
                raise RuntimeError(
                    f"source_index {source_index} appears in worker {source_index_to_worker[source_index]} and worker {shard['worker_id']}"
                )
            source_index_to_worker[source_index] = int(shard["worker_id"])

    staged_rows = list(iter_jsonl(Path(manifest["staged_rows_path"])))
    staged_source_indices = {int(row["source_index"]) for row in staged_rows}
    shard_source_indices = set(source_index_to_worker.keys())
    if staged_source_indices != shard_source_indices:
        missing = sorted(staged_source_indices - shard_source_indices)[:10]
        extra = sorted(shard_source_indices - staged_source_indices)[:10]
        raise RuntimeError(f"manifest mismatch | missing={missing} | extra={extra}")

    summary = {
        "timestamp_utc": utc_now_iso(),
        "manifest_path": str(args.workers_manifest),
        "worker_count": len(manifest["worker_shards"]),
        "staged_rows": len(staged_rows),
        "sharded_rows": total_rows,
        "min_shard_size": min_size,
        "max_shard_size": max_size,
        "balanced_to_within_one_row": (max_size - min_size) <= 1 if min_size is not None and max_size is not None else True,
        "has_overlap": False,
        "has_missing_rows": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_successful_results_from_jsonl(results_path: Path, results_by_source: dict[int, dict[str, Any]]) -> None:
    if not results_path.exists():
        return
    for row in iter_jsonl(results_path):
        if str(row.get("status")) != "ok":
            continue
        source_index = int(row["source_index"])
        if source_index in results_by_source:
            raise RuntimeError(f"duplicate successful result for source_index {source_index}")
        results_by_source[source_index] = row


def load_worker_results(manifests: list[dict[str, Any]], extra_results_jsonl: list[str]) -> dict[int, dict[str, Any]]:
    results_by_source: dict[int, dict[str, Any]] = {}
    for manifest in manifests:
        for shard in manifest["worker_shards"]:
            worker_dir = Path(shard["output_dir"])
            results_path = worker_dir / "results.jsonl"
            load_successful_results_from_jsonl(results_path, results_by_source)
    for path_str in extra_results_jsonl:
        load_successful_results_from_jsonl(Path(path_str), results_by_source)
    return results_by_source


def cmd_merge_publish(args: argparse.Namespace) -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    manifest = load_manifest(Path(args.workers_manifest))
    extra_manifests = [load_manifest(Path(path)) for path in args.extra_results_manifest]
    api = resolve_api(push=args.push_to_hub)
    prepare_repo(api, args.repo_id or manifest["repo_id"])

    run_root = Path(manifest["stage_root"])
    final_dir = ensure_dir(run_root / "final")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=False)

    worker_results = load_worker_results([manifest, *extra_manifests], args.extra_results_jsonl)
    staged_rows = list(iter_jsonl(Path(manifest["staged_rows_path"])))

    final_rows_path = final_dir / "final_rows.jsonl"
    if final_rows_path.exists():
        final_rows_path.unlink()

    merged_rows = 0
    rows_over_token_cap = 0
    missing_results: list[int] = []

    for row in staged_rows:
        source_index = int(row["source_index"])
        result = worker_results.get(source_index)
        if result is None:
            missing_results.append(source_index)
            continue
        enhanced_description = str(result["enhanced_description"]).strip()
        prompt = build_sft_prompt(
            base_meter=str(row["base_meter"]),
            form=str(row["form"]),
            description=enhanced_description,
            num_lines=len(row["poem verses"]),
        )
        completion = build_sft_completion(row["poem verses"])
        full_text = build_sft_full_text(prompt, completion)
        token_count = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        if token_count > args.max_tokens:
            rows_over_token_cap += 1
            continue
        final_row = dict(row)
        final_row["enhanced_description"] = enhanced_description
        final_row["sft_prompt"] = prompt
        final_row["sft_completion"] = completion
        final_row["sft_full_text"] = full_text
        final_row["sft_num_lines"] = len(row["poem verses"])
        final_row["sft_total_tokens"] = token_count
        append_jsonl(final_rows_path, final_row)
        merged_rows += 1

    if missing_results:
        raise RuntimeError(
            f"cannot publish final dataset: {len(missing_results)} rows are missing successful enhanced_description results"
        )

    ds = load_dataset("json", data_files=str(final_rows_path), split="train")
    repo_id = args.repo_id or manifest["repo_id"]

    publish_summary = {
        "timestamp_utc": utc_now_iso(),
        "repo_id": repo_id,
        "merged_rows": len(staged_rows),
        "rows_over_token_cap": rows_over_token_cap,
        "extra_results_manifests": args.extra_results_manifest,
        "extra_results_jsonl": args.extra_results_jsonl,
        "published_rows": merged_rows,
        "max_tokens": args.max_tokens,
    }
    summary_path = final_dir / "publish_summary.json"
    dump_json(summary_path, publish_summary)

    readme_text = build_final_readme(
        source_dataset=manifest["source_dataset"],
        repo_id=repo_id,
        summary=json.loads((run_root / "preprocess_summary.json").read_text(encoding="utf-8")),
        publish_summary=publish_summary,
    )
    readme_path = final_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")

    if args.push_to_hub:
        ds.push_to_hub(repo_id=repo_id, token=api.token)
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(summary_path),
            path_in_repo=f"{manifest['progress_prefix']}/publish_summary.json",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(json.dumps(publish_summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and publish the final enhanced-description SFT dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-name", default="")
    prepare.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    prepare.add_argument("--repo-id", default=DEFAULT_FINAL_DATASET_REPO)
    prepare.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    prepare.add_argument("--max-bayts", type=int, default=DEFAULT_MAX_BAYTS)
    prepare.add_argument("--min-meter-count", type=int, default=DEFAULT_MIN_METER_COUNT)
    prepare.add_argument("--push-progress", action="store_true", default=True)
    prepare.set_defaults(func=cmd_prepare)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--workers-manifest", required=True)
    verify.set_defaults(func=cmd_verify)

    merge_publish = subparsers.add_parser("merge_publish")
    merge_publish.add_argument("--workers-manifest", required=True)
    merge_publish.add_argument("--extra-results-manifest", action="append", default=[])
    merge_publish.add_argument("--extra-results-jsonl", action="append", default=[])
    merge_publish.add_argument("--repo-id", default="")
    merge_publish.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    merge_publish.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    merge_publish.add_argument("--push-to-hub", action="store_true", default=True)
    merge_publish.set_defaults(func=cmd_merge_publish)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
