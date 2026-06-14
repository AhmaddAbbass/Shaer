#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    DEFAULT_OUTPUT_DATASET_ID,
    atomic_write_json,
    difficulty_rule_spec,
    ensure_dir,
    load_dotenv_if_present,
    load_manifest,
    load_run_config,
    read_json_or_none,
    row_score_summary,
    scored_path,
    utc_now_iso,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble scored meter/count artifacts into a derived train dataset.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", default="")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional assembly cap for smoke artifacts.")
    parser.add_argument("--include-completions", action="store_true", help="Keep candidate completion text in the assembled dataset.")
    parser.add_argument("--skip-scored-backup", action="store_true", help="Do not build/upload the compressed raw scored backup artifact.")
    return parser.parse_args()


def slim_preprocess_metadata(scored: dict[str, Any]) -> dict[str, Any]:
    gen = dict(scored.get("generator_metadata") or {})
    score = dict(scored.get("scoring_metadata") or {})
    return {
        "generator_id": gen.get("generator_id", ""),
        "generator_shard_id": int(scored["generator_shard_id"]),
        "worker_id": score.get("worker_id", ""),
        "num_candidates": int(scored["num_candidates_scored"]),
        "generator_batch_size": int(gen.get("batch_size") or 0),
        "generator_max_new_tokens": int(gen.get("max_new_tokens") or 0),
        "temperature": float(gen.get("temperature") or 0.0),
        "top_p": float(gen.get("top_p") or 0.0),
        "score_runtime_seconds": float(score.get("runtime_seconds") or 0.0),
    }


def ensured_score_fields(scored: dict[str, Any]) -> dict[str, Any]:
    if "mean_meter_score" in scored and "difficulty_rule" in scored:
        return scored
    summary = row_score_summary(scored["candidate_scores"])
    merged = dict(scored)
    merged["mean_meter_score"] = float(summary["mean_meter_score"])
    merged["mean_count_adherence_score"] = float(summary["mean_count_adherence_score"])
    merged["num_strong_meter_candidates"] = int(summary["num_strong_meter_candidates"])
    merged["num_strong_exact_candidates"] = int(summary["num_strong_exact_candidates"])
    merged["num_bad_meter_candidates"] = int(summary["num_bad_meter_candidates"])
    merged["strong_meter_rate"] = float(summary["strong_meter_rate"])
    merged["strong_exact_rate"] = float(summary["strong_exact_rate"])
    merged["bad_meter_rate"] = float(summary["bad_meter_rate"])
    merged["difficulty_rule"] = summary["difficulty_rule"]
    merged["difficulty"] = summary["difficulty"]
    return merged


def derived_row(scored: dict[str, Any], *, include_completions: bool) -> dict[str, Any]:
    scored = ensured_score_fields(scored)
    candidate_rows = []
    for cand in scored["candidate_scores"]:
        row = {
            "candidate_id": cand["candidate_id"],
            "generation_index": int(cand["generation_index"]),
            "meter_score": float(cand["meter_score"]),
            "meter_mean_score": float(cand["meter_mean_score"]),
            "meter_logmean_score": float(cand["meter_logmean_score"]),
            "meter_target_used": cand["meter_target_used"],
            "meter_target_resolution": cand["meter_target_resolution"],
            "meter_num_valid_bayts": int(cand["meter_num_valid_bayts"]),
            "meter_num_skipped_bayts": int(cand["meter_num_skipped_bayts"]),
            "count_adherence_score": float(cand["count_adherence_score"]),
            "generated_bayts": int(cand["generated_bayts"]),
            "count_requested_bayts": int(cand["count_requested_bayts"]),
            "count_has_odd_tail": bool(cand["count_has_odd_tail"]),
            "finish_reason": cand.get("finish_reason", ""),
            "token_count": int(cand.get("token_count") or 0),
            "cumulative_logprob": cand.get("cumulative_logprob"),
        }
        if include_completions:
            row["completion"] = cand["completion"]
            row["per_bayt_meter_scores"] = cand["per_bayt_meter_scores"]
            row["per_bayt_meter_details"] = cand["per_bayt_meter_details"]
        candidate_rows.append(row)
    return {
        "row_uid": scored["row_uid"],
        "source_dataset_id": scored["source_dataset_id"],
        "source_split": scored["source_split"],
        "source_index": int(scored["source_index"]),
        "source_row_index_in_split": int(scored["source_row_index_in_split"]),
        "source_id": scored["source_id"],
        "row_preprocess_status": scored["row_preprocess_status"],
        "difficulty": scored["difficulty"],
        "base_meter": scored["base_meter"],
        "form": scored["form"],
        "meter_label": scored["meter_label"],
        "requested_bayts": int(scored["requested_bayts"]),
        "requested_lines": int(scored["requested_lines"]),
        "length_bucket": scored["length_bucket"],
        "sampler_group": scored["sampler_group"],
        "split_group": scored["split_group"],
        "description": scored["description"],
        "enhanced_description": scored["enhanced_description"],
        "sft_prompt": scored["sft_prompt"],
        "poem_url": scored["poem_url"],
        "num_candidates_requested": int(scored["num_candidates_requested"]),
        "num_candidates_generated": int(scored["num_candidates_generated"]),
        "num_candidates_scored": int(scored["num_candidates_scored"]),
        "candidates": candidate_rows,
        "mean_meter_score": float(scored["mean_meter_score"]),
        "mean_count_adherence_score": float(scored["mean_count_adherence_score"]),
        "num_strong_meter_candidates": int(scored["num_strong_meter_candidates"]),
        "num_strong_exact_candidates": int(scored["num_strong_exact_candidates"]),
        "num_bad_meter_candidates": int(scored["num_bad_meter_candidates"]),
        "strong_meter_rate": float(scored["strong_meter_rate"]),
        "strong_exact_rate": float(scored["strong_exact_rate"]),
        "bad_meter_rate": float(scored["bad_meter_rate"]),
        "meter_summary": scored["meter_summary"],
        "count_adherence_summary": scored["count_adherence_summary"],
        "meter_count_combined_summary": scored["meter_count_combined_summary"],
        "meter_count_success_rate": float(scored["meter_count_success_rate"]),
        "count_exact_rate": float(scored["count_exact_rate"]),
        "generator_shard_id": int(scored["generator_shard_id"]),
        "manifest_order": int(scored["manifest_order"]),
        "difficulty_rule": scored["difficulty_rule"],
        "preprocess_metadata": slim_preprocess_metadata(scored),
    }


def markdown_count_lines(counts: dict[str, Any]) -> str:
    if not counts:
        return "- none"
    return "\n".join(f"- `{key}`: **{int(value)}**" for key, value in counts.items())


def build_readme(repo_id: str, cfg: dict[str, Any], summary: dict[str, Any]) -> str:
    source_dataset_id = str(cfg["dataset_id"])
    parent_dataset_id = source_dataset_id.removesuffix("-splits") if source_dataset_id.endswith("-splits") else source_dataset_id
    difficulty_counts = dict(summary.get("difficulty_counts") or {})
    meter_counts = dict(summary.get("base_meter_counts") or {})
    num_candidates = int(summary.get("candidate_count_max") or cfg.get("num_candidates") or 0)
    difficulty_rule = difficulty_rule_spec(num_candidates) if num_candidates > 0 else {}
    manifest_rows = int(summary.get("manifest_rows") or 0)
    candidate_count_sum = int(summary.get("candidate_count_sum") or 0)
    missing_scored = int(summary.get("missing_scored_rows") or 0)
    scored_backup_mb = float(int(summary.get("scored_backup_bytes") or 0) / (1024 * 1024)) if summary.get("scored_backup_bytes") else 0.0
    difficulty_rule_path = "artifacts/difficulty_rule.json"
    assembly_summary_path = "artifacts/assembly_summary.json"
    scored_backup_path = "artifacts/scored_rows.jsonl.gz"
    return f"""---
language:
- ar
license: apache-2.0
pretty_name: Shaer GRPO Preprocess Meter Count V1
task_categories:
- text-generation
size_categories:
- 100K<n<1M
---

# Shaer GRPO Preprocess: Meter + Count

This is a derived preprocessing dataset for future GRPO-side analysis and difficulty-aware sampling.

It is built from the `train` split of the released stratified split dataset, and it keeps one derived row per source row together with candidate-level meter/count scoring signals.

This repo is intentionally **not** a chosen-completion dataset:
- there is no selected / chosen completion column
- the main `train` table is a score-bearing dataset for relabeling and analysis
- the full scored backup artifact in this repo keeps the richer per-candidate records, including completion text

## Dataset lineage

Upstream datasets:
- parent dataset: `{parent_dataset_id}`
- split dataset consumed here: `{source_dataset_id}`
- derived output dataset: `{repo_id}`

The split dataset publishes deterministic `train / eval / test` splits with a `94 / 3 / 3` policy.

Original split policy inherited from the source split dataset:
- primary stratification key:
  - `base_meter`
  - `form`
  - `length_bucket`
- length buckets:
  - `1-3`
  - `4-6`
  - `7-10`
  - `11-20`
- small groups fall back gracefully when needed

Known counts in the source split dataset:
- `train`: **109070**
- `eval`: **3481**
- `test`: **3481**

This derived repo uses **train only**.

## Run configuration

Generation model:
- base model: `{cfg["base_model_id"]}`
- adapter repo: `{cfg["sft_adapter_repo"]}`
- adapter mode: `{cfg["sft_adapter_mode"]}`

Preprocess configuration:
- source split: `{cfg["source_split"]}`
- rows assembled: **{manifest_rows}**
- missing scored rows at assembly time: **{missing_scored}**
- sampled candidates per row: **{num_candidates}**
- total candidate records: **{candidate_count_sum}**
- preprocess design: `{cfg.get("preprocess_design", "")}`

Difficulty counts in this release:
{markdown_count_lines(difficulty_counts)}

Base-meter counts in this release:
{markdown_count_lines(meter_counts)}

## What the main `train` dataset contains

Every row preserves the source row identity plus derived preprocess outputs.

Important row-level fields:
- source identity: `row_uid`, `source_dataset_id`, `source_split`, `source_index`, `source_row_index_in_split`, `source_id`
- source context retained for downstream analysis: `base_meter`, `form`, `meter_label`, `requested_bayts`, `requested_lines`, `length_bucket`, `sampler_group`, `split_group`, `description`, `enhanced_description`, `sft_prompt`, `poem_url`
- row-level derived metrics: `difficulty`, `mean_meter_score`, `mean_count_adherence_score`, `num_strong_meter_candidates`, `num_strong_exact_candidates`, `num_bad_meter_candidates`, `strong_meter_rate`, `strong_exact_rate`, `bad_meter_rate`
- row-level aggregate summaries: `meter_summary`, `count_adherence_summary`, `meter_count_combined_summary`, `meter_count_success_rate`, `count_exact_rate`
- lightweight runtime provenance: `generator_shard_id`, `manifest_order`, `preprocess_metadata`

Each row also contains a `candidates` list. In the main `train` table each candidate keeps only the compact score-bearing fields needed for relabeling:
- `candidate_id`, `generation_index`
- `meter_score`, `meter_mean_score`, `meter_logmean_score`
- `meter_target_used`, `meter_target_resolution`
- `meter_num_valid_bayts`, `meter_num_skipped_bayts`
- `count_adherence_score`, `generated_bayts`, `count_requested_bayts`, `count_has_odd_tail`
- `finish_reason`, `token_count`, `cumulative_logprob`

To keep the published table smaller, the main dataset intentionally omits:
- candidate `completion` text
- `per_bayt_meter_scores`
- `per_bayt_meter_details`
- full `generator_metadata`
- full `scoring_metadata`

## Backup artifacts stored in this repo

This repo also stores side artifacts alongside the main dataset:
- `{scored_backup_path}`: compressed raw scored rows backup
- `{assembly_summary_path}`: assembly counts and publish summary
- `artifacts/run_config.json`: run configuration snapshot
- `artifacts/model_meta.json`: resolved model + adapter metadata
- `{difficulty_rule_path}`: difficulty rule used for this release

The compressed scored backup keeps the richer row records that were not copied into the main table, including:
- candidate `completion` text
- `per_bayt_meter_scores`
- `per_bayt_meter_details`
- full `generator_metadata`
- full `scoring_metadata`

## Where the full 8 completions live

Yes: all `{num_candidates}` sampled completions per source row were pushed, but they live in the backup artifact rather than the main `train` table.

Storage layout:
- the main `train` dataset keeps a compact `candidates` list with score-bearing fields only
- the full completion texts are stored in `{scored_backup_path}`
- `{scored_backup_path}` is a gzip-compressed JSONL file
- each JSON line corresponds to one source row
- each row contains `candidate_scores`, which for this release has length `{num_candidates}`
- each entry inside `candidate_scores` includes the candidate `completion` plus the richer scoring details

So if you want to inspect or rescore the exact generated poems later, `{scored_backup_path}` is the file to load.

Current compressed scored backup size:
- **{scored_backup_mb:.1f} MB**

## Scoring design

This preprocessing redesign uses **meter + count only**.

Scorers used:
- **meter**: the BiLSTM meter scorer from this repo, using the requested meter/base-meter context
- **count adherence**: exact bayt-count adherence helper from this repo

Candidate-level scoring notes:
- `meter_score` is the main meter score
- `meter_mean_score` and `meter_logmean_score` are alternate aggregate views exposed per candidate
- `count_adherence_score = 1.0` when generated bayt count exactly matches the requested bayt count
- otherwise `count_adherence_score = max(0, 1 - abs(generated_bayts - requested_bayts) / max(1, requested_bayts))`
- row-level `meter_count_combined_summary` is computed over `0.8 * meter_score + 0.2 * count_adherence_score`

This release does **not** use:
- `meaning_fit`
- `meaning_substance`
- any chosen / selected completion target

## Difficulty labeling rule used in this release

For `K={num_candidates}` candidates per row:
- `strong_meter` means candidate `meter_score >= {float(difficulty_rule.get("strong_meter_threshold", 0.70)):.2f}`
- `strong_exact` means candidate `meter_score >= {float(difficulty_rule.get("strong_meter_threshold", 0.70)):.2f}` **and** exact bayt count
- `easy`: `num_strong_exact >= {int(difficulty_rule.get("easy_min_strong_exact_count", 0))}` and `mean_meter_score > {float(difficulty_rule.get("easy_min_mean_meter", 0.0)):.2f}`
- `hard`: `num_strong_meter == 0` or (`num_strong_meter <= {int(difficulty_rule.get("hard_max_strong_meter_count", 0))}` and `mean_meter_score < {float(difficulty_rule.get("hard_max_mean_meter", 0.0)):.2f}`)
- `medium`: everything else

The `difficulty` label is a convenience label, not a sacred frozen annotation.

## How to relabel rows later with a new rule

You can relabel rows without rerunning generation.

### Option 1: relabel from the main published `train` dataset

Use the compact candidate score records already published in the main dataset:

```python
from datasets import load_dataset

repo_id = "{repo_id}"
ds = load_dataset(repo_id, split="train")

def relabel(row):
    candidates = row["candidates"]
    num_strong_meter = sum(c["meter_score"] >= 0.70 for c in candidates)
    num_strong_exact = sum(
        c["meter_score"] >= 0.70 and c["count_adherence_score"] >= 0.999999
        for c in candidates
    )
    mean_meter = sum(c["meter_score"] for c in candidates) / max(1, len(candidates))

    if num_strong_exact >= {int(difficulty_rule.get("easy_min_strong_exact_count", 0))} and mean_meter > {float(difficulty_rule.get("easy_min_mean_meter", 0.0)):.2f}:
        return "easy"
    if num_strong_meter == 0 or (num_strong_meter <= {int(difficulty_rule.get("hard_max_strong_meter_count", 0))} and mean_meter < {float(difficulty_rule.get("hard_max_mean_meter", 0.0)):.2f}):
        return "hard"
    return "medium"

ds = ds.map(lambda row: {{"difficulty_recomputed": relabel(row)}})
```

Replace the thresholds above with your new rule.

### Option 2: relabel from the full scored backup artifact

If your new rule needs completion text or per-bayt details, read `{scored_backup_path}`:

```python
import gzip
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    filename="{scored_backup_path}",
)

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        # row['candidate_scores'] contains completion text and richer meter details
```

### Option 3: rescore the pushed completions with a new reward

Because the backup artifact stores the full candidate completions, you can compute a brand-new reward later without rerunning generation.

Example sketch:

```python
import gzip
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    repo_type="dataset",
    filename="{scored_backup_path}",
)

def new_reward(row, candidate):
    completion = candidate["completion"]
    description = row["description"]
    enhanced_description = row["enhanced_description"]
    # Replace this with your new reward logic.
    return 0.0

rescored_rows = []
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        new_scores = []
        for candidate in row["candidate_scores"]:
            score = new_reward(row, candidate)
            candidate = dict(candidate)
            candidate["new_reward_score"] = float(score)
            new_scores.append(candidate)
        row["candidate_scores"] = new_scores
        rescored_rows.append(row)
```

Typical follow-up flow:
- download `{scored_backup_path}`
- compute a new per-candidate reward from `completion` and any row context you need
- store the new score beside the existing meter/count scores
- recompute row-level summaries and difficulty labels from the rescored candidate set

This means generation is reusable: if you want to test a new reward later, you can usually start from the pushed completions rather than rerunning the full preprocess.

## Local assembly summary

Machine-readable release summary:

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""


def build_scored_backup(run_dir: Path) -> Path:
    backup_path = run_dir / "assembled" / "scored_rows.jsonl.gz"
    with gzip.open(backup_path, "wt", encoding="utf-8", compresslevel=1) as gz:
        for path in sorted((run_dir / "scored").glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            gz.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return backup_path


def push_dataset(
    run_dir: Path,
    repo_id: str,
    train_jsonl: Path,
    readme_text: str,
    *,
    include_scored_backup: bool,
) -> dict[str, Any]:
    try:
        from datasets import DatasetDict, load_dataset
        from huggingface_hub import HfApi, create_repo
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets and huggingface_hub are required for publishing.") from exc

    token = os.getenv("HF_TOKEN", "").strip() or None
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
    loaded = load_dataset("json", data_files={"train": str(train_jsonl)})
    DatasetDict({"train": loaded["train"]}).push_to_hub(repo_id, token=token)
    readme_path = run_dir / "assembled" / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(readme_path), path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    uploaded = {"dataset_rows": int(len(loaded["train"])), "uploaded_files": ["README.md"]}
    for local_name, repo_name in [
        ("assembly_summary.json", "artifacts/assembly_summary.json"),
        ("run_config.json", "artifacts/run_config.json"),
        ("model_meta.json", "artifacts/model_meta.json"),
        ("difficulty_rule.json", "artifacts/difficulty_rule.json"),
    ]:
        local_path = run_dir / "assembled" / local_name if local_name in {"assembly_summary.json", "difficulty_rule.json"} else run_dir / local_name
        if local_path.exists():
            api.upload_file(path_or_fileobj=str(local_path), path_in_repo=repo_name, repo_id=repo_id, repo_type="dataset")
            uploaded["uploaded_files"].append(repo_name)
    if include_scored_backup:
        backup_path = build_scored_backup(run_dir)
        api.upload_file(
            path_or_fileobj=str(backup_path),
            path_in_repo="artifacts/scored_rows.jsonl.gz",
            repo_id=repo_id,
            repo_type="dataset",
        )
        uploaded["uploaded_files"].append("artifacts/scored_rows.jsonl.gz")
        uploaded["scored_backup_bytes"] = int(backup_path.stat().st_size)
    return uploaded


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)
    repo_id = args.repo_id or str(cfg.get("output_dataset_id") or DEFAULT_OUTPUT_DATASET_ID)
    manifest = load_manifest(run_dir)
    assembled_rows = []
    missing = []
    duplicates = Counter()
    for row in sorted(manifest, key=lambda item: int(item["manifest_order"])):
        uid = str(row["row_uid"])
        duplicates[uid] += 1
        scored = read_json_or_none(scored_path(run_dir, uid))
        if scored is None:
            missing.append(uid)
            continue
        assembled_rows.append(derived_row(scored, include_completions=bool(args.include_completions)))
        if args.max_rows > 0 and len(assembled_rows) >= args.max_rows:
            break

    duplicate_uids = sorted(uid for uid, count in duplicates.items() if count > 1)
    if duplicate_uids:
        raise RuntimeError(f"Duplicate row_uid values in manifest: {duplicate_uids[:10]}")
    if missing and not args.allow_incomplete and args.max_rows <= 0:
        raise RuntimeError(f"Missing scored rows: {len(missing)}. First missing: {missing[:10]}")

    train_jsonl = run_dir / "assembled" / "train.jsonl"
    write_jsonl(train_jsonl, assembled_rows)

    difficulty = Counter(row["difficulty"] for row in assembled_rows)
    meters = Counter(row["base_meter"] for row in assembled_rows)
    candidate_counts = [int(row["num_candidates_scored"]) for row in assembled_rows]
    summary = {
        "assembled_at": utc_now_iso(),
        "repo_id": repo_id,
        "run_dir": str(run_dir),
        "manifest_rows": len(manifest),
        "assembled_rows": len(assembled_rows),
        "missing_scored_rows": len(missing),
        "train_jsonl": str(train_jsonl),
        "difficulty_counts": dict(sorted(difficulty.items())),
        "base_meter_counts": dict(sorted(meters.items())),
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "candidate_count_sum": sum(candidate_counts),
        "dry_run": bool(args.dry_run),
        "push": bool(args.push),
        "include_completions": bool(args.include_completions),
        "scored_backup_requested": bool(not args.skip_scored_backup),
    }
    atomic_write_json(run_dir / "assembled" / "assembly_summary.json", summary)
    difficulty_rule = next((row.get("difficulty_rule") for row in assembled_rows if row.get("difficulty_rule")), {})
    atomic_write_json(run_dir / "assembled" / "difficulty_rule.json", difficulty_rule)
    readme_text = build_readme(repo_id, cfg, summary)
    (run_dir / "assembled" / "README.md").write_text(readme_text, encoding="utf-8")

    if args.push:
        uploaded = push_dataset(
            run_dir,
            repo_id,
            train_jsonl,
            readme_text,
            include_scored_backup=bool(not args.skip_scored_backup),
        )
        summary.update(uploaded)
        atomic_write_json(run_dir / "assembled" / "assembly_summary.json", summary)
    elif args.dry_run:
        try:
            from datasets import load_dataset

            loaded = load_dataset("json", data_files={"train": str(train_jsonl)})
            summary["dry_run_loaded_rows"] = int(len(loaded["train"]))
            if not args.skip_scored_backup:
                backup_path = build_scored_backup(run_dir)
                summary["scored_backup_bytes"] = int(backup_path.stat().st_size)
            atomic_write_json(run_dir / "assembled" / "assembly_summary.json", summary)
        except Exception as exc:
            raise RuntimeError(f"Dry-run dataset load failed: {type(exc).__name__}: {exc}") from exc

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
