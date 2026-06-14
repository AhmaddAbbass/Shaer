#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from baseline_runtime import build_generator
from common import (
    OUTPUTS_ROOT,
    append_jsonl,
    atomic_write_json,
    baseline_input_row_id,
    env_snapshot,
    health_check,
    load_completed_keys,
    load_env,
    read_jsonl,
    setup_logger,
    strip_prefix_if_present,
    utc_now_iso,
)
from model_registry import parse_model_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate continuation-baseline outputs on the selected Shaer subset.")
    parser.add_argument("--subset-jsonl", required=True)
    parser.add_argument("--models", default="ashaar_model,gpt2_small_arabic_poetry,gpt2_medium_arabic_poetry")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gguf-filename", default="")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    model_specs = [spec for spec in parse_model_list(args.models) if spec.group == "continuation"]
    if not model_specs:
        raise RuntimeError("No continuation-generation models were selected.")
    run_dir = resolve_run_dir(args.run_dir)
    logger = setup_logger(run_dir, "generate_continuation_baselines")
    output_path = run_dir / "continuation_generations.jsonl"
    failure_path = run_dir / "continuation_failures.jsonl"
    status_path = run_dir / "continuation_status.json"

    rows = read_jsonl(Path(args.subset_jsonl))
    if args.limit_rows:
        rows = rows[: int(args.limit_rows)]
    expected = len(rows) * len(model_specs) * int(args.samples_per_row)
    done_keys = load_completed_keys(output_path, ("model_name", "input_row_id", "sample_index"))

    atomic_write_json(run_dir / "env_snapshot.json", env_snapshot({"script": "generate_continuation_baselines.py"}))
    atomic_write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "selected_models": [spec.name for spec in model_specs],
            "started_at_utc": utc_now_iso(),
            "expected_rows": expected,
        },
    )
    logger.info("run_dir=%s rows=%d models=%d expected=%d", run_dir, len(rows), len(model_specs), expected)

    started = time.time()
    produced = len(done_keys)
    batch_size = max(1, int(args.batch_size))
    for spec in model_specs:
        pending_tasks = build_pending_tasks(rows, spec, args, done_keys, failure_path)
        logger.info(
            "loading_model name=%s model_id=%s pending=%d batch_size=%d",
            spec.name,
            spec.model_id,
            len(pending_tasks),
            batch_size,
        )
        if not pending_tasks:
            continue
        generator = build_generator(spec, args, logger)
        try:
            for batch in chunked(pending_tasks, batch_size):
                try:
                    raw_texts = generate_batch_with_oom_retry(generator, [task["prefix"] for task in batch], logger)
                except Exception as exc:
                    for task in batch:
                        append_generation_failure(failure_path, spec, task, exc)
                    write_status(status_path, produced, expected, started)
                    continue
                for task, raw_text in zip(batch, raw_texts):
                    stripped_text, stripped_prefix = strip_prefix_if_present(task["prefix"], raw_text)
                    health_status, health_reason = health_check(stripped_text)
                    generation_status = "ok" if stripped_text else "empty"
                    payload = make_payload(
                        task["row"],
                        spec,
                        args,
                        task["input_row_id"],
                        task["sample_index"],
                        task["prefix"],
                        raw_text,
                        stripped_text,
                        stripped_prefix,
                        generation_status,
                        health_status,
                        health_reason,
                    )
                    if generation_status == "ok" and health_status == "ok":
                        append_jsonl(output_path, payload)
                        done_keys.add(task["key"])
                        produced += 1
                    else:
                        append_jsonl(failure_path, payload)
                    write_status(status_path, produced, expected, started)
        finally:
            del generator
            clear_cuda_cache()

    write_status(status_path, produced, expected, started, status="completed")
    logger.info("generation_complete produced=%d expected=%d", produced, expected)
    return 0


def build_pending_tasks(
    rows: list[dict[str, Any]],
    spec: Any,
    args: Any,
    done_keys: set[tuple[Any, ...]],
    failure_path: Path,
) -> list[dict[str, Any]]:
    pending_tasks: list[dict[str, Any]] = []
    for row in rows:
        prefix = str(row.get("reference_prefix_text") or "").strip()
        if not prefix:
            append_jsonl(
                failure_path,
                {
                    "model_name": spec.name,
                    "source_row_index": int(row.get("source_row_index") or 0),
                    "error": "missing reference_prefix_text",
                    "timestamp_utc": utc_now_iso(),
                },
            )
            continue
        input_row_id = baseline_input_row_id(row)
        for sample_index in range(int(args.samples_per_row)):
            key = (spec.name, input_row_id, sample_index)
            if key in done_keys:
                continue
            pending_tasks.append(
                {
                    "row": row,
                    "prefix": prefix,
                    "input_row_id": input_row_id,
                    "sample_index": sample_index,
                    "key": key,
                }
            )
    return pending_tasks


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def generate_batch_with_oom_retry(generator: Any, prompts: list[str], logger: Any) -> list[str]:
    try:
        method = getattr(generator, "generate_batch", None)
        if callable(method):
            texts = method(prompts)
        else:
            texts = [generator.generate(prompt) for prompt in prompts]
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            clear_cuda_cache()
            if len(prompts) > 1:
                logger.warning("cuda_oom batch_size=%d retrying_as_singletons", len(prompts))
                out: list[str] = []
                for prompt in prompts:
                    out.extend(generate_batch_with_oom_retry(generator, [prompt], logger))
                return out
        raise
    if len(texts) != len(prompts):
        raise RuntimeError(f"Batch generation returned {len(texts)} texts for {len(prompts)} prompts.")
    return [str(text or "").strip() for text in texts]


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "cublas" in message)


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def append_generation_failure(failure_path: Path, spec: Any, task: dict[str, Any], exc: Exception) -> None:
    row = task["row"]
    append_jsonl(
        failure_path,
        {
            "model_name": spec.name,
            "source_row_index": int(row.get("source_row_index") or 0),
            "input_row_id": task.get("input_row_id", ""),
            "sample_index": int(task.get("sample_index") or 0),
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp_utc": utc_now_iso(),
        },
    )


def resolve_run_dir(run_dir: str) -> Path:
    if run_dir:
        path = Path(run_dir)
    else:
        timestamp = utc_now_iso().replace(":", "").replace("-", "")
        path = OUTPUTS_ROOT / f"continuation_baselines_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_payload(
    row: dict[str, Any],
    spec: Any,
    args: Any,
    input_row_id: str,
    sample_index: int,
    prefix: str,
    raw_text: str,
    stripped_text: str,
    stripped_prefix: bool,
    generation_status: str,
    health_status: str,
    health_reason: str,
) -> dict[str, Any]:
    requested_num_lines = int(row.get("continuation_requested_num_lines") or 0)
    return {
        "generation_id": f"{spec.name}_{input_row_id}_sample_{sample_index}",
        "input_row_id": input_row_id,
        "manifest_row_id": str(row.get("manifest_row_id") or input_row_id),
        "subset_rank": int(row.get("subset_rank") or 0),
        "selected_shaer_generation_id": str(row.get("selected_shaer_generation_id") or ""),
        "source_row_index": int(row.get("source_row_index") or 0),
        "source_id": str(row.get("source_id") or ""),
        "shaer_source_sample_index": int(row.get("shaer_source_sample_index") or row.get("sample_index") or 0),
        "sample_index": int(sample_index),
        "model_name": spec.name,
        "model_display_name": spec.display_name,
        "model_id": spec.model_id,
        "model_group": spec.group,
        "model_role": spec.role,
        "paper_source": spec.paper_source,
        "base_meter": str(row.get("base_meter") or ""),
        "form": str(row.get("form") or ""),
        "meter_label": str(row.get("meter_label") or ""),
        "requested_bayts": int(row.get("continuation_requested_bayts") or 0),
        "requested_num_lines": requested_num_lines,
        "continuation_requested_num_lines": requested_num_lines,
        "reference_prefix_text": prefix,
        "reference_remaining_completion": str(row.get("reference_remaining_completion") or ""),
        "reference_completion": str(row.get("reference_completion") or ""),
        "prompt_or_prefix": prefix,
        "raw_generated_text": raw_text,
        "generated_text": stripped_text,
        "prefix_was_returned": bool(stripped_prefix),
        "generation_status": generation_status,
        "health_status": health_status,
        "health_reason": health_reason,
        "decode_config": json.dumps(
            {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "batch_size": int(getattr(args, "batch_size", 1)),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "created_at_utc": utc_now_iso(),
    }


def write_status(path: Path, produced: int, expected: int, started: float, status: str = "running") -> None:
    elapsed = max(0.001, time.time() - started)
    atomic_write_json(
        path,
        {
            "status": status,
            "generated_rows": int(produced),
            "expected_rows": int(expected),
            "remaining_rows": max(0, int(expected) - int(produced)),
            "rows_per_second": float(produced / elapsed),
            "elapsed_seconds": elapsed,
            "updated_at_utc": utc_now_iso(),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
