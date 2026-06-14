#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

from baseline_runtime import build_generator
from common import (
    OUTPUTS_ROOT,
    append_jsonl,
    atomic_write_json,
    env_snapshot,
    health_check,
    load_completed_keys,
    load_env,
    read_jsonl,
    setup_logger,
    utc_now_iso,
)
from model_registry import get_model_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Ashaar outputs using Ashaar-native control prompts.")
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--upload-every-rows", type=int, default=0)
    parser.add_argument("--upload-repo-id", default="")
    parser.add_argument("--upload-token-env-var", default="HF_TOKEN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    spec = get_model_spec("ashaar_model")
    run_dir = resolve_run_dir(args.run_dir)
    logger = setup_logger(run_dir, "generate_ashaar_native_baselines")
    output_path = run_dir / "ashaar_native_generations.jsonl"
    failure_path = run_dir / "ashaar_native_failures.jsonl"
    status_path = run_dir / "ashaar_native_status.json"

    rows = read_jsonl(Path(args.manifest_jsonl))
    if args.limit_rows:
        rows = rows[: int(args.limit_rows)]
    expected = len(rows) * int(args.samples_per_row)
    done_keys = load_completed_keys(output_path, ("model_name", "input_row_id", "sample_index"))

    atomic_write_json(run_dir / "env_snapshot.json", env_snapshot({"script": "generate_ashaar_native_baselines.py"}))
    atomic_write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "model_name": spec.name,
            "model_id": spec.model_id,
            "started_at_utc": utc_now_iso(),
            "expected_rows": expected,
        },
    )
    logger.info("run_dir=%s rows=%d expected=%d", run_dir, len(rows), expected)

    pending_tasks = build_pending_tasks(rows, args, done_keys, failure_path)
    logger.info("loading_model name=%s model_id=%s pending=%d", spec.name, spec.model_id, len(pending_tasks))
    started = time.time()
    produced = len(done_keys)
    if pending_tasks:
        generator = build_generator(spec, args, logger)
        try:
            for batch in chunked(pending_tasks, max(1, int(args.batch_size))):
                try:
                    raw_texts = generate_batch_with_oom_retry(generator, [task["prompt"] for task in batch], logger)
                except Exception as exc:
                    for task in batch:
                        append_generation_failure(failure_path, task, exc)
                    write_status(status_path, produced, expected, started)
                    continue
                for task, raw_text in zip(batch, raw_texts):
                    text = str(raw_text or "").strip()
                    health_status, health_reason = health_check(text)
                    generation_status = "ok" if text else "empty"
                    payload = make_payload(task["row"], args, task["input_row_id"], task["sample_index"], text, generation_status, health_status, health_reason)
                    if generation_status == "ok" and health_status == "ok":
                        append_jsonl(output_path, payload)
                        done_keys.add(task["key"])
                        produced += 1
                    else:
                        append_jsonl(failure_path, payload)
                    write_status(status_path, produced, expected, started)
                    maybe_upload_checkpoint(args, output_path, status_path, produced, expected, logger)
        finally:
            del generator
            clear_cuda_cache()

    write_status(status_path, produced, expected, started, status="completed")
    maybe_upload_checkpoint(args, output_path, status_path, produced, expected, logger, force=True)
    logger.info("generation_complete produced=%d expected=%d", produced, expected)
    return 0


def build_pending_tasks(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    done_keys: set[tuple[Any, ...]],
    failure_path: Path,
) -> list[dict[str, Any]]:
    tasks = []
    for row in rows:
        prompt = str(row.get("ashaar_native_prompt") or row.get("ashaar_controls_prompt") or "").strip()
        input_row_id = str(row.get("input_row_id") or row.get("manifest_row_id") or "").strip()
        if not prompt:
            append_jsonl(
                failure_path,
                {
                    "model_name": "ashaar_model",
                    "input_row_id": input_row_id,
                    "source_row_index": int(row.get("source_row_index") or 0),
                    "error": "missing ashaar_native_prompt",
                    "timestamp_utc": utc_now_iso(),
                },
            )
            continue
        for sample_index in range(int(args.samples_per_row)):
            key = ("ashaar_model", input_row_id, sample_index)
            if key in done_keys:
                continue
            tasks.append(
                {
                    "row": row,
                    "prompt": prompt,
                    "input_row_id": input_row_id,
                    "sample_index": sample_index,
                    "key": key,
                }
            )
    return tasks


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


def append_generation_failure(failure_path: Path, task: dict[str, Any], exc: Exception) -> None:
    row = task["row"]
    append_jsonl(
        failure_path,
        {
            "model_name": "ashaar_model",
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
        path = OUTPUTS_ROOT / f"ashaar_native_baseline_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_payload(
    row: dict[str, Any],
    args: argparse.Namespace,
    input_row_id: str,
    sample_index: int,
    text: str,
    generation_status: str,
    health_status: str,
    health_reason: str,
) -> dict[str, Any]:
    return {
        "generation_id": f"ashaar_native_{input_row_id}_sample_{sample_index}",
        "input_row_id": input_row_id,
        "manifest_row_id": str(row.get("manifest_row_id") or input_row_id),
        "source_row_index": int(row.get("source_row_index") or 0),
        "source_id": str(row.get("source_id") or ""),
        "sample_index": int(sample_index),
        "model_name": "ashaar_model",
        "model_display_name": "Ashaar",
        "model_id": "arbml/Ashaar_model",
        "model_group": "native_controls",
        "model_role": "Poetry generation with Ashaar-native meter/qafiyah/theme controls",
        "paper_source": "Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches",
        "base_meter": str(row.get("base_meter") or ""),
        "form": str(row.get("form") or ""),
        "meter_label": str(row.get("meter_label") or ""),
        "requested_bayts": int(row.get("requested_bayts") or 0),
        "requested_num_lines": int(row.get("requested_num_lines") or row.get("sft_num_lines") or 0),
        "poem_theme": str(row.get("poem_theme") or ""),
        "poem_meter": str(row.get("poem_meter") or ""),
        "description": str(row.get("description") or ""),
        "enhanced_description": str(row.get("enhanced_description") or ""),
        "reference_completion": str(row.get("reference_completion") or ""),
        "reference_prefix_text": str(row.get("reference_prefix_text") or ""),
        "ashaar_prompt_mode": str(row.get("ashaar_prompt_mode") or ""),
        "ashaar_prefix_mode": str(row.get("ashaar_prefix_mode") or ""),
        "ashaar_meter_token": str(row.get("ashaar_meter_token") or ""),
        "ashaar_qafiyah": str(row.get("ashaar_qafiyah") or ""),
        "ashaar_theme": str(row.get("ashaar_theme") or ""),
        "ashaar_theme_token": str(row.get("ashaar_theme_token") or ""),
        "prompt_or_prefix": str(row.get("ashaar_native_prompt") or ""),
        "raw_generated_text": text,
        "generated_text": text,
        "generation_status": generation_status,
        "health_status": health_status,
        "health_reason": health_reason,
        "decode_config": json.dumps(
            {
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "batch_size": int(args.batch_size),
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


_LAST_UPLOADED_COUNT = -1
_UPLOAD_TOKEN_WARNING_EMITTED = False


def maybe_upload_checkpoint(
    args: argparse.Namespace,
    output_path: Path,
    status_path: Path,
    produced: int,
    expected: int,
    logger: Any,
    force: bool = False,
) -> None:
    global _LAST_UPLOADED_COUNT, _UPLOAD_TOKEN_WARNING_EMITTED
    every = int(getattr(args, "upload_every_rows", 0) or 0)
    repo_id = str(getattr(args, "upload_repo_id", "") or "").strip()
    if not repo_id or every <= 0 or produced <= 0:
        return
    if not force and produced % every != 0:
        return
    if produced == _LAST_UPLOADED_COUNT:
        return
    token = os.getenv(str(getattr(args, "upload_token_env_var", "HF_TOKEN") or "HF_TOKEN")) or ""
    if not token:
        if not _UPLOAD_TOKEN_WARNING_EMITTED:
            logger.warning("periodic_upload_skipped missing_token_env=%s", getattr(args, "upload_token_env_var", "HF_TOKEN"))
            _UPLOAD_TOKEN_WARNING_EMITTED = True
        _LAST_UPLOADED_COUNT = produced
        return
    if not output_path.exists():
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo="data/test.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Checkpoint Ashaar native generations at {produced}/{expected} rows",
        )
        if status_path.exists():
            api.upload_file(
                path_or_fileobj=str(status_path),
                path_in_repo="artifacts/ashaar_native_status.json",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Update Ashaar native generation status at {produced}/{expected} rows",
            )
        _LAST_UPLOADED_COUNT = produced
        logger.info("periodic_upload_complete repo_id=%s rows=%d expected=%d", repo_id, produced, expected)
    except Exception as exc:
        logger.warning("periodic_upload_failed repo_id=%s rows=%d error=%s: %s", repo_id, produced, type(exc).__name__, exc)


if __name__ == "__main__":
    raise SystemExit(main())
