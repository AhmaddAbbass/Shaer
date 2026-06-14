#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    build_fanar_metadata_prompt,
    build_plain_arabic_instruction_prompt,
    env_snapshot,
    health_check,
    load_completed_keys,
    load_env,
    read_jsonl,
    setup_logger,
    utc_now_iso,
)
from model_registry import parse_model_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instruction-baseline outputs on the selected Shaer subset.")
    parser.add_argument("--subset-jsonl", required=True)
    parser.add_argument("--models", default="yehia_base,qwen3_poetry_gguf")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--gguf-filename", default="")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    model_specs = [spec for spec in parse_model_list(args.models) if spec.group == "instruction"]
    if not model_specs:
        raise RuntimeError("No instruction-generation models were selected.")
    run_dir = resolve_run_dir(args.run_dir)
    logger = setup_logger(run_dir, "generate_instruction_baselines")
    output_path = run_dir / "instruction_generations.jsonl"
    failure_path = run_dir / "instruction_failures.jsonl"
    status_path = run_dir / "instruction_status.json"

    rows = read_jsonl(Path(args.subset_jsonl))
    if args.limit_rows:
        rows = rows[: int(args.limit_rows)]
    expected = len(rows) * len(model_specs) * int(args.samples_per_row)
    done_keys = load_completed_keys(output_path, ("model_name", "input_row_id", "sample_index"))

    atomic_write_json(run_dir / "env_snapshot.json", env_snapshot({"script": "generate_instruction_baselines.py"}))
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
    for spec in model_specs:
        logger.info("loading_model name=%s model_id=%s", spec.name, spec.model_id)
        generator = build_generator(spec, args, logger)
        try:
            for row in rows:
                prompt, prompt_error = build_model_prompt(spec, row)
                raw_sft_prompt = str(row.get("sft_prompt") or "").strip()
                if not prompt:
                    append_jsonl(
                        failure_path,
                        {
                            "model_name": spec.name,
                            "source_row_index": int(row.get("source_row_index") or 0),
                            "error": prompt_error or "missing prompt",
                            "timestamp_utc": utc_now_iso(),
                        },
                    )
                    continue
                for sample_index in range(int(args.samples_per_row)):
                    input_row_id = baseline_input_row_id(row)
                    key = (spec.name, input_row_id, sample_index)
                    if key in done_keys:
                        continue
                    try:
                        text = generator.generate(prompt)
                    except Exception as exc:
                        append_jsonl(
                            failure_path,
                            {
                                "model_name": spec.name,
                                "source_row_index": int(row.get("source_row_index") or 0),
                                "sample_index": sample_index,
                                "error": f"{type(exc).__name__}: {exc}",
                                "timestamp_utc": utc_now_iso(),
                            },
                        )
                        continue
                    health_status, health_reason = health_check(text)
                    generation_status = "ok" if text else "empty"
                    payload = make_payload(
                        row,
                        spec,
                        args,
                        input_row_id,
                        sample_index,
                        prompt,
                        raw_sft_prompt,
                        text,
                        generation_status,
                        health_status,
                        health_reason,
                    )
                    if generation_status == "ok" and health_status == "ok":
                        append_jsonl(output_path, payload)
                        done_keys.add(key)
                        produced += 1
                    else:
                        append_jsonl(failure_path, payload)
                    write_status(status_path, produced, expected, started)
        finally:
            del generator

    write_status(status_path, produced, expected, started, status="completed")
    logger.info("generation_complete produced=%d expected=%d", produced, expected)
    return 0


def resolve_run_dir(run_dir: str) -> Path:
    if run_dir:
        path = Path(run_dir)
    else:
        timestamp = utc_now_iso().replace(":", "").replace("-", "")
        path = OUTPUTS_ROOT / f"instruction_baselines_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_payload(
    row: dict[str, Any],
    spec: Any,
    args: Any,
    input_row_id: str,
    sample_index: int,
    prompt: str,
    raw_sft_prompt: str,
    text: str,
    generation_status: str,
    health_status: str,
    health_reason: str,
) -> dict[str, Any]:
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
        "requested_bayts": int(row.get("requested_bayts") or 0),
        "requested_num_lines": int(row.get("sft_num_lines") or row.get("requested_num_lines") or 0),
        "sft_num_lines": int(row.get("sft_num_lines") or 0),
        "description": str(row.get("description") or ""),
        "enhanced_description": str(row.get("enhanced_description") or ""),
        "prompt_or_prefix": prompt,
        "raw_sft_prompt": raw_sft_prompt,
        "reference_completion": str(row.get("reference_completion") or ""),
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
                "use_chat_template": bool(getattr(args, "use_chat_template", False)),
                "prompt_style": str(getattr(spec, "default_prompt_mode", "") or "plain_instruction"),
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


def build_model_prompt(spec: Any, row: dict[str, Any]) -> tuple[str, str]:
    prompt_mode = str(getattr(spec, "default_prompt_mode", "") or "plain_instruction")
    if prompt_mode == "fanar_metadata":
        return build_fanar_metadata_prompt(row)
    return build_plain_arabic_instruction_prompt(row), ""


if __name__ == "__main__":
    raise SystemExit(main())
