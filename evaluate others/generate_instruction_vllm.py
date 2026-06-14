#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from common import (
    OUTPUTS_ROOT,
    append_jsonl,
    atomic_write_json,
    baseline_input_row_id,
    build_plain_arabic_instruction_prompt,
    env_snapshot,
    health_check,
    load_completed_keys,
    load_env,
    read_jsonl,
    setup_logger,
    utc_now_iso,
)

DEFAULT_MODEL_ID = "Navid-AI/Yehia-7B-preview"
VARIANT_MODEL_NAMES = {
    "plain": "yehia_base_plain_instruction",
    "chat": "yehia_base_chat_template",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Yehia instruction baselines with vLLM.")
    parser.add_argument("--subset-jsonl", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--prompt-variant", choices=("plain", "chat"), default="plain")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-chat-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    model_name = args.model_name or VARIANT_MODEL_NAMES[args.prompt_variant]
    run_dir = resolve_run_dir(args.run_dir, model_name)
    logger = setup_logger(run_dir, "generate_instruction_vllm")
    output_path = run_dir / "instruction_generations.jsonl"
    failure_path = run_dir / "instruction_failures.jsonl"
    status_path = run_dir / "instruction_status.json"

    rows = read_jsonl(Path(args.subset_jsonl))
    if args.limit_rows:
        rows = rows[: int(args.limit_rows)]
    expected = len(rows) * int(args.samples_per_row)
    done_keys = load_completed_keys(output_path, ("model_name", "input_row_id", "sample_index"))

    atomic_write_json(run_dir / "env_snapshot.json", env_snapshot({"script": "generate_instruction_vllm.py"}))
    atomic_write_json(
        run_dir / "run_config.json",
        {
            **vars(args),
            "model_name": model_name,
            "started_at_utc": utc_now_iso(),
            "expected_rows": expected,
        },
    )
    logger.info("run_dir=%s model=%s rows=%d expected=%d", run_dir, model_name, len(rows), expected)

    tokenizer = load_tokenizer(args)
    prompts: list[str] = []
    prompt_rows: list[dict[str, Any]] = []
    prompt_errors = 0
    for row in rows:
        input_row_id = baseline_input_row_id(row)
        missing = [s for s in range(int(args.samples_per_row)) if (model_name, input_row_id, s) not in done_keys]
        if not missing:
            continue
        try:
            prompt = build_prompt(row, tokenizer, args)
        except Exception as exc:
            prompt_errors += 1
            append_jsonl(
                failure_path,
                {
                    "model_name": model_name,
                    "input_row_id": input_row_id,
                    "source_row_index": int(row.get("source_row_index") or 0),
                    "error": f"prompt_error:{type(exc).__name__}: {exc}",
                    "timestamp_utc": utc_now_iso(),
                },
            )
            continue
        prompts.append(prompt)
        prompt_rows.append(row)

    if prompt_errors:
        logger.warning("prompt_errors=%d", prompt_errors)
    if not prompts:
        write_status(status_path, len(done_keys), expected, time.time(), status="completed")
        logger.info("nothing_pending produced=%d expected=%d", len(done_keys), expected)
        return 0

    llm, sampling_params = build_vllm(args, logger)
    started = time.time()
    produced = len(done_keys)

    try:
        for offset in range(0, len(prompts), int(args.batch_size)):
            batch_prompts = prompts[offset : offset + int(args.batch_size)]
            batch_rows = prompt_rows[offset : offset + int(args.batch_size)]
            request_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for row, request_output, prompt in zip(batch_rows, request_outputs, batch_prompts):
                input_row_id = baseline_input_row_id(row)
                for sample_index, completion in enumerate(request_output.outputs):
                    key = (model_name, input_row_id, sample_index)
                    if key in done_keys:
                        continue
                    text = str(completion.text or "").strip()
                    generation_status = "ok" if text else "empty"
                    health_status, health_reason = health_check(text)
                    payload = make_payload(row, args, model_name, input_row_id, sample_index, prompt, text, generation_status, health_status, health_reason)
                    if generation_status == "ok" and health_status == "ok":
                        append_jsonl(output_path, payload)
                        done_keys.add(key)
                        produced += 1
                    else:
                        append_jsonl(failure_path, payload)
                write_status(status_path, produced, expected, started)
            logger.info("progress produced=%d expected=%d", produced, expected)
    finally:
        del llm

    final_status = "completed" if produced == expected else "incomplete"
    write_status(status_path, produced, expected, started, status=final_status)
    logger.info("generation_complete status=%s produced=%d expected=%d", final_status, produced, expected)
    return 0 if final_status == "completed" else 2


def resolve_run_dir(run_dir: str, model_name: str) -> Path:
    if run_dir:
        path = Path(run_dir)
    else:
        timestamp = utc_now_iso().replace(":", "").replace("-", "")
        path = OUTPUTS_ROOT / f"instruction_vllm_{timestamp}" / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        args.model_id,
        token=os.getenv("HF_TOKEN") or None,
        trust_remote_code=bool(args.trust_remote_code),
    )


def build_vllm(args: argparse.Namespace, logger: Any):
    from vllm import LLM, SamplingParams

    logger.info("loading_vllm model_id=%s", args.model_id)
    llm = LLM(
        model=args.model_id,
        tokenizer=args.model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype=str(args.dtype),
        max_model_len=int(args.max_model_len),
        trust_remote_code=bool(args.trust_remote_code),
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        n=int(args.samples_per_row),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        repetition_penalty=float(args.repetition_penalty),
        max_tokens=int(args.max_new_tokens),
        detokenize=True,
        skip_special_tokens=True,
    )
    return llm, sampling_params


def build_prompt(row: dict[str, Any], tokenizer: Any, args: argparse.Namespace) -> str:
    plain = build_plain_arabic_instruction_prompt(row)
    if args.prompt_variant == "plain":
        return plain
    messages = [{"role": "user", "content": plain}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        if args.allow_chat_fallback:
            return plain
        raise


def make_payload(
    row: dict[str, Any],
    args: argparse.Namespace,
    model_name: str,
    input_row_id: str,
    sample_index: int,
    prompt: str,
    text: str,
    generation_status: str,
    health_status: str,
    health_reason: str,
) -> dict[str, Any]:
    return {
        "generation_id": f"{model_name}_{input_row_id}_sample_{sample_index}",
        "input_row_id": input_row_id,
        "manifest_row_id": str(row.get("manifest_row_id") or input_row_id),
        "subset_rank": int(row.get("subset_rank") or 0),
        "selected_shaer_generation_id": str(row.get("selected_shaer_generation_id") or ""),
        "source_row_index": int(row.get("source_row_index") or 0),
        "source_id": str(row.get("source_id") or ""),
        "shaer_source_sample_index": int(row.get("shaer_source_sample_index") or row.get("sample_index") or 0),
        "sample_index": int(sample_index),
        "model_name": model_name,
        "model_display_name": "Original Yehia",
        "model_id": str(args.model_id),
        "model_group": "instruction",
        "model_role": "Your baseline/original model",
        "paper_source": "Model card for Navid-AI/Yehia-7B-preview",
        "prompt_variant": str(args.prompt_variant),
        "base_meter": str(row.get("base_meter") or ""),
        "form": str(row.get("form") or ""),
        "meter_label": str(row.get("meter_label") or ""),
        "requested_bayts": int(row.get("requested_bayts") or 0),
        "requested_num_lines": int(row.get("sft_num_lines") or row.get("requested_num_lines") or 0),
        "sft_num_lines": int(row.get("sft_num_lines") or 0),
        "description": str(row.get("description") or ""),
        "enhanced_description": str(row.get("enhanced_description") or ""),
        "prompt_or_prefix": prompt,
        "raw_sft_prompt": str(row.get("sft_prompt") or "").strip(),
        "reference_completion": str(row.get("reference_completion") or ""),
        "generated_text": text,
        "generation_status": generation_status,
        "health_status": health_status,
        "health_reason": health_reason,
        "decode_config": json.dumps(
            {
                "engine": "vllm",
                "max_new_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "repetition_penalty": float(args.repetition_penalty),
                "prompt_variant": str(args.prompt_variant),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "max_model_len": int(args.max_model_len),
                "dtype": str(args.dtype),
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
