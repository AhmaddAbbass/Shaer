#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_SFT_ADAPTER_MODE,
    DEFAULT_SFT_ADAPTER_REPO,
    append_jsonl,
    atomic_write_json,
    ensure_dir,
    event_log_path,
    failure_log_path,
    generated_path,
    generator_done_path,
    load_dotenv_if_present,
    load_run_config,
    load_shard_manifest,
    read_json_or_none,
    status_path,
    utc_now_iso,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate candidate poems for one meter/count preprocess shard.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--generator-id", default="")
    parser.add_argument("--num-candidates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=1664)
    parser.add_argument("--stop-after-generated-rows", type=int, default=0)
    parser.add_argument("--mock-generate", action="store_true", help="For plumbing tests only; does not load the model.")
    return parser.parse_args()


def build_vllm_engine(args: argparse.Namespace, cfg: dict[str, Any]):
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("vLLM is required for real generation. Install grpo/requirements.txt.") from exc

    from rewards.common import resolve_sft_adapter_path

    base_model_id = (
        os.getenv("GRPO_MC_BASE_MODEL_ID", "").strip()
        or str(cfg.get("base_model_id") or "")
        or os.getenv("BASE_MODEL_ID", "").strip()
        or DEFAULT_BASE_MODEL_ID
    )
    adapter_repo = (
        os.getenv("GRPO_MC_SFT_ADAPTER_REPO", "").strip()
        or str(cfg.get("sft_adapter_repo") or "")
        or DEFAULT_SFT_ADAPTER_REPO
    )
    adapter_mode = (
        os.getenv("GRPO_MC_SFT_ADAPTER_MODE", "").strip()
        or str(cfg.get("sft_adapter_mode") or "")
        or DEFAULT_SFT_ADAPTER_MODE
    )
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    adapter_path, adapter_meta = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token, mode=adapter_mode)
    llm = LLM(
        model=base_model_id,
        tokenizer=base_model_id,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        dtype="bfloat16",
        max_model_len=int(args.max_model_len),
        trust_remote_code=True,
        disable_log_stats=True,
    )
    lora_request = LoRARequest(
        lora_name=Path(adapter_path).name or "shaer_sft_adapter",
        lora_int_id=1,
        lora_path=adapter_path,
        base_model_name=base_model_id,
    )
    sampling_cls = SamplingParams
    meta = {
        "base_model_id": base_model_id,
        "adapter_repo": adapter_repo,
        "adapter_mode": adapter_mode,
        "adapter_path": adapter_path,
        **adapter_meta,
    }
    return llm, lora_request, sampling_cls, meta


def mock_completions(row: dict[str, Any], k: int) -> list[dict[str, Any]]:
    completion = str(row.get("sft_completion") or "").strip()
    if not completion:
        completion = "بيت تجريبي على السطر الأول\nبيت تجريبي على السطر الثاني"
    return [
        {
            "candidate_id": f"{row['row_uid']}__cand_{idx:02d}",
            "completion": completion,
            "generation_index": idx,
            "finish_reason": "mock",
            "token_count": 0,
            "cumulative_logprob": None,
        }
        for idx in range(k)
    ]


def generate_batch_vllm(
    llm,
    lora_request,
    sampling_cls,
    rows: list[dict[str, Any]],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[list[dict[str, Any]]]:
    sampling_params = sampling_cls(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        detokenize=True,
        skip_special_tokens=True,
    )
    prompts = [str(row["sft_prompt"]) for row in rows]
    outputs = llm.generate(prompts, sampling_params, lora_request=[lora_request] * len(prompts), use_tqdm=False)
    out: list[list[dict[str, Any]]] = []
    for row, request_output in zip(rows, outputs):
        candidate_rows = []
        for idx, completion in enumerate(request_output.outputs):
            candidate_rows.append(
                {
                    "candidate_id": f"{row['row_uid']}__cand_{idx:02d}",
                    "completion": completion.text.strip(),
                    "generation_index": idx,
                    "finish_reason": str(getattr(completion, "finish_reason", "") or ""),
                    "token_count": len(getattr(completion, "token_ids", []) or []),
                    "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
                }
            )
        out.append(candidate_rows)
    return out


def write_generated_row(run_dir: Path, row: dict[str, Any], candidates: list[dict[str, Any]], generator_meta: dict[str, Any]) -> None:
    payload = {
        "row_uid": row["row_uid"],
        "source_split": row["source_split"],
        "source_index": int(row["source_index"]),
        "source_row_index_in_split": int(row["source_row_index_in_split"]),
        "generator_shard_id": int(row["generator_shard_id"]),
        "manifest_order": int(row["manifest_order"]),
        "num_candidates_requested": int(generator_meta["num_candidates"]),
        "num_candidates_generated": len(candidates),
        "candidates": candidates,
        "row": row,
        "generator_metadata": generator_meta,
        "generated_at": utc_now_iso(),
    }
    atomic_write_json(generated_path(run_dir, int(row["generator_shard_id"]), str(row["row_uid"])), payload)


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    run_dir = ensure_dir(args.run_dir)
    shard_id = int(args.shard_id)
    generator_id = args.generator_id or f"gen{shard_id:02d}"
    cfg = load_run_config(run_dir)
    k = int(args.num_candidates or cfg.get("num_candidates") or 4)
    rows = load_shard_manifest(run_dir, shard_id)
    done = 0
    skipped = 0
    stop_after = int(args.stop_after_generated_rows or os.getenv("GRPO_MC_STOP_AFTER_ROWS_PER_SHARD", "0") or 0)

    generator_meta = {
        "generator_id": generator_id,
        "generator_shard_id": shard_id,
        "pid": os.getpid(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "num_candidates": k,
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "mock_generate": bool(args.mock_generate),
        "started_at": utc_now_iso(),
    }
    atomic_write_json(status_path(run_dir, f"generator_{generator_id}"), {"status": "starting", **generator_meta})

    llm = lora_request = sampling_cls = None
    model_meta: dict[str, Any] = {}
    if not args.mock_generate:
        llm, lora_request, sampling_cls, model_meta = build_vllm_engine(args, cfg)
        generator_meta["model_metadata"] = model_meta
        atomic_write_json(run_dir / "model_meta.json", model_meta)

    pending_batch: list[dict[str, Any]] = []
    t0_run = time.time()

    def flush_batch() -> None:
        nonlocal done, skipped, pending_batch
        if not pending_batch:
            return
        batch = pending_batch
        pending_batch = []
        t0 = time.time()
        if args.mock_generate:
            candidates_batch = [mock_completions(row, k) for row in batch]
        else:
            candidates_batch = generate_batch_vllm(
                llm,
                lora_request,
                sampling_cls,
                batch,
                k=k,
                max_new_tokens=int(args.max_new_tokens),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
            )
        seconds = time.time() - t0
        candidates_written = 0
        for row, candidates in zip(batch, candidates_batch):
            write_generated_row(run_dir, row, candidates, generator_meta)
            done += 1
            candidates_written += len(candidates)
        append_jsonl(
            event_log_path(run_dir, f"generator_{generator_id}"),
            {
                "event": "batch_generated",
                "timestamp": utc_now_iso(),
                "generator_id": generator_id,
                "shard_id": shard_id,
                "rows": len(batch),
                "candidates": candidates_written,
                "seconds": seconds,
                "candidates_per_second": float(candidates_written / seconds) if seconds > 0 else 0.0,
                "rows_done_this_process": done,
                "rows_skipped_this_process": skipped,
            },
        )
        atomic_write_json(
            status_path(run_dir, f"generator_{generator_id}"),
            {
                "status": "running",
                **generator_meta,
                "rows_done_this_process": done,
                "rows_skipped_this_process": skipped,
                "run_seconds": time.time() - t0_run,
                "updated_at": utc_now_iso(),
            },
        )

    try:
        for row in rows:
            uid = str(row["row_uid"])
            out_path = generated_path(run_dir, shard_id, uid)
            if read_json_or_none(out_path) is not None:
                skipped += 1
                continue
            pending_batch.append(row)
            if len(pending_batch) >= int(args.batch_size):
                flush_batch()
            if stop_after > 0 and done >= stop_after:
                break
        flush_batch()
    except Exception as exc:
        append_jsonl(
            failure_log_path(run_dir, f"generator_{generator_id}"),
            {"event": "generator_failed", "timestamp": utc_now_iso(), "error": f"{type(exc).__name__}: {exc}"},
        )
        atomic_write_json(
            status_path(run_dir, f"generator_{generator_id}"),
            {"status": "failed", **generator_meta, "error": f"{type(exc).__name__}: {exc}", "updated_at": utc_now_iso()},
        )
        raise

    shard_complete = done + skipped >= len(rows)
    if shard_complete:
        atomic_write_json(
            generator_done_path(run_dir, shard_id),
            {"status": "completed", "generator_id": generator_id, "shard_id": shard_id, "completed_at": utc_now_iso()},
        )
    atomic_write_json(
        status_path(run_dir, f"generator_{generator_id}"),
        {
            "status": "completed" if shard_complete else "stopped_early",
            **generator_meta,
            "shard_complete": shard_complete,
            "rows_done_this_process": done,
            "rows_skipped_this_process": skipped,
            "run_seconds": time.time() - t0_run,
            "updated_at": utc_now_iso(),
        },
    )
    print(f"GENERATOR_DONE generator_id={generator_id} shard_id={shard_id} rows_done={done} rows_skipped={skipped} shard_complete={shard_complete}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
