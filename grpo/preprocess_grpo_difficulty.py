import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None
    LoRARequest = None


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT.parent / ".env"
DEFAULT_DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
DEFAULT_OUTPUT_DATASET_ID = f"{DEFAULT_DATASET_ID}-grpo-preprocessed-v1"
DEFAULT_SFT_ADAPTER_REPO = "Shaer-AI/Shaer-adapters"
DEFAULT_SFT_ADAPTER_MODE = "fresh_sft/train"
DEFAULT_FIT_PROMPT = "prompts/meaning_fit.yaml"
DEFAULT_SUBSTANCE_PROMPT = "prompts/meaning_substance.yaml"

sys.path.append(str(ROOT))

from rewards.common import (  # noqa: E402
    append_jsonl,
    ensure_dir,
    extract_text,
    load_and_prepare_dataset,
    load_env,
    load_yaml,
    resolve_sft_adapter_path,
    save_json,
    score_count_adherence,
    setup_logger,
)
from rewards.meaning_fit import score_meaning_fit  # noqa: E402
from rewards.meaning_substance import score_meaning_substance  # noqa: E402
from rewards.meter import score_meter_poem  # noqa: E402


def load_cfg() -> dict[str, Any]:
    with open(ROOT / "grpo_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def summarize(values: list[float]) -> dict[str, float]:
    vals = [safe_float(v) for v in values]
    if not vals:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(pstdev(vals) if len(vals) > 1 else 0.0),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def rate_at_least(values: list[float], threshold: float) -> float:
    if not values:
        return 0.0
    vals = [safe_float(v) for v in values]
    return float(sum(v >= threshold for v in vals) / len(vals))


def build_vllm_engine(cfg: dict[str, Any]):
    if LLM is None or SamplingParams is None or LoRARequest is None:
        raise RuntimeError("vLLM is not available in the current environment.")

    base_model_id = os.getenv("BASE_MODEL_ID", "").strip()
    if not base_model_id:
        raise ValueError("BASE_MODEL_ID is missing")

    adapter_repo = os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO).strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    adapter_path, adapter_meta = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token)

    gen_cfg = cfg.get("generation", {})
    prep_cfg = cfg.get("preprocess_grpo", {})
    if bool(prep_cfg.get("flashinfer_disable_version_check", True)):
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    max_prompt_length = int(prep_cfg.get("max_prompt_length", gen_cfg.get("max_prompt_length", 1024)))
    max_completion_length = int(prep_cfg.get("max_completion_length", gen_cfg.get("max_completion_length", 640)))
    max_model_len = max_prompt_length + max_completion_length

    llm = LLM(
        model=base_model_id,
        tokenizer=base_model_id,
        enable_lora=True,
        max_loras=int(prep_cfg.get("vllm_max_loras", 1)),
        max_lora_rank=int(prep_cfg.get("vllm_max_lora_rank", 64)),
        tensor_parallel_size=int(prep_cfg.get("vllm_tensor_parallel_size", 1)),
        gpu_memory_utilization=float(prep_cfg.get("vllm_gpu_memory_utilization", 0.30)),
        dtype=str(prep_cfg.get("vllm_dtype", "bfloat16")),
        max_model_len=max_model_len,
        trust_remote_code=bool(prep_cfg.get("trust_remote_code", True)),
        hf_token=hf_token,
    )
    lora_request = LoRARequest(
        lora_name=Path(adapter_path).name or "sft_adapter",
        lora_int_id=1,
        lora_path=adapter_path,
        base_model_name=base_model_id or None,
    )
    return llm, lora_request, {"adapter_path": adapter_path, **adapter_meta}


def generate_k_vllm(
    llm,
    lora_request,
    prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        detokenize=True,
        skip_special_tokens=True,
    )
    outputs = llm.generate(
        [prompt],
        sampling_params,
        lora_request=[lora_request],
        use_tqdm=False,
    )
    if not outputs:
        return []
    return [completion.text.strip() for completion in outputs[0].outputs]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_processed_source_indexes(path: Path) -> set[int]:
    processed: set[int] = set()
    for row in read_jsonl(path):
        try:
            processed.add(int(row["source_index"]))
        except Exception:
            continue
    return processed


def classify_difficulty(weighted_scores: list[float], meter_scores: list[float], count_scores: list[float]) -> str:
    total_summary = summarize(weighted_scores)
    meter_summary = summarize(meter_scores)
    count_exact_rate = rate_at_least(count_scores, 0.999999)
    total_07 = rate_at_least(weighted_scores, 0.70)
    total_05 = rate_at_least(weighted_scores, 0.50)

    if total_summary["mean"] >= 0.68 or total_07 >= 0.50:
        return "easy"
    if total_summary["max"] < 0.45 or (total_05 == 0.0 and meter_summary["mean"] < 0.25 and count_exact_rate < 0.25):
        return "hard"
    return "medium"


def row_result_record(
    row: dict[str, Any],
    completions: list[str],
    reward_rows: list[dict[str, Any]],
    reward_weights: dict[str, float],
    model_meta: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    meter_scores = [safe_float(x["meter_score"]) for x in reward_rows]
    fit_scores = [safe_float(x["fit_score"]) for x in reward_rows]
    substance_scores = [safe_float(x["substance_score"]) for x in reward_rows]
    count_scores = [safe_float(x["count_score"]) for x in reward_rows]

    weighted_total_scores = []
    for idx in range(len(reward_rows)):
        weighted_total_scores.append(
            reward_weights["meter"] * meter_scores[idx]
            + reward_weights["meaning_fit"] * fit_scores[idx]
            + reward_weights["meaning_substance"] * substance_scores[idx]
            + reward_weights["count_adherence"] * count_scores[idx]
        )

    meter_summary = summarize(meter_scores)
    fit_summary = summarize(fit_scores)
    substance_summary = summarize(substance_scores)
    count_summary = summarize(count_scores)
    total_summary = summarize(weighted_total_scores)
    generated_bayts = [int(x["generated_bayts"]) for x in reward_rows]
    valid_bayts = [int(x["meter_num_valid_bayts"]) for x in reward_rows]
    odd_tail_flags = [bool(x["count_has_odd_tail"]) for x in reward_rows]

    return {
        "source_index": int(row["source_index"]),
        "dataset_split": row["dataset_split"],
        "prompt": row["prompt"],
        "description": row["description"],
        "base_meter": row["base_meter"],
        "form": row.get("form", ""),
        "meter_label": row["meter_label"],
        "requested_bayts": int(row["requested_bayts"]),
        "requested_lines": int(row["requested_lines"]),
        "length_bucket": row["length_bucket"],
        "model_adapter_repo": os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO),
        "model_adapter_mode": os.getenv("SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE),
        "model_adapter_path": model_meta.get("adapter_path", ""),
        "num_generations": len(completions),
        "sample_completions": completions,
        "sample_meter_scores": meter_scores,
        "sample_fit_scores": fit_scores,
        "sample_substance_scores": substance_scores,
        "sample_count_scores": count_scores,
        "sample_weighted_total_scores": weighted_total_scores,
        "sample_generated_bayts": generated_bayts,
        "sample_meter_num_valid_bayts": valid_bayts,
        "sample_count_has_odd_tail": odd_tail_flags,
        "meter_mean": meter_summary["mean"],
        "meter_median": meter_summary["median"],
        "meter_std": meter_summary["std"],
        "meter_min": meter_summary["min"],
        "meter_max": meter_summary["max"],
        "fit_mean": fit_summary["mean"],
        "fit_median": fit_summary["median"],
        "fit_std": fit_summary["std"],
        "fit_min": fit_summary["min"],
        "fit_max": fit_summary["max"],
        "substance_mean": substance_summary["mean"],
        "substance_median": substance_summary["median"],
        "substance_std": substance_summary["std"],
        "substance_min": substance_summary["min"],
        "substance_max": substance_summary["max"],
        "count_mean": count_summary["mean"],
        "count_median": count_summary["median"],
        "count_std": count_summary["std"],
        "count_min": count_summary["min"],
        "count_max": count_summary["max"],
        "weighted_total_mean": total_summary["mean"],
        "weighted_total_median": total_summary["median"],
        "weighted_total_std": total_summary["std"],
        "weighted_total_min": total_summary["min"],
        "weighted_total_max": total_summary["max"],
        "pass_rate_total_ge_03": rate_at_least(weighted_total_scores, 0.30),
        "pass_rate_total_ge_05": rate_at_least(weighted_total_scores, 0.50),
        "pass_rate_total_ge_07": rate_at_least(weighted_total_scores, 0.70),
        "pass_rate_meter_ge_05": rate_at_least(meter_scores, 0.50),
        "pass_rate_meter_ge_07": rate_at_least(meter_scores, 0.70),
        "pass_rate_count_exact": rate_at_least(count_scores, 0.999999),
        "difficulty_bucket": classify_difficulty(weighted_total_scores, meter_scores, count_scores),
        "processing_started_at": runtime["started_at"],
        "processing_finished_at": utc_now_iso(),
        "runtime_seconds": float(time.time() - runtime["t0"]),
        "fit_cache_hits": int(sum(1 for x in reward_rows if x["fit_cache_hit"])),
        "substance_cache_hits": int(sum(1 for x in reward_rows if x["substance_cache_hit"])),
        "fit_errors": int(sum(1 for x in reward_rows if x["fit_error"])),
        "substance_errors": int(sum(1 for x in reward_rows if x["substance_error"])),
    }


def compute_split_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    difficulty = Counter()
    meters = Counter()
    for row in rows:
        difficulty[str(row.get("difficulty_bucket", ""))] += 1
        meters[str(row.get("base_meter", ""))] += 1
    return {
        "rows": len(rows),
        "difficulty_counts": dict(difficulty),
        "base_meter_counts": dict(meters),
    }


def score_single_completion(
    row: dict[str, Any],
    completion_text: str,
    fit_prompt_file: str,
    substance_prompt_file: str,
    cache_dir: str,
) -> dict[str, Any]:
    meter_out = score_meter_poem(completion_text, row["meter_label"], base_meter=row["base_meter"], aggregator="logmean")
    fit_out = score_meaning_fit(row["description"], completion_text, prompt_file=fit_prompt_file, cache_dir=cache_dir)
    substance_out = score_meaning_substance(completion_text, prompt_file=substance_prompt_file, cache_dir=cache_dir)
    count_out = score_count_adherence(int(row["requested_bayts"]), completion_text)
    return {
        "meter_score": safe_float(meter_out["score"]),
        "meter_num_valid_bayts": int(meter_out["num_valid_bayts"]),
        "meter_num_skipped_bayts": int(meter_out["num_skipped_bayts"]),
        "fit_score": safe_float(fit_out["score"]),
        "fit_cache_hit": bool(fit_out["cache_hit"]),
        "fit_error": bool(fit_out.get("error")),
        "substance_score": safe_float(substance_out["score"]),
        "substance_cache_hit": bool(substance_out["cache_hit"]),
        "substance_error": bool(substance_out.get("error")),
        "count_score": safe_float(count_out["score"]),
        "generated_bayts": int(count_out["generated_bayts"]),
        "count_has_odd_tail": bool(count_out.get("has_odd_tail", False)),
    }


def maybe_limit_dataset(ds, max_rows: int | None):
    if max_rows is None:
        return ds
    max_rows = max(0, min(int(max_rows), len(ds)))
    return ds.select(range(max_rows))


def process_split(
    split_name: str,
    ds,
    llm,
    lora_request,
    output_dir: Path,
    logger,
    cfg: dict[str, Any],
    model_meta: dict[str, Any],
    run_state: dict[str, Any],
):
    prep_cfg = cfg["preprocess_grpo"]
    reward_weights = cfg["phase1"]["reward_weights"]
    fit_prompt_file = os.getenv("MEANING_FIT_PROMPT_FILE", DEFAULT_FIT_PROMPT).strip() or DEFAULT_FIT_PROMPT
    substance_prompt_file = os.getenv("MEANING_SUBSTANCE_PROMPT_FILE", DEFAULT_SUBSTANCE_PROMPT).strip() or DEFAULT_SUBSTANCE_PROMPT
    cache_dir = os.getenv("REWARD_CACHE_DIR", "./cache/reward_cache")
    split_file = output_dir / f"{split_name}.jsonl"
    failures_file = output_dir / f"{split_name}_failures.jsonl"
    processed = load_processed_source_indexes(split_file)

    total_rows = len(ds)
    logger.info("split=%s total_rows=%d already_processed=%d", split_name, total_rows, len(processed))

    for idx, row in enumerate(ds):
        source_index = int(row["source_index"])
        if source_index in processed:
            continue

        runtime = {"started_at": utc_now_iso(), "t0": time.time()}
        try:
            completions = generate_k_vllm(
                llm=llm,
                lora_request=lora_request,
                prompt=row["prompt"],
                k=int(prep_cfg["num_generations"]),
                max_new_tokens=int(prep_cfg["max_completion_length"]),
                temperature=float(prep_cfg["temperature"]),
                top_p=float(prep_cfg["top_p"]),
            )
            reward_rows = [
                score_single_completion(
                    row=row,
                    completion_text=completion,
                    fit_prompt_file=fit_prompt_file,
                    substance_prompt_file=substance_prompt_file,
                    cache_dir=cache_dir,
                )
                for completion in completions
            ]
            record = row_result_record(
                row=row,
                completions=completions,
                reward_rows=reward_rows,
                reward_weights=reward_weights,
                model_meta=model_meta,
                runtime=runtime,
            )
            append_jsonl(split_file, record)
            processed.add(source_index)
        except Exception as exc:
            append_jsonl(
                failures_file,
                {
                    "split": split_name,
                    "source_index": source_index,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timestamp": utc_now_iso(),
                },
            )
            logger.exception("row_failed split=%s source_index=%s", split_name, source_index)

        if len(processed) % int(prep_cfg.get("progress_every", 10)) == 0:
            progress = {
                "timestamp": utc_now_iso(),
                "split": split_name,
                "processed_rows": len(processed),
                "total_rows": total_rows,
                "remaining_rows": max(0, total_rows - len(processed)),
            }
            save_json(progress, output_dir / "progress.json")
            logger.info(
                "progress split=%s processed=%d/%d remaining=%d",
                split_name,
                progress["processed_rows"],
                total_rows,
                progress["remaining_rows"],
            )


def publish_dataset(output_dir: Path, output_dataset_id: str, logger):
    split_files = {split: str(output_dir / f"{split}.jsonl") for split in ("train", "eval", "test")}
    missing = [split for split, path in split_files.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files for publish: {missing}")

    ds = load_dataset("json", data_files=split_files)
    dataset_dict = DatasetDict({split: ds[split] for split in ds.keys()})

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    create_repo(repo_id=output_dataset_id, repo_type="dataset", token=hf_token, exist_ok=True)
    logger.info("Pushing dataset to hub: %s", output_dataset_id)
    dataset_dict.push_to_hub(output_dataset_id, token=hf_token)


def save_run_summary(output_dir: Path, dataset_id: str, output_dataset_id: str, model_meta: dict[str, Any]):
    summary = {
        "dataset_id": dataset_id,
        "output_dataset_id": output_dataset_id,
        "model_adapter_repo": os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO),
        "model_adapter_mode": os.getenv("SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE),
        "model_meta": model_meta,
        "completed_at": utc_now_iso(),
        "splits": {
            split: compute_split_summary(output_dir / f"{split}.jsonl")
            for split in ("train", "eval", "test")
        },
    }
    save_json(summary, output_dir / "run_summary.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()

    load_dotenv(ENV_PATH, override=False)
    load_env()

    cfg = load_cfg()
    prep_cfg = cfg["preprocess_grpo"]
    run_root = ensure_dir(ROOT / prep_cfg["output_root"])
    if args.publish_only:
        run_dir_raw = os.getenv("GRPO_PREPROCESS_RUN_DIR", "").strip()
        if not run_dir_raw:
            raise ValueError("GRPO_PREPROCESS_RUN_DIR is required for --publish-only")
        run_dir = Path(run_dir_raw).expanduser()
    else:
        run_dir = ensure_dir(run_root / f"preprocess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    logger = setup_logger("preprocess_grpo_difficulty", run_dir / "preprocess.log", also_stdout=True)

    dataset_id = os.getenv("GRPO_PREPROCESS_DATASET_ID", DEFAULT_DATASET_ID).strip() or DEFAULT_DATASET_ID
    output_dataset_id = os.getenv("GRPO_PREPROCESS_OUTPUT_DATASET_ID", DEFAULT_OUTPUT_DATASET_ID).strip() or DEFAULT_OUTPUT_DATASET_ID
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    save_json(cfg, run_dir / "config_snapshot.json")
    save_json(
        {
            "dataset_id": dataset_id,
            "output_dataset_id": output_dataset_id,
            "base_model_id": os.getenv("BASE_MODEL_ID", ""),
            "sft_adapter_repo": os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO),
            "sft_adapter_mode": os.getenv("SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE),
            "started_at": utc_now_iso(),
            "max_rows_per_split": args.max_rows_per_split,
            "skip_push": bool(args.skip_push),
            "publish_only": bool(args.publish_only),
        },
        run_dir / "runtime_snapshot.json",
    )

    if args.publish_only:
        publish_dataset(run_dir, output_dataset_id, logger)
        logger.info("publish_only complete run_dir=%s output_dataset_id=%s", run_dir, output_dataset_id)
        return

    llm, lora_request, model_meta = build_vllm_engine(cfg)
    save_json(model_meta, run_dir / "model_meta.json")
    logger.info(
        "Resolved adapter | repo=%s mode=%s path=%s",
        os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO),
        os.getenv("SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE),
        model_meta.get("adapter_path", ""),
    )

    split_map = {
        "train": "train",
        "eval": "eval",
        "test": "test",
    }
    for split_name, source_split in split_map.items():
        ds = load_and_prepare_dataset(
            dataset_id=dataset_id,
            split=source_split,
            hf_token=hf_token,
        )
        ds = maybe_limit_dataset(ds, args.max_rows_per_split)
        ds = ds.map(lambda row: {**row, "dataset_split": split_name})
        process_split(
            split_name=split_name,
            ds=ds,
            llm=llm,
            lora_request=lora_request,
            output_dir=run_dir,
            logger=logger,
            cfg=cfg,
            model_meta=model_meta,
            run_state={},
        )

    save_run_summary(run_dir, dataset_id, output_dataset_id, model_meta)

    if not args.skip_push:
        publish_dataset(run_dir, output_dataset_id, logger)
    logger.info("PREPROCESS_GRPO_DIFFICULTY_DONE run_dir=%s output_dataset_id=%s", run_dir, output_dataset_id)


if __name__ == "__main__":
    main()
