import json
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import yaml
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import create_repo

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
    load_and_prepare_dataset,
    load_env,
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


def load_runtime() -> dict[str, Any]:
    load_dotenv(ENV_PATH, override=False)
    load_env()
    cfg = load_cfg()
    runtime = {
        "dataset_id": os.getenv("GRPO_PREPROCESS_DATASET_ID", DEFAULT_DATASET_ID).strip() or DEFAULT_DATASET_ID,
        "output_dataset_id": os.getenv("GRPO_PREPROCESS_OUTPUT_DATASET_ID", DEFAULT_OUTPUT_DATASET_ID).strip()
        or DEFAULT_OUTPUT_DATASET_ID,
        "base_model_id": os.getenv("BASE_MODEL_ID", "").strip(),
        "sft_adapter_repo": os.getenv("SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO).strip() or DEFAULT_SFT_ADAPTER_REPO,
        "sft_adapter_mode": os.getenv("SFT_ADAPTER_MODE", DEFAULT_SFT_ADAPTER_MODE).strip() or DEFAULT_SFT_ADAPTER_MODE,
    }
    return {"cfg": cfg, "runtime": runtime}


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
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
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


def maybe_limit_dataset(ds, max_rows: int | None):
    if max_rows is None:
        return ds
    max_rows = max(0, min(int(max_rows), len(ds)))
    return ds.select(range(max_rows))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def iter_meter_balanced_pending_rows(ds, processed_source_indexes: set[int]):
    rows_by_meter: dict[str, deque] = defaultdict(deque)
    for row in ds:
        source_index = int(row["source_index"])
        if source_index in processed_source_indexes:
            continue
        meter_key = str(row.get("base_meter") or row.get("meter_label") or "__unknown__")
        rows_by_meter[meter_key].append(row)

    active_meters = deque(sorted(rows_by_meter.keys()))
    while active_meters:
        meter_key = active_meters.popleft()
        bucket = rows_by_meter.get(meter_key)
        if not bucket:
            continue
        yield bucket.popleft()
        if bucket:
            active_meters.append(meter_key)


def pending_meter_counts(ds, processed_source_indexes: set[int]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in ds:
        source_index = int(row["source_index"])
        if source_index in processed_source_indexes:
            continue
        meter_key = str(row.get("base_meter") or row.get("meter_label") or "__unknown__")
        counts[meter_key] += 1
    return dict(sorted(counts.items()))


def pipeline_paths(run_dir: Path, split_name: str) -> dict[str, Path]:
    return {
        "generated": run_dir / f"{split_name}_generations.jsonl",
        "generated_failures": run_dir / f"{split_name}_generation_failures.jsonl",
        "judged": run_dir / f"{split_name}.jsonl",
        "judge_failures": run_dir / f"{split_name}_judge_failures.jsonl",
    }


def claim_path(run_dir: Path, split_name: str, source_index: int) -> Path:
    return run_dir / "judge_claims" / split_name / f"{int(source_index)}.json"


def done_path(run_dir: Path, split_name: str, source_index: int) -> Path:
    return run_dir / "judge_done" / split_name / f"{int(source_index)}.done"


def try_claim_row(run_dir: Path, split_name: str, source_index: int, worker_id: str, stale_after_seconds: int) -> Path | None:
    path = claim_path(run_dir, split_name, source_index)
    ensure_dir(path.parent)
    payload = {
        "worker_id": str(worker_id),
        "source_index": int(source_index),
        "split": split_name,
        "claimed_at": utc_now_iso(),
        "pid": os.getpid(),
    }

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags)
    except FileExistsError:
        try:
            age_seconds = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return try_claim_row(run_dir, split_name, source_index, worker_id, stale_after_seconds)
        if age_seconds <= max(1, int(stale_after_seconds)):
            return None
        try:
            path.unlink()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            fd = os.open(path, flags)
        except FileExistsError:
            return None

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path


def release_claim(path: Path | None):
    if not path:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def seed_done_markers(run_dir: Path, split_name: str, source_indexes: set[int]) -> None:
    target_dir = run_dir / "judge_done" / split_name
    ensure_dir(target_dir)
    for source_index in source_indexes:
        marker = target_dir / f"{int(source_index)}.done"
        if marker.exists():
            continue
        marker.touch()


def try_mark_done(run_dir: Path, split_name: str, source_index: int) -> Path | None:
    path = done_path(run_dir, split_name, source_index)
    ensure_dir(path.parent)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(path, flags)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(utc_now_iso())
    return path


def build_vllm_engine(cfg: dict[str, Any]):
    if LLM is None or SamplingParams is None or LoRARequest is None:
        raise RuntimeError("vLLM is not available in the current environment.")

    runtime = load_runtime()["runtime"]
    base_model_id = runtime["base_model_id"]
    if not base_model_id:
        raise ValueError("BASE_MODEL_ID is missing")

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    adapter_path, adapter_meta = resolve_sft_adapter_path(adapter_repo=runtime["sft_adapter_repo"], hf_token=hf_token)
    prep_cfg = cfg["preprocess_grpo_pipeline"]

    if bool(prep_cfg.get("flashinfer_disable_version_check", True)):
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    max_prompt_length = int(prep_cfg.get("max_prompt_length", 1024))
    max_completion_length = int(prep_cfg.get("max_completion_length", 640))
    max_model_len = max_prompt_length + max_completion_length

    llm = LLM(
        model=base_model_id,
        tokenizer=base_model_id,
        enable_lora=True,
        max_loras=int(prep_cfg.get("vllm_max_loras", 1)),
        max_lora_rank=int(prep_cfg.get("vllm_max_lora_rank", 64)),
        tensor_parallel_size=int(prep_cfg.get("vllm_tensor_parallel_size", 1)),
        gpu_memory_utilization=float(prep_cfg.get("vllm_gpu_memory_utilization", 0.75)),
        dtype=str(prep_cfg.get("vllm_dtype", "bfloat16")),
        max_model_len=max_model_len,
        trust_remote_code=bool(prep_cfg.get("trust_remote_code", True)),
        hf_token=hf_token,
        disable_log_stats=True,
    )
    lora_request = LoRARequest(
        lora_name=Path(adapter_path).name or "sft_adapter",
        lora_int_id=1,
        lora_path=adapter_path,
        base_model_name=base_model_id or None,
    )
    return llm, lora_request, {"adapter_path": adapter_path, **adapter_meta}


def generate_k_vllm(llm, lora_request, prompt: str, k: int, max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        detokenize=True,
        skip_special_tokens=True,
    )
    outputs = llm.generate([prompt], sampling_params, lora_request=[lora_request], use_tqdm=False)
    if not outputs:
        return []
    return [completion.text.strip() for completion in outputs[0].outputs]


def generate_batch_k_vllm(llm, lora_request, prompts: list[str], k: int, max_new_tokens: int, temperature: float, top_p: float) -> list[list[str]]:
    if not prompts:
        return []
    sampling_params = SamplingParams(
        n=k,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        detokenize=True,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=[lora_request] * len(prompts), use_tqdm=False)
    out: list[list[str]] = []
    for item in outputs:
        out.append([completion.text.strip() for completion in item.outputs])
    return out


def score_single_completion(row: dict[str, Any], completion_text: str, fit_prompt_file: str, substance_prompt_file: str, cache_dir: str) -> dict[str, Any]:
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


def row_result_record(row: dict[str, Any], reward_rows: list[dict[str, Any]], reward_weights: dict[str, float], model_meta: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    completions = row["sample_completions"]
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
        "model_adapter_repo": load_runtime()["runtime"]["sft_adapter_repo"],
        "model_adapter_mode": load_runtime()["runtime"]["sft_adapter_mode"],
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
    return {"rows": len(rows)}


def publish_dataset(run_dir: Path, output_dataset_id: str, logger):
    split_files = {split: str(run_dir / f"{split}.jsonl") for split in ("train", "eval", "test")}
    missing = [split for split, path in split_files.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing split files for publish: {missing}")
    ds = load_dataset("json", data_files=split_files)
    dataset_dict = DatasetDict({split: ds[split] for split in ds.keys()})
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    create_repo(repo_id=output_dataset_id, repo_type="dataset", token=hf_token, exist_ok=True)
    logger.info("Pushing dataset to hub: %s", output_dataset_id)
    dataset_dict.push_to_hub(output_dataset_id, token=hf_token)


def write_status(path: Path, payload: dict[str, Any]) -> None:
    save_json(payload, path)
