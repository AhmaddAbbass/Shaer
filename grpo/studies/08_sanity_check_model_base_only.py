import json
import os
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover
    LLM = None
    SamplingParams = None
    LoRARequest = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    append_jsonl,
    ensure_dir,
    get_dataset_id_for_phase,
    load_and_prepare_dataset,
    load_yaml,
    parse_allowed_meters_env,
    resolve_sft_adapter_path,
    save_json,
    setup_logger,
)
from rewards.meter import score_meter_poem

load_dotenv(override=False)


def resolve_generation_backend(cfg):
    study_cfg = cfg["studies"]["sanity_check_model"]
    backend = str(study_cfg.get("backend", "")).strip().lower()
    if backend:
        return backend
    if cfg.get("generation", {}).get("use_vllm", False):
        return "vllm"
    return "transformers"


def safe_filename(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "meter"


def build_transformers_model():
    base_model_id = os.getenv("BASE_MODEL_ID", "").strip()
    adapter_repo = os.getenv("SFT_ADAPTER_REPO", "").strip()
    hf_token = os.getenv("HF_TOKEN")
    adapter_path, _ = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model = PeftModel.from_pretrained(model, adapter_path, token=hf_token)
    model.eval()
    return tokenizer, model


def build_vllm_engine(cfg):
    if LLM is None or SamplingParams is None or LoRARequest is None:
        raise RuntimeError("vLLM is not available in the current environment.")

    base_model_id = os.getenv("BASE_MODEL_ID", "").strip()
    adapter_repo = os.getenv("SFT_ADAPTER_REPO", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    adapter_path, adapter_meta = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token)

    gen_cfg = cfg.get("generation", {})
    study_cfg = cfg["studies"]["sanity_check_model"]

    if bool(study_cfg.get("flashinfer_disable_version_check", True)):
        os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    max_prompt_length = int(gen_cfg.get("max_prompt_length", 1024))
    max_completion_length = int(gen_cfg.get("max_completion_length", study_cfg["max_new_tokens"]))
    default_max_model_len = max_prompt_length + max(max_completion_length, int(study_cfg["max_new_tokens"]))

    llm = LLM(
        model=base_model_id,
        tokenizer=base_model_id,
        enable_lora=True,
        max_loras=int(study_cfg.get("vllm_max_loras", 1)),
        max_lora_rank=int(study_cfg.get("vllm_max_lora_rank", 64)),
        tensor_parallel_size=int(study_cfg.get("vllm_tensor_parallel_size", 1)),
        gpu_memory_utilization=float(
            study_cfg.get(
                "vllm_gpu_memory_utilization",
                gen_cfg.get("vllm_gpu_memory_utilization", 0.30),
            )
        ),
        dtype=study_cfg.get("vllm_dtype", "bfloat16"),
        max_model_len=int(study_cfg.get("vllm_max_model_len", default_max_model_len)),
        trust_remote_code=bool(study_cfg.get("trust_remote_code", False)),
        hf_token=hf_token,
    )
    lora_request = LoRARequest(
        lora_name=Path(adapter_path).name or "sft_adapter",
        lora_int_id=1,
        lora_path=adapter_path,
        base_model_name=base_model_id or None,
    )
    return llm, lora_request, adapter_meta


def generate_k_transformers(tokenizer, model, prompt, k, max_new_tokens, temperature, top_p):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return [t[len(prompt):].strip() if t.startswith(prompt) else t.strip() for t in texts]


def generate_k_vllm(llm, lora_request, prompt, k, max_new_tokens, temperature, top_p):
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        path.unlink()
    for row in rows:
        append_jsonl(path, row)


def add_threshold_flags(row: dict, thresholds: list[float], score: float) -> None:
    for thr in thresholds:
        row[f"above_{str(thr).replace('.', '_')}"] = float(score >= thr)


def summarize_prompt_group(rows: list[dict], thresholds: list[float]) -> dict:
    scores = [float(r["meter_score"]) for r in rows]
    scored_bayts = [int(r["num_scored_bayts"]) for r in rows]
    output_pairs = [int(r["num_output_line_pairs"]) for r in rows]
    output_lines = [int(r["num_output_lines"]) for r in rows]
    odd_tail_flags = [bool(r["has_unpaired_last_line"]) for r in rows]
    prompt_row = rows[0]
    summary = {
        "base_meter": prompt_row["base_meter"],
        "requested_meter_label": prompt_row["requested_meter_label"],
        "source_index": int(prompt_row["source_index"]),
        "prompt_group_id": prompt_row["prompt_group_id"],
        "prompt_rank_within_base": int(prompt_row["prompt_rank_within_base"]),
        "requested_bayts": int(prompt_row["requested_bayts"]),
        "requested_lines": int(prompt_row["requested_lines"]),
        "prompt_preview": prompt_row["prompt_preview"],
        "num_candidates": len(rows),
        "candidate_mean_score": float(np.mean(scores)),
        "candidate_median_score": float(np.median(scores)),
        "candidate_std_score": float(np.std(scores)),
        "candidate_min_score": float(np.min(scores)),
        "candidate_max_score": float(np.max(scores)),
        "candidate_score_gap": float(np.max(scores) - np.min(scores)),
        "mean_num_scored_bayts": float(np.mean(scored_bayts)),
        "min_num_scored_bayts": int(np.min(scored_bayts)),
        "max_num_scored_bayts": int(np.max(scored_bayts)),
        "mean_num_output_line_pairs": float(np.mean(output_pairs)),
        "min_num_output_line_pairs": int(np.min(output_pairs)),
        "max_num_output_line_pairs": int(np.max(output_pairs)),
        "mean_num_output_lines": float(np.mean(output_lines)),
        "odd_tail_rate_within_prompt": float(np.mean(odd_tail_flags)),
    }
    for thr in thresholds:
        summary[f"has_good_candidate_{str(thr).replace('.', '_')}"] = float(any(score >= thr for score in scores))
    return summary


def decide_action(row: pd.Series) -> str:
    if (
        row["prompt_has_good_candidate_0_7_rate"] >= 0.6
        and row["prompt_best_score_mean"] >= 0.7
        and row["candidate_median_score"] >= 0.45
    ):
        return "keep"
    if (
        row["prompt_has_good_candidate_0_5_rate"] <= 0.2
        and row["prompt_best_score_mean"] < 0.35
        and row["candidate_median_score"] < 0.15
    ):
        return "candidate_for_removal_or_extra_filtering"
    return "keep_but_flag_weak"


def build_base_summary(results_df: pd.DataFrame, prompt_df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    candidate_agg = (
        results_df.groupby("base_meter", as_index=False)
        .agg(
            num_generations=("meter_score", "size"),
            num_prompt_groups=("prompt_group_id", "nunique"),
            candidate_mean_score=("meter_score", "mean"),
            candidate_median_score=("meter_score", "median"),
            candidate_std_score=("meter_score", "std"),
            candidate_min_score=("meter_score", "min"),
            candidate_max_score=("meter_score", "max"),
            mean_requested_bayts=("requested_bayts", "mean"),
            mean_num_scored_bayts=("num_scored_bayts", "mean"),
            mean_num_output_line_pairs=("num_output_line_pairs", "mean"),
            odd_tail_rate=("has_unpaired_last_line", "mean"),
        )
    )
    for thr in thresholds:
        col = f"above_{str(thr).replace('.', '_')}"
        rate = (
            results_df.groupby("base_meter", as_index=False)[col]
            .mean()
            .rename(columns={col: f"candidate_pass_rate_{str(thr).replace('.', '_')}"})
        )
        candidate_agg = candidate_agg.merge(rate, on="base_meter", how="left")

    prompt_agg = (
        prompt_df.groupby("base_meter", as_index=False)
        .agg(
            prompt_best_score_mean=("candidate_max_score", "mean"),
            prompt_best_score_min=("candidate_max_score", "min"),
            prompt_best_score_max=("candidate_max_score", "max"),
            prompt_mean_score_mean=("candidate_mean_score", "mean"),
            prompt_mean_score_min=("candidate_mean_score", "min"),
            prompt_mean_score_max=("candidate_mean_score", "max"),
            prompt_worst_score_mean=("candidate_min_score", "mean"),
            prompt_worst_score_min=("candidate_min_score", "min"),
            prompt_worst_score_max=("candidate_min_score", "max"),
            prompt_gap_mean=("candidate_score_gap", "mean"),
            prompt_gap_min=("candidate_score_gap", "min"),
            prompt_gap_max=("candidate_score_gap", "max"),
        )
    )
    for thr in thresholds:
        col = f"has_good_candidate_{str(thr).replace('.', '_')}"
        rate = (
            prompt_df.groupby("base_meter", as_index=False)[col]
            .mean()
            .rename(columns={col: f"prompt_has_good_candidate_{str(thr).replace('.', '_')}_rate"})
        )
        prompt_agg = prompt_agg.merge(rate, on="base_meter", how="left")

    summary = candidate_agg.merge(prompt_agg, on="base_meter", how="left")
    summary = summary.sort_values(["candidate_mean_score", "candidate_median_score"], ascending=[False, False]).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary["suggested_action"] = summary.apply(decide_action, axis=1)
    return summary


def plot_ranked_bar(df: pd.DataFrame, value_col: str, title: str, ylabel: str, out_path: Path, color: str) -> None:
    ordered = df.sort_values(value_col, ascending=False)
    plt.figure(figsize=(14, 7))
    plt.bar(ordered["base_meter"], ordered[value_col], color=color)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grouped_rates(df: pd.DataFrame, cols: list[str], labels: list[str], title: str, out_path: Path) -> None:
    width = 0.22
    x = np.arange(len(df))
    plt.figure(figsize=(15, 7))
    colors = ["#7db7ff", "#2f5aa8", "#0f2b5b"]
    for idx, (col, label) in enumerate(zip(cols, labels)):
        plt.bar(x + (idx - 1) * width, df[col], width=width, label=label, color=colors[idx])
    plt.xticks(x, df["base_meter"], rotation=90)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction of prompts")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prompt_signal(df: pd.DataFrame, out_path: Path) -> None:
    width = 0.25
    x = np.arange(len(df))
    plt.figure(figsize=(15, 7))
    plt.bar(x - width, df["prompt_best_score_mean"], width=width, label="best candidate mean", color="#1b9e77")
    plt.bar(x, df["prompt_mean_score_mean"], width=width, label="candidate mean mean", color="#7570b3")
    plt.bar(x + width, df["prompt_worst_score_mean"], width=width, label="worst candidate mean", color="#d95f02")
    plt.xticks(x, df["base_meter"], rotation=90)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Prompt-Level Candidate Quality by Base Meter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_min_mean_max(df: pd.DataFrame, out_path: Path) -> None:
    width = 0.25
    x = np.arange(len(df))
    plt.figure(figsize=(15, 7))
    plt.bar(x - width, df["candidate_min_score"], width=width, label="candidate min", color="#d95f02")
    plt.bar(x, df["candidate_mean_score"], width=width, label="candidate mean", color="#2f5aa8")
    plt.bar(x + width, df["candidate_max_score"], width=width, label="candidate max", color="#1b9e77")
    plt.xticks(x, df["base_meter"], rotation=90)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Candidate Min / Mean / Max by Base Meter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_boxplot_by_base(results_df: pd.DataFrame, ordered_bases: list[str], out_path: Path) -> None:
    data = [results_df.loc[results_df["base_meter"] == meter, "meter_score"].tolist() for meter in ordered_bases]
    plt.figure(figsize=(15, 7))
    plt.boxplot(data, tick_labels=ordered_bases, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Candidate score")
    plt.title("Candidate Score Distribution by Base Meter")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(values: pd.Series, title: str, xlabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=25, color="#2f5aa8", edgecolor="black", alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prompt_candidate_ranges(prompt_df: pd.DataFrame, title: str, out_path: Path) -> None:
    ordered = prompt_df.sort_values("prompt_rank_within_base").reset_index(drop=True)
    x = np.arange(len(ordered))
    plt.figure(figsize=(12, 6))
    plt.vlines(x, ordered["candidate_min_score"], ordered["candidate_max_score"], color="#2f5aa8", linewidth=2)
    plt.scatter(x, ordered["candidate_max_score"], color="#1b9e77", label="best", s=60, zorder=3)
    plt.scatter(x, ordered["candidate_mean_score"], color="#7570b3", label="mean", s=50, zorder=3)
    plt.scatter(x, ordered["candidate_min_score"], color="#d95f02", label="worst", s=60, zorder=3)
    plt.xticks(x, [f"P{i+1}" for i in range(len(ordered))])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_top_level_readme(output_root: Path, report: dict, summary_df: pd.DataFrame) -> None:
    keep_list = summary_df.loc[summary_df["suggested_action"] == "keep", "base_meter"].tolist()
    weak_list = summary_df.loc[summary_df["suggested_action"] == "keep_but_flag_weak", "base_meter"].tolist()
    drop_list = summary_df.loc[
        summary_df["suggested_action"] == "candidate_for_removal_or_extra_filtering", "base_meter"
    ].tolist()
    strongest = summary_df.head(5)[["base_meter", "candidate_mean_score"]].values.tolist()
    weakest = summary_df.tail(5)[["base_meter", "candidate_mean_score"]].values.tolist()
    lines = [
        "# Base-Only Sanity Check Model",
        "",
        "This run is intentionally base-meter only in its primary outputs.",
        "Prompts may come from rows whose requested label is a variant, but the reward target is always the base meter.",
        "",
        "## Run Setup",
        "",
        f"- Backend: `{report['generation_backend']}`",
        f"- Dataset: `{report['dataset_id']}`",
        f"- Base meters evaluated: `{report['num_base_meters']}`",
        f"- Prompts per base meter (`n`): `{report['n_prompts_per_base_meter']}`",
        f"- Generations per prompt (`k`): `{report['k_generations']}`",
        f"- Total generated candidates: `{report['num_generations_total']}`",
        "",
        "## What The Main Files Mean",
        "",
        "- `base_meter_summary.csv`: main decision table, one row per base meter.",
        "- `all_generations.jsonl`: one JSON row per generated candidate across the whole run.",
        "- `all_prompt_groups.jsonl`: one JSON row per prompt group, summarizing the `k` candidates for that prompt.",
        "- `per_meter/`: one folder per base meter with JSONL, CSV, plots, and README.",
        "",
        "## How To Interpret The Prompt-Level Columns",
        "",
        "- `prompt_best_score_mean`: on average, if you sample `k` candidates for one prompt, how good is the best one.",
        "- `prompt_worst_score_mean`: on average, how bad is the worst candidate among the `k` samples.",
        "- `prompt_gap_mean`: average gap between best and worst candidate for a prompt. This is a useful proxy for whether GRPO can rank candidates and get a learning signal.",
        "- `prompt_has_good_candidate_0_5_rate`, `0_7_rate`, `0_9_rate`: fraction of prompts where at least one of the `k` candidates crossed that threshold.",
        "",
        "## Why This Is More Relevant For GRPO",
        "",
        "- GRPO compares candidates generated for the same prompt.",
        "- A meter can have a modest candidate average but still be promising for GRPO if `prompt_best_score_mean` is decent and `prompt_gap_mean` is healthy.",
        "- A meter is much more concerning when even the best candidate per prompt stays weak.",
        "",
        "## Analysis",
        "",
        f"- Overall candidate mean score: `{report['overall_candidate_mean_score']:.4f}`",
        f"- Overall candidate median score: `{report['overall_candidate_median_score']:.4f}`",
        f"- Overall prompt-best mean score: `{report['overall_prompt_best_mean_score']:.4f}`",
        f"- Overall prompt-gap mean: `{report['overall_prompt_gap_mean']:.4f}`",
        "",
        "### Strongest Base Meters By Candidate Mean",
        "",
    ]
    for meter, score in strongest:
        lines.append(f"- `{meter}`: `{score:.4f}`")
    lines += [
        "",
        "### Weakest Base Meters By Candidate Mean",
        "",
    ]
    for meter, score in weakest:
        lines.append(f"- `{meter}`: `{score:.4f}`")
    lines += [
        "",
        "### Heuristic Review Buckets",
        "",
        f"- `keep`: {', '.join(keep_list) if keep_list else 'none'}",
        f"- `keep_but_flag_weak`: {', '.join(weak_list) if weak_list else 'none'}",
        f"- `candidate_for_removal_or_extra_filtering`: {', '.join(drop_list) if drop_list else 'none'}",
        "",
        "These buckets are only decision aids. They are not automatic rules.",
        "",
        "## Variant Note",
        "",
        "- Primary plots and summaries intentionally hide requested variants.",
        "- If later you want to inspect whether one requested variant is dragging a base meter down, the per-candidate JSONL still retains `requested_meter_label` for debugging.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_meter_readme(meter_dir: Path, row: pd.Series, report: dict) -> None:
    lines = [
        f"# {row['base_meter']}",
        "",
        "This folder contains the base-only sanity-check view for this base meter.",
        "",
        "## Headline Stats",
        "",
        f"- Candidate mean score: `{row['candidate_mean_score']:.4f}`",
        f"- Candidate median score: `{row['candidate_median_score']:.4f}`",
        f"- Candidate std score: `{row['candidate_std_score']:.4f}`",
        f"- Candidate min score: `{row['candidate_min_score']:.4f}`",
        f"- Candidate max score: `{row['candidate_max_score']:.4f}`",
        f"- Prompt best score mean: `{row['prompt_best_score_mean']:.4f}`",
        f"- Prompt best score min/max: `{row['prompt_best_score_min']:.4f}` / `{row['prompt_best_score_max']:.4f}`",
        f"- Prompt mean score mean: `{row['prompt_mean_score_mean']:.4f}`",
        f"- Prompt worst score mean: `{row['prompt_worst_score_mean']:.4f}`",
        f"- Prompt gap mean: `{row['prompt_gap_mean']:.4f}`",
        f"- Prompts with a candidate >= 0.5: `{row['prompt_has_good_candidate_0_5_rate']:.4f}`",
        f"- Prompts with a candidate >= 0.7: `{row['prompt_has_good_candidate_0_7_rate']:.4f}`",
        f"- Prompts with a candidate >= 0.9: `{row['prompt_has_good_candidate_0_9_rate']:.4f}`",
        f"- Suggested action: `{row['suggested_action']}`",
        "",
        "## How To Read The Files",
        "",
        "- `generations.jsonl`: every generated candidate with score details and per-bayt metadata.",
        "- `prompt_groups.jsonl`: one row per prompt, summarizing the `k` candidates for that prompt.",
        "- `generations.csv`: candidate table without the heavy nested per-bayt details.",
        "- `prompt_groups.csv`: CSV version of the prompt-group summaries.",
        "- `examples_best.csv`: top 5 candidates by score for this meter.",
        "- `examples_worst.csv`: bottom 5 candidates by score for this meter.",
        "- `plots/candidate_score_histogram.png`: score distribution across all candidates.",
        "- `plots/prompt_candidate_ranges.png`: for each of the `n` prompts, shows worst/mean/best across the `k` candidates.",
        "",
        "## GRPO Interpretation",
        "",
        "- If `prompt_best_score_mean` is decent, then sampling multiple candidates often finds at least one acceptable poem.",
        "- If `prompt_gap_mean` is healthy, GRPO can often prefer one candidate over another and get a ranking signal.",
        "- If both are low, then even sampling `k` times does not reliably produce a good candidate, and RL will be much harder.",
    ]
    (meter_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    cfg = load_yaml("grpo_config.yaml")
    study_cfg = cfg["studies"]["sanity_check_model"]
    generation_backend = resolve_generation_backend(cfg)

    out_root = Path(study_cfg["output_root"])
    ensure_dir(out_root)
    plots_root = ensure_dir(out_root / "plots")
    per_meter_root = ensure_dir(out_root / "per_meter")
    logger = setup_logger("sanity_check_model_base_only", out_root / "sanity_check_model.log")

    dataset_id = get_dataset_id_for_phase()
    max_bayts = os.getenv("PHASE1_MAX_BAYTS", "").strip() or None
    allowed_meters = parse_allowed_meters_env()
    ds = load_and_prepare_dataset(
        dataset_id=dataset_id,
        split="train",
        max_bayts=max_bayts,
        allowed_meters=allowed_meters or None,
        hf_token=os.getenv("HF_TOKEN", "").strip() or None,
    )
    df = ds.to_pandas()
    base_meters = sorted(df["base_meter"].dropna().unique().tolist())
    thresholds = study_cfg["thresholds"]

    logger.info(f"dataset_id={dataset_id}")
    logger.info(f"num_rows_after_filter={len(ds)}")
    logger.info(f"generation_backend={generation_backend}")
    logger.info(f"num_base_meters={len(base_meters)}")
    logger.info(f"n_prompts_per_base_meter={study_cfg['n_prompts_per_meter']}")
    logger.info(f"k_generations={study_cfg['k_generations']}")

    tokenizer = None
    model = None
    llm = None
    lora_request = None
    if generation_backend == "vllm":
        llm, lora_request, adapter_meta = build_vllm_engine(cfg)
        logger.info(f"adapter_resolution={json.dumps(adapter_meta, ensure_ascii=False)}")
    elif generation_backend == "transformers":
        tokenizer, model = build_transformers_model()
    else:
        raise ValueError(f"Unsupported generation backend: {generation_backend}")

    random.seed(int(study_cfg["seed"]))

    all_generations_jsonl = out_root / "all_generations.jsonl"
    all_prompt_groups_jsonl = out_root / "all_prompt_groups.jsonl"
    if all_generations_jsonl.exists():
        all_generations_jsonl.unlink()
    if all_prompt_groups_jsonl.exists():
        all_prompt_groups_jsonl.unlink()

    rows = []
    prompt_rows = []

    for meter_index, base_meter in enumerate(base_meters):
        meter_df = df[df["base_meter"] == base_meter].copy()
        sample_n = min(int(study_cfg["n_prompts_per_meter"]), len(meter_df))
        sampled = meter_df.sample(sample_n, random_state=int(study_cfg["seed"]) + meter_index).reset_index(drop=True)
        meter_dir = ensure_dir(per_meter_root / safe_filename(base_meter))
        meter_plots_dir = ensure_dir(meter_dir / "plots")
        meter_generations_jsonl = meter_dir / "generations.jsonl"
        meter_prompt_groups_jsonl = meter_dir / "prompt_groups.jsonl"
        if meter_generations_jsonl.exists():
            meter_generations_jsonl.unlink()
        if meter_prompt_groups_jsonl.exists():
            meter_prompt_groups_jsonl.unlink()

        logger.info(f"running base_meter={base_meter} prompts={sample_n}")

        for prompt_rank, row in sampled.iterrows():
            prompt = row["prompt"]
            requested_label = row["meter_label"]
            prompt_group_id = f"{base_meter}::{int(row['source_index'])}::{prompt_rank + 1}"

            if generation_backend == "transformers":
                gens = generate_k_transformers(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    k=int(study_cfg["k_generations"]),
                    max_new_tokens=int(study_cfg["max_new_tokens"]),
                    temperature=float(study_cfg["temperature"]),
                    top_p=float(study_cfg["top_p"]),
                )
            else:
                gens = generate_k_vllm(
                    llm=llm,
                    lora_request=lora_request,
                    prompt=prompt,
                    k=int(study_cfg["k_generations"]),
                    max_new_tokens=int(study_cfg["max_new_tokens"]),
                    temperature=float(study_cfg["temperature"]),
                    top_p=float(study_cfg["top_p"]),
                )

            prompt_candidate_rows = []
            for generation_index, generated_text in enumerate(gens):
                meter_out = score_meter_poem(
                    generated_text,
                    target_meter=base_meter,
                    base_meter=base_meter,
                    aggregator="logmean",
                )
                jsonl_rec = {
                    "base_meter": base_meter,
                    "requested_meter_label": requested_label,
                    "reward_target_meter": meter_out["target_meter_used"],
                    "target_resolution": meter_out["target_resolution"],
                    "source_index": int(row["source_index"]),
                    "prompt_group_id": prompt_group_id,
                    "prompt_rank_within_base": int(prompt_rank + 1),
                    "generation_index": int(generation_index),
                    "requested_bayts": int(row["requested_bayts"]),
                    "requested_lines": int(row["requested_lines"]),
                    "prompt_has_odd_tail": bool(row["has_odd_tail"]),
                    "meter_score": float(meter_out["score"]),
                    "reward_mean_bayt_score": float(meter_out["mean_score"]),
                    "reward_logmean_bayt_score": float(meter_out["logmean_score"]),
                    "reward_std_bayt_score": float(meter_out["std_score"]),
                    "reward_min_bayt_score": float(meter_out["min_score"]),
                    "reward_max_bayt_score": float(meter_out["max_score"]),
                    "num_scored_bayts": int(meter_out["num_valid_bayts"]),
                    "num_unscored_bayts": int(meter_out["num_skipped_bayts"]),
                    "num_output_lines": int(meter_out["num_lines"]),
                    "num_output_line_pairs": int(meter_out["complete_bayts"]),
                    "has_unpaired_last_line": bool(meter_out["has_odd_tail"]),
                    "unpaired_last_line_text": meter_out["odd_tail_line"],
                    "per_bayt_scores": meter_out["per_bayt_scores"],
                    "per_bayt_details": meter_out["per_bayt_details"],
                    "prompt_preview": prompt[:220],
                    "generated_text": generated_text,
                }
                add_threshold_flags(jsonl_rec, thresholds, float(meter_out["score"]))
                append_jsonl(all_generations_jsonl, jsonl_rec)
                append_jsonl(meter_generations_jsonl, jsonl_rec)

                flat_rec = {k: v for k, v in jsonl_rec.items() if k not in {"per_bayt_scores", "per_bayt_details"}}
                rows.append(flat_rec)
                prompt_candidate_rows.append(flat_rec)

            prompt_summary = summarize_prompt_group(prompt_candidate_rows, thresholds)
            append_jsonl(all_prompt_groups_jsonl, prompt_summary)
            append_jsonl(meter_prompt_groups_jsonl, prompt_summary)
            prompt_rows.append(prompt_summary)

        logger.info(f"finished base_meter={base_meter}")

    results_df = pd.DataFrame(rows)
    prompt_df = pd.DataFrame(prompt_rows)
    results_df.to_csv(out_root / "all_generations.csv", index=False)
    prompt_df.to_csv(out_root / "all_prompt_groups.csv", index=False)

    if len(results_df) == 0:
        logger.warning("No results generated.")
        return

    summary_df = build_base_summary(results_df, prompt_df, thresholds)
    summary_df.to_csv(out_root / "base_meter_summary.csv", index=False)

    lowest_generations = results_df.sort_values("meter_score", ascending=True).head(50)
    highest_generations = results_df.sort_values("meter_score", ascending=False).head(50)
    lowest_generations.to_csv(out_root / "lowest_generations.csv", index=False)
    highest_generations.to_csv(out_root / "highest_generations.csv", index=False)

    report = {
        "generation_backend": generation_backend,
        "dataset_id": dataset_id,
        "dataset_rows_after_filter": int(len(ds)),
        "num_base_meters": int(len(base_meters)),
        "n_prompts_per_base_meter": int(study_cfg["n_prompts_per_meter"]),
        "k_generations": int(study_cfg["k_generations"]),
        "num_generations_total": int(len(results_df)),
        "overall_candidate_mean_score": float(results_df["meter_score"].mean()),
        "overall_candidate_median_score": float(results_df["meter_score"].median()),
        "overall_candidate_std_score": float(results_df["meter_score"].std()),
        "overall_candidate_min_score": float(results_df["meter_score"].min()),
        "overall_candidate_max_score": float(results_df["meter_score"].max()),
        "overall_prompt_best_mean_score": float(prompt_df["candidate_max_score"].mean()),
        "overall_prompt_worst_mean_score": float(prompt_df["candidate_min_score"].mean()),
        "overall_prompt_gap_mean": float(prompt_df["candidate_score_gap"].mean()),
        "thresholds": thresholds,
        "suggested_keep": summary_df.loc[summary_df["suggested_action"] == "keep", "base_meter"].tolist(),
        "suggested_keep_but_flag_weak": summary_df.loc[
            summary_df["suggested_action"] == "keep_but_flag_weak", "base_meter"
        ].tolist(),
        "suggested_candidate_for_removal_or_extra_filtering": summary_df.loc[
            summary_df["suggested_action"] == "candidate_for_removal_or_extra_filtering", "base_meter"
        ].tolist(),
    }
    save_json(report, out_root / "report.json")

    ordered_bases = summary_df["base_meter"].tolist()
    plot_ranked_bar(summary_df, "candidate_mean_score", "Candidate Mean Score by Base Meter", "Mean score", plots_root / "base_meter_mean_score.png", "#2f5aa8")
    plot_grouped_rates(
        summary_df,
        cols=[
            "prompt_has_good_candidate_0_5_rate",
            "prompt_has_good_candidate_0_7_rate",
            "prompt_has_good_candidate_0_9_rate",
        ],
        labels=["prompt has >=0.5", "prompt has >=0.7", "prompt has >=0.9"],
        title="Chance of At Least One Good Candidate When Sampling k Candidates",
        out_path=plots_root / "base_meter_good_candidate_rates.png",
    )
    plot_prompt_signal(summary_df, plots_root / "base_meter_prompt_signal.png")
    plot_min_mean_max(summary_df, plots_root / "base_meter_candidate_min_mean_max.png")
    plot_boxplot_by_base(results_df, ordered_bases, plots_root / "base_meter_candidate_boxplot.png")
    plot_histogram(results_df["meter_score"], "Overall Candidate Score Distribution", "Candidate score", plots_root / "overall_candidate_score_histogram.png")

    for _, summary_row in summary_df.iterrows():
        base_meter = summary_row["base_meter"]
        meter_dir = ensure_dir(per_meter_root / safe_filename(base_meter))
        meter_plots_dir = ensure_dir(meter_dir / "plots")
        meter_results = results_df[results_df["base_meter"] == base_meter].copy()
        meter_prompts = prompt_df[prompt_df["base_meter"] == base_meter].copy().sort_values("prompt_rank_within_base")

        meter_results.to_csv(meter_dir / "generations.csv", index=False)
        meter_prompts.to_csv(meter_dir / "prompt_groups.csv", index=False)
        meter_results.sort_values("meter_score", ascending=False).head(5).to_csv(meter_dir / "examples_best.csv", index=False)
        meter_results.sort_values("meter_score", ascending=True).head(5).to_csv(meter_dir / "examples_worst.csv", index=False)
        save_json(summary_row.to_dict(), meter_dir / "summary.json")
        write_meter_readme(meter_dir, summary_row, report)

        plot_histogram(
            meter_results["meter_score"],
            f"Candidate Score Distribution: {base_meter}",
            "Candidate score",
            meter_plots_dir / "candidate_score_histogram.png",
        )
        plot_prompt_candidate_ranges(
            meter_prompts,
            f"Prompt Candidate Ranges (k={int(study_cfg['k_generations'])}): {base_meter}",
            meter_plots_dir / "prompt_candidate_ranges.png",
        )

    write_top_level_readme(out_root, report, summary_df)

    logger.info(f"overall_candidate_mean_score={report['overall_candidate_mean_score']:.4f}")
    logger.info(f"overall_prompt_best_mean_score={report['overall_prompt_best_mean_score']:.4f}")
    logger.info(f"overall_prompt_gap_mean={report['overall_prompt_gap_mean']:.4f}")
    logger.info(f"saved_plots_root={plots_root}")
    logger.info(f"saved_per_meter_root={per_meter_root}")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
