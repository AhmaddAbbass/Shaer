import json
import os
import random
import re
import sys
from pathlib import Path

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
except Exception:  # pragma: no cover - fallback for environments without vLLM
    LLM = None
    SamplingParams = None
    LoRARequest = None

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rewards.common import (
    ensure_dir,
    get_dataset_id_for_phase,
    load_and_prepare_dataset,
    load_yaml,
    parse_allowed_meters_env,
    poem_structure,
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


def safe_filename(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "meter"


def flatten_columns(df):
    out = df.copy()
    out.columns = ["_".join(str(x) for x in col if str(x)).strip("_") for col in out.columns]
    return out


def plot_ranked_bar(summary, value_col, title, ylabel, path, ascending=False, color="#1f77b4"):
    data = summary.sort_values(value_col, ascending=ascending)
    plt.figure(figsize=(14, 7))
    plt.bar(data["meter_label"], data[value_col], color=color)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_histogram(values, title, xlabel, path, bins=20, color="#1f77b4"):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_ecdf(values, title, xlabel, path, color="#1f77b4"):
    xs = np.sort(np.asarray(values, dtype=float))
    ys = np.arange(1, len(xs) + 1) / len(xs) if len(xs) else np.asarray([])
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, color=color, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("ECDF")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_boxplot_by_meter(results, path):
    ordered = (
        results.groupby("meter_label")["meter_reward"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    data = [results.loc[results["meter_label"] == meter, "meter_reward"].tolist() for meter in ordered]
    plt.figure(figsize=(15, 7))
    plt.boxplot(data, tick_labels=ordered, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Meter Reward")
    plt.title("Meter Reward Distribution by Meter")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_heatmap(matrix_df, title, path, cmap="viridis", vmin=None, vmax=None):
    values = matrix_df.to_numpy(dtype=float)
    plt.figure(figsize=(12, max(6, 0.4 * len(matrix_df.index))))
    plt.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xticks(range(len(matrix_df.columns)), matrix_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix_df.index)), matrix_df.index)
    plt.title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter(summary, x_col, y_col, title, xlabel, ylabel, path, color="#1f77b4"):
    plt.figure(figsize=(10, 7))
    plt.scatter(summary[x_col], summary[y_col], s=70, color=color, alpha=0.85)
    for _, row in summary.iterrows():
        plt.annotate(row["meter_label"], (row[x_col], row[y_col]), fontsize=8, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_target_resolution_stacked(results, path):
    resolution = pd.crosstab(results["meter_label"], results["target_resolution"], normalize="index")
    resolution = resolution.loc[
        results.groupby("meter_label")["meter_reward"].mean().sort_values(ascending=False).index
    ]
    ax = resolution.plot(kind="bar", stacked=True, figsize=(14, 7), colormap="tab20")
    ax.set_ylabel("Fraction")
    ax.set_title("Target Resolution Mix by Meter")
    ax.set_xlabel("Meter")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_per_meter_pages(results, summary, per_meter_dir):
    summary_lookup = summary.set_index("meter_label").to_dict(orient="index")

    ordered = (
        results.groupby("meter_label")["meter_reward"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for rank, meter in enumerate(ordered, 1):
        sub = results[results["meter_label"] == meter].copy()
        stats = summary_lookup[meter]
        safe_meter = safe_filename(meter)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        axes[0, 0].hist(sub["meter_reward"], bins=min(12, max(4, len(sub))), color="#1f77b4", edgecolor="black")
        axes[0, 0].set_title("Score Histogram")
        axes[0, 0].set_xlabel("Meter Reward")
        axes[0, 0].set_ylabel("Count")

        sorted_scores = np.sort(sub["meter_reward"].to_numpy(dtype=float))
        axes[0, 1].plot(sorted_scores, marker="o", linewidth=2, color="#ff7f0e")
        for thr in [0.3, 0.5, 0.7, 0.9]:
            axes[0, 1].axhline(thr, linestyle="--", alpha=0.25, color="gray")
        axes[0, 1].set_title("Sorted Generation Scores")
        axes[0, 1].set_xlabel("Generation Rank")
        axes[0, 1].set_ylabel("Meter Reward")

        axes[1, 0].hist(
            sub["generated_complete_bayts"],
            bins=min(12, max(3, int(sub["generated_complete_bayts"].nunique()))),
            color="#2ca02c",
            edgecolor="black",
        )
        axes[1, 0].set_title("Generated Complete Bayts")
        axes[1, 0].set_xlabel("Complete Bayts")
        axes[1, 0].set_ylabel("Count")

        axes[1, 1].axis("off")
        text_lines = [
            f"Meter: {meter}",
            f"Rank by mean score: {rank}/{len(ordered)}",
            f"Mean: {stats['meter_reward_mean']:.4f}",
            f"Median: {stats['meter_reward_median']:.4f}",
            f"Std across generations: {stats['meter_reward_std']:.4f}",
            f"Min / Max: {stats['meter_reward_min']:.4f} / {stats['meter_reward_max']:.4f}",
            f"Generations: {int(stats['meter_reward_count'])}",
            f"Avg valid bayts: {stats['num_valid_bayts_mean']:.2f}",
            f"Avg skipped bayts: {stats['num_skipped_bayts_mean']:.2f}",
            f"Odd-tail fraction: {stats['has_odd_tail_mean']:.2f}",
            f"Resolution base-only / fallback: {stats.get('target_resolution_base_meter_only', 0.0):.2f} / {stats.get('target_resolution_requested_meter_no_base', 0.0):.2f}",
        ]
        y = 0.95
        for line in text_lines:
            axes[1, 1].text(0.02, y, line, fontsize=11, va="top")
            y -= 0.08

        fig.suptitle(f"Per-Meter Study Page: {meter}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(per_meter_dir / f"{rank:02d}_{safe_meter}_overview.png")
        plt.close(fig)

        top_examples = sub.sort_values("meter_reward", ascending=False).head(3)
        bottom_examples = sub.sort_values("meter_reward", ascending=True).head(3)
        examples = pd.concat(
            [
                top_examples.assign(example_bucket="top"),
                bottom_examples.assign(example_bucket="bottom"),
            ],
            ignore_index=True,
        )
        examples.to_csv(per_meter_dir / f"{rank:02d}_{safe_meter}_examples.csv", index=False)


def write_study_notes(out_root, overview, weak, strong):
    lines = [
        "# Sanity Check Model Outputs",
        "",
        "This folder contains model-generation sanity outputs for the current base + SFT adapter policy.",
        "",
        "## Quick Read",
        "",
        f"- Generation backend: `{overview['generation_backend']}`",
        f"- Dataset: `{overview['dataset_id']}`",
        f"- Rows after filter: `{overview['dataset_rows_after_filter']}`",
        f"- Meters studied: `{overview['num_meters']}`",
        f"- Generated samples: `{overview['num_generations_total']}`",
        f"- Mean reward overall: `{overview['overall_mean_reward']:.4f}`",
        f"- Median reward overall: `{overview['overall_median_reward']:.4f}`",
        f"- Weak meters by current rule (`mean < 0.35`): `{', '.join(weak) if weak else 'none'}`",
        f"- Strongest meters by mean: `{', '.join(strong) if strong else 'n/a'}`",
        "",
        "## Useful Files",
        "",
        "- `all_generations.csv`: one row per generated poem",
        "- `sampled_prompts.csv`: which prompts were sampled per meter",
        "- `meter_summary.csv`: main per-meter table",
        "- `target_resolution_summary.csv`: base-meter-only target usage summary",
        "- `report.json`: compact machine-readable overview",
        "- `plots/`: overall plots",
        "- `plots/per_meter/`: one figure and one examples CSV per meter",
        "",
        "## Suggested Group/Paper Materials",
        "",
        "- `plots/avg_meter_reward_by_meter.png`",
        "- `plots/meter_reward_boxplot_by_meter.png`",
        "- `plots/threshold_heatmap_by_meter.png`",
        "- `plots/target_resolution_by_meter.png`",
        "- `plots/per_meter/` figures for appendix or group review",
        "",
    ]
    with open(out_root / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    cfg = load_yaml("grpo_config.yaml")
    study_cfg = cfg["studies"]["sanity_check_model"]
    generation_backend = resolve_generation_backend(cfg)

    out_root = Path(study_cfg["output_root"])
    plots_root = ensure_dir(out_root / "plots")
    per_meter_dir = ensure_dir(plots_root / "per_meter")
    ensure_dir(out_root)
    logger = setup_logger("sanity_check_model", out_root / "sanity_check_model.log")

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

    logger.info(f"dataset_id={dataset_id}")
    logger.info(f"num_rows_after_filter={len(ds)}")
    logger.info(f"generation_backend={generation_backend}")

    df = ds.to_pandas()
    meters = sorted(df["meter_label"].dropna().unique().tolist())
    logger.info(f"num_meters={len(meters)}")

    tokenizer = None
    model = None
    llm = None
    lora_request = None
    adapter_meta = {}
    if generation_backend == "vllm":
        llm, lora_request, adapter_meta = build_vllm_engine(cfg)
        logger.info(f"adapter_resolution={json.dumps(adapter_meta, ensure_ascii=False)}")
    elif generation_backend == "transformers":
        tokenizer, model = build_transformers_model()
    else:
        raise ValueError(f"Unsupported generation backend: {generation_backend}")

    random.seed(study_cfg["seed"])

    rows = []
    sampled_prompt_rows = []
    thresholds = study_cfg["thresholds"]

    for meter in meters:
        meter_df = df[df["meter_label"] == meter]
        if len(meter_df) == 0:
            continue

        sample_n = min(study_cfg["n_prompts_per_meter"], len(meter_df))
        sampled = meter_df.sample(sample_n, random_state=study_cfg["seed"])

        logger.info(f"running meter={meter} prompts={sample_n}")

        for prompt_rank, (_, row) in enumerate(sampled.iterrows(), 1):
            prompt = row["prompt"]
            sampled_prompt_rows.append({
                "meter_label": meter,
                "base_meter": row["base_meter"],
                "source_index": int(row["source_index"]),
                "requested_bayts": int(row["requested_bayts"]),
                "requested_lines": int(row["requested_lines"]),
                "has_odd_tail": bool(row["has_odd_tail"]),
                "prompt_rank_within_meter": prompt_rank,
                "prompt_preview": prompt[:400],
            })

            if generation_backend == "transformers":
                gens = generate_k_transformers(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                    k=study_cfg["k_generations"],
                    max_new_tokens=study_cfg["max_new_tokens"],
                    temperature=study_cfg["temperature"],
                    top_p=study_cfg["top_p"],
                )
            else:
                gens = generate_k_vllm(
                    llm=llm,
                    lora_request=lora_request,
                    prompt=prompt,
                    k=study_cfg["k_generations"],
                    max_new_tokens=study_cfg["max_new_tokens"],
                    temperature=study_cfg["temperature"],
                    top_p=study_cfg["top_p"],
                )

            for j, g in enumerate(gens):
                struct = poem_structure(g)
                meter_out = score_meter_poem(
                    g,
                    target_meter=meter,
                    base_meter=row["base_meter"],
                    aggregator="logmean",
                )
                rec = {
                    "meter_label": meter,
                    "base_meter": row["base_meter"],
                    "target_meter_used": meter_out["target_meter_used"],
                    "target_resolution": meter_out["target_resolution"],
                    "source_index": int(row["source_index"]),
                    "prompt_rank_within_meter": prompt_rank,
                    "generation_index": j,
                    "requested_bayts": int(row["requested_bayts"]),
                    "requested_lines": int(row["requested_lines"]),
                    "prompt_has_odd_tail": bool(row["has_odd_tail"]),
                    "meter_reward": meter_out["score"],
                    "meter_reward_mean": meter_out["mean_score"],
                    "meter_reward_logmean": meter_out["logmean_score"],
                    "meter_reward_std": meter_out["std_score"],
                    "meter_reward_min": meter_out["min_score"],
                    "meter_reward_max": meter_out["max_score"],
                    "num_valid_bayts": meter_out["num_valid_bayts"],
                    "num_skipped_bayts": meter_out["num_skipped_bayts"],
                    "generated_num_lines": meter_out["num_lines"],
                    "generated_complete_bayts": meter_out["complete_bayts"],
                    "has_odd_tail": meter_out["has_odd_tail"],
                    "odd_tail_line": meter_out["odd_tail_line"],
                    "generated_text": g,
                    "prompt_preview": prompt[:220],
                }
                for thr in thresholds:
                    rec[f"above_{str(thr).replace('.', '_')}"] = float(meter_out["score"] >= thr)
                rows.append(rec)

        logger.info(f"finished meter={meter}")

    results = pd.DataFrame(rows)
    sampled_prompts = pd.DataFrame(sampled_prompt_rows)
    results.to_csv(out_root / "all_generations.csv", index=False)
    sampled_prompts.to_csv(out_root / "sampled_prompts.csv", index=False)

    if len(results) == 0:
        logger.warning("No results generated.")
        return

    agg_dict = {
        "meter_reward": ["mean", "median", "std", "min", "max", "count"],
        "meter_reward_std": ["mean", "median"],
        "num_valid_bayts": ["mean", "median"],
        "num_skipped_bayts": ["mean", "median"],
        "generated_num_lines": ["mean", "median"],
        "generated_complete_bayts": ["mean", "median"],
        "has_odd_tail": ["mean"],
    }
    for thr in thresholds:
        agg_dict[f"above_{str(thr).replace('.', '_')}"] = ["mean"]

    summary = results.groupby("meter_label").agg(agg_dict)
    summary = flatten_columns(summary).reset_index()

    resolution_summary = pd.crosstab(results["meter_label"], results["target_resolution"], normalize="index")
    resolution_summary = resolution_summary.add_prefix("target_resolution_").reset_index()
    resolution_summary.to_csv(out_root / "target_resolution_summary.csv", index=False)

    summary = summary.merge(resolution_summary, on="meter_label", how="left").fillna(0.0)
    summary["score_range"] = summary["meter_reward_max"] - summary["meter_reward_min"]
    summary = summary.sort_values("meter_reward_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(out_root / "meter_summary.csv", index=False)

    weak = summary[summary["meter_reward_mean"] < 0.35]["meter_label"].tolist()
    strong = summary.head(5)["meter_label"].tolist()

    overall = {
        "generation_backend": generation_backend,
        "dataset_id": dataset_id,
        "dataset_rows_after_filter": int(len(ds)),
        "num_meters": int(summary["meter_label"].nunique()),
        "num_generations_total": int(len(results)),
        "overall_mean_reward": float(results["meter_reward"].mean()),
        "overall_median_reward": float(results["meter_reward"].median()),
        "overall_std_reward": float(results["meter_reward"].std()),
        "overall_min_reward": float(results["meter_reward"].min()),
        "overall_max_reward": float(results["meter_reward"].max()),
        "thresholds": thresholds,
        "suggested_weak_meters": weak,
        "strongest_meters_by_mean": strong,
        "target_resolution_counts": results["target_resolution"].value_counts().to_dict(),
    }
    save_json(overall, out_root / "report.json")

    plot_ranked_bar(
        summary,
        value_col="meter_reward_mean",
        title="Average Meter Reward by Meter",
        ylabel="Mean Meter Reward",
        path=plots_root / "avg_meter_reward_by_meter.png",
        color="#1f77b4",
    )
    plot_ranked_bar(
        summary,
        value_col="meter_reward_median",
        title="Median Meter Reward by Meter",
        ylabel="Median Meter Reward",
        path=plots_root / "median_meter_reward_by_meter.png",
        color="#ff7f0e",
    )
    plot_ranked_bar(
        summary,
        value_col="meter_reward_std",
        title="Standard Deviation of Meter Reward by Meter",
        ylabel="Std Meter Reward",
        path=plots_root / "std_meter_reward_by_meter.png",
        ascending=False,
        color="#d62728",
    )
    plot_ranked_bar(
        summary,
        value_col="meter_reward_count",
        title="Number of Generated Samples by Meter",
        ylabel="Sample Count",
        path=plots_root / "samples_per_meter.png",
        color="#2ca02c",
    )
    plot_ranked_bar(
        summary,
        value_col="generated_complete_bayts_mean",
        title="Average Generated Complete Bayts by Meter",
        ylabel="Mean Complete Bayts",
        path=plots_root / "generated_bayts_by_meter.png",
        color="#9467bd",
    )
    plot_ranked_bar(
        summary,
        value_col="generated_num_lines_mean",
        title="Average Generated Lines by Meter",
        ylabel="Mean Generated Lines",
        path=plots_root / "generated_lines_by_meter.png",
        color="#8c564b",
    )
    plot_ranked_bar(
        summary,
        value_col="has_odd_tail_mean",
        title="Fraction of Odd-Tail Generations by Meter",
        ylabel="Odd-Tail Fraction",
        path=plots_root / "odd_tail_fraction_by_meter.png",
        color="#e377c2",
    )

    plot_histogram(
        results["meter_reward"],
        title="Overall Meter Reward Histogram",
        xlabel="Meter Reward",
        path=plots_root / "meter_reward_histogram_overall.png",
        bins=25,
    )
    plot_ecdf(
        results["meter_reward"],
        title="Overall Meter Reward ECDF",
        xlabel="Meter Reward",
        path=plots_root / "meter_reward_ecdf_overall.png",
    )
    plot_boxplot_by_meter(results, plots_root / "meter_reward_boxplot_by_meter.png")

    threshold_cols = [f"above_{str(thr).replace('.', '_')}_mean" for thr in thresholds]
    threshold_matrix = summary.set_index("meter_label")[threshold_cols].copy()
    threshold_matrix.columns = [f">= {thr}" for thr in thresholds]
    plot_heatmap(
        threshold_matrix,
        title="Fraction Above Threshold by Meter",
        path=plots_root / "threshold_heatmap_by_meter.png",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    metric_cols = [
        "meter_reward_mean",
        "meter_reward_median",
        "meter_reward_std",
        "num_valid_bayts_mean",
        "generated_complete_bayts_mean",
    ]
    metric_matrix = summary.set_index("meter_label")[metric_cols].copy()
    metric_matrix.columns = ["mean", "median", "std", "valid_bayts", "gen_bayts"]
    plot_heatmap(
        metric_matrix,
        title="Metric Heatmap by Meter",
        path=plots_root / "metric_heatmap_by_meter.png",
        cmap="magma",
    )

    plot_scatter(
        summary,
        x_col="meter_reward_mean",
        y_col="meter_reward_std",
        title="Mean vs Std Meter Reward",
        xlabel="Mean Meter Reward",
        ylabel="Std Meter Reward",
        path=plots_root / "mean_vs_std_meter_reward.png",
        color="#1f77b4",
    )
    plot_scatter(
        summary,
        x_col="meter_reward_mean",
        y_col="num_valid_bayts_mean",
        title="Mean Meter Reward vs Average Valid Bayts",
        xlabel="Mean Meter Reward",
        ylabel="Avg Valid Bayts",
        path=plots_root / "mean_reward_vs_valid_bayts.png",
        color="#2ca02c",
    )

    plot_target_resolution_stacked(results, plots_root / "target_resolution_by_meter.png")
    plot_per_meter_pages(results, summary, per_meter_dir)
    write_study_notes(out_root, overall, weak, strong)

    lowest_generations = results.sort_values("meter_reward", ascending=True).head(25)
    highest_generations = results.sort_values("meter_reward", ascending=False).head(25)
    lowest_generations.to_csv(out_root / "lowest_generations.csv", index=False)
    highest_generations.to_csv(out_root / "highest_generations.csv", index=False)

    logger.info(f"overall_mean_reward={overall['overall_mean_reward']:.4f}")
    logger.info(f"overall_median_reward={overall['overall_median_reward']:.4f}")
    logger.info(f"suggested_weak_meters={weak}")
    logger.info(f"strongest_meters_by_mean={strong}")
    logger.info(f"saved_plots_root={plots_root}")
    logger.info(f"saved_per_meter_root={per_meter_dir}")

    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
