import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
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

from rewards.common import (
    ensure_dir,
    load_and_prepare_dataset,
    resolve_sft_adapter_path,
    save_json,
    score_arabic_cleanliness,
    score_count_adherence,
    score_repetition_penalty,
)
from rewards.meter import score_meter_poem


ROOT = Path(__file__).resolve().parent


def load_cfg():
    with open(ROOT / "grpo_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(seed: int, *parts) -> int:
    blob = "|".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def sort_key_for_bucket(value: str):
    order = ["1-3", "4-6", "7-10", "11-20", "short_le_8", "full_gt_8"]
    if value in order:
        return (order.index(value), value)
    return (len(order), value)


def choose_balanced_rows(rows, *, per_meter_per_bucket: int, seed: int):
    grouped = {}
    for row in rows:
        key = (str(row.get("base_meter", "")), str(row.get("length_bucket", "")))
        grouped.setdefault(key, []).append(row)

    selected = []
    for (meter, bucket), bucket_rows in sorted(grouped.items(), key=lambda item: (item[0][0], sort_key_for_bucket(item[0][1]))):
        ordered = sorted(
            bucket_rows,
            key=lambda row: (
                stable_hash(seed, meter, bucket, row.get("row_uid", row.get("source_id", row.get("source_index", "")))),
                int(row.get("source_index", 0) or 0),
            ),
        )
        selected.extend(ordered[: min(per_meter_per_bucket, len(ordered))])
    selected.sort(
        key=lambda row: (
            str(row.get("base_meter", "")),
            sort_key_for_bucket(str(row.get("length_bucket", ""))),
            int(row.get("source_index", 0) or 0),
        )
    )
    return selected


def build_vllm_engine(base_model_id: str, adapter_repo: str, *, hf_token: str | None, cfg):
    if LLM is None or SamplingParams is None or LoRARequest is None:
        raise RuntimeError("vLLM is not available in the current environment.")

    gen_cfg = cfg["generation"]
    adapter_path, _ = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token, mode="fresh_sft/train")
    max_prompt_length = int(gen_cfg.get("max_prompt_length", 1024))
    max_completion_length = int(gen_cfg.get("max_completion_length", 512))
    default_max_model_len = max_prompt_length + max_completion_length

    llm = LLM(
        model=base_model_id,
        tokenizer=base_model_id,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gen_cfg.get("vllm_gpu_memory_utilization", 0.5)),
        dtype="bfloat16",
        max_model_len=default_max_model_len,
        trust_remote_code=False,
        hf_token=hf_token,
    )
    lora_request = LoRARequest(
        lora_name=Path(adapter_path).name or "shaer_adapters",
        lora_int_id=1,
        lora_path=adapter_path,
        base_model_name=base_model_id or None,
    )
    return llm, lora_request


def build_transformers_model(base_model_id: str, adapter_repo: str, *, hf_token: str | None):
    adapter_path, _ = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token, mode="fresh_sft/train")
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


def generate_batch_vllm(llm, lora_request, prompts, *, max_new_tokens: int, temperature: float, top_p: float):
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        detokenize=True,
        skip_special_tokens=True,
    )
    outputs = llm.generate(prompts, sampling_params, lora_request=[lora_request] * len(prompts), use_tqdm=False)
    texts = []
    for output in outputs:
        if not output.outputs:
            texts.append("")
        else:
            texts.append(output.outputs[0].text.strip())
    return texts


def generate_batch_transformers(tokenizer, model, prompts, *, max_new_tokens: int, temperature: float, top_p: float):
    texts = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        texts.append(decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip())
    return texts


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values):
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def render_grouped_bar(labels, series_map, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(max(12, 0.8 * len(labels) + 4), 6))
    names = list(series_map.keys())
    width = 0.8 / max(1, len(names))
    xs = list(range(len(labels)))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    for idx, name in enumerate(names):
        vals = series_map[name]
        offset = (idx - (len(names) - 1) / 2) * width
        ax.bar([x + offset for x in xs], vals, width=width, label=name, color=colors[idx % len(colors)])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(6, len(names)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_histograms(values_map, output_path: Path, title: str):
    keys = list(values_map.keys())
    ncols = 2
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.4 * nrows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    for ax, key, color in zip(axes, keys, colors * 10):
        vals = [float(v) for v in values_map[key]]
        ax.hist(vals, bins=12, color=color, alpha=0.85, edgecolor="white")
        ax.set_xlim(0.0, 1.02)
        ax.set_title(key)
        ax.grid(True, axis="y", alpha=0.2)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_poem_markdown(best_rows, output_path: Path, title: str):
    lines = [f"# {title}", ""]
    for meter in sorted(best_rows):
        row = best_rows[meter]
        lines.append(f"## {meter}")
        lines.append("")
        lines.append(f"- total: `{float(row.get('reward_total', 0.0)):.4f}`")
        lines.append(f"- meter: `{float(row.get('reward_meter', 0.0)):.4f}`")
        lines.append(f"- count: `{float(row.get('reward_count_adherence', 0.0)):.4f}`")
        lines.append(f"- repeat: `{float(row.get('reward_repeat_penalty', 0.0)):.4f}`")
        lines.append("")
        if str(row.get("description", "")).strip():
            lines.append(f"- description: `{str(row.get('description', '')).strip()[:220]}`")
            lines.append("")
        lines.append("```text")
        lines.append(str(row.get("completion_text", "")).strip())
        lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build a pre-GRPO baseline eval bundle for Shaer-adapters on the GRPO eval bank.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env", override=False)
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    base_model_id = os.getenv("BASE_MODEL_ID", "").strip() or "Navid-AI/Yehia-7B-preview"
    adapter_repo = "Shaer-AI/Shaer-adapters"

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else run_dir / "plots_before"
    ensure_dir(output_dir)

    cfg = load_cfg()
    effective = json.loads((run_dir / "effective_runtime_config.json").read_text(encoding="utf-8"))
    dataset_effective = effective.get("dataset", {})
    source_dataset_id = str(dataset_effective["dataset_source_id"])
    eval_split = str(dataset_effective["eval_split"])
    phase1_max_bayts = dataset_effective.get("phase1_max_bayts")
    max_bayts = int(phase1_max_bayts) if str(phase1_max_bayts or "").strip() else None
    eval_bucket_quota = int((json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))).get("eval_bank_per_meter_per_bucket", 2))
    seed = int(cfg["run"]["seed"])

    eval_prepared = load_and_prepare_dataset(
        dataset_id=source_dataset_id,
        split=eval_split,
        max_bayts=max_bayts,
        allowed_meters=None,
        hf_token=hf_token,
    )
    eval_rows = choose_balanced_rows(list(eval_prepared), per_meter_per_bucket=eval_bucket_quota, seed=seed)

    gen_cfg = cfg["generation"]
    prompts = [row["prompt"] for row in eval_rows]
    completions = []

    if args.backend == "vllm":
        llm, lora_request = build_vllm_engine(base_model_id, adapter_repo, hf_token=hf_token, cfg=cfg)
        for start in range(0, len(prompts), int(args.batch_size)):
            batch_prompts = prompts[start : start + int(args.batch_size)]
            completions.extend(
                generate_batch_vllm(
                    llm,
                    lora_request,
                    batch_prompts,
                    max_new_tokens=int(gen_cfg["max_completion_length"]),
                    temperature=float(gen_cfg["temperature"]),
                    top_p=float(gen_cfg["top_p"]),
                )
            )
    else:
        tokenizer, model = build_transformers_model(base_model_id, adapter_repo, hf_token=hf_token)
        for start in range(0, len(prompts), int(args.batch_size)):
            batch_prompts = prompts[start : start + int(args.batch_size)]
            completions.extend(
                generate_batch_transformers(
                    tokenizer,
                    model,
                    batch_prompts,
                    max_new_tokens=int(gen_cfg["max_completion_length"]),
                    temperature=float(gen_cfg["temperature"]),
                    top_p=float(gen_cfg["top_p"]),
                )
            )

    rows = []
    for row, completion in zip(eval_rows, completions):
        meter_out = score_meter_poem(
            completion,
            target_meter=str(row.get("meter_label", "")),
            base_meter=str(row.get("base_meter", "")),
            aggregator="logmean",
        )
        count_out = score_count_adherence(int(row.get("requested_bayts", 0) or 0), completion)
        clean_out = score_arabic_cleanliness(completion)
        repeat_out = score_repetition_penalty(completion)
        total = float(meter_out["score"]) * float(count_out["score"]) * float(clean_out["score"]) * float(repeat_out["score"])
        rows.append(
            {
                "base_meter": str(row.get("base_meter", "")),
                "meter_label": str(row.get("meter_label", "")),
                "length_bucket": str(row.get("length_bucket", "")),
                "requested_bayts": int(row.get("requested_bayts", 0) or 0),
                "requested_lines": int(row.get("requested_lines", 0) or 0),
                "source_index": int(row.get("source_index", 0) or 0),
                "description": str(row.get("description", "")),
                "prompt": str(row.get("prompt", "")),
                "completion_text": completion,
                "reward_meter": float(meter_out["score"]),
                "reward_count_adherence": float(count_out["score"]),
                "reward_arabic_clean": float(clean_out["score"]),
                "reward_repeat_penalty": float(repeat_out["score"]),
                "reward_total": total,
                "has_arabic": bool(clean_out["has_arabic"]),
                "contains_latin": bool(clean_out["contains_latin"]),
                "generated_bayts": int(count_out.get("generated_bayts", 0) or 0),
                "max_line_repeat_count": int(repeat_out.get("max_line_repeat_count", 0) or 0),
                "dominant_line_fraction": float(repeat_out.get("dominant_line_fraction", 0.0) or 0.0),
            }
        )

    (output_dir / "baseline_eval_generations.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    per_meter = []
    meter_buckets = defaultdict(list)
    for row in rows:
        meter_buckets[row["base_meter"]].append(row)
    for meter, bucket_rows in sorted(meter_buckets.items(), key=lambda item: mean(r["reward_total"] for r in item[1]), reverse=True):
        per_meter.append(
            {
                "base_meter": meter,
                "n": len(bucket_rows),
                "reward_total_mean": mean(r["reward_total"] for r in bucket_rows),
                "reward_meter_mean": mean(r["reward_meter"] for r in bucket_rows),
                "reward_count_adherence_mean": mean(r["reward_count_adherence"] for r in bucket_rows),
                "reward_repeat_penalty_mean": mean(r["reward_repeat_penalty"] for r in bucket_rows),
                "reward_arabic_clean_mean": mean(r["reward_arabic_clean"] for r in bucket_rows),
            }
        )

    by_length = []
    length_buckets = defaultdict(list)
    for row in rows:
        length_buckets[row["length_bucket"]].append(row)
    for bucket in ["1-3", "4-6", "7-10", "11-20"]:
        bucket_rows = length_buckets.get(bucket, [])
        if not bucket_rows:
            continue
        by_length.append(
            {
                "length_bucket": bucket,
                "n": len(bucket_rows),
                "reward_total_mean": mean(r["reward_total"] for r in bucket_rows),
                "reward_meter_mean": mean(r["reward_meter"] for r in bucket_rows),
                "reward_count_adherence_mean": mean(r["reward_count_adherence"] for r in bucket_rows),
                "reward_repeat_penalty_mean": mean(r["reward_repeat_penalty"] for r in bucket_rows),
                "reward_arabic_clean_mean": mean(r["reward_arabic_clean"] for r in bucket_rows),
            }
        )

    write_csv(
        output_dir / "baseline_eval_per_meter.csv",
        per_meter,
        ["base_meter", "n", "reward_total_mean", "reward_meter_mean", "reward_count_adherence_mean", "reward_repeat_penalty_mean", "reward_arabic_clean_mean"],
    )
    write_csv(
        output_dir / "baseline_eval_by_length_bucket.csv",
        by_length,
        ["length_bucket", "n", "reward_total_mean", "reward_meter_mean", "reward_count_adherence_mean", "reward_repeat_penalty_mean", "reward_arabic_clean_mean"],
    )

    summary = {
        "adapter_repo": adapter_repo,
        "base_model_id": base_model_id,
        "source_dataset_id": source_dataset_id,
        "eval_split": eval_split,
        "num_rows": len(rows),
        "backend": args.backend,
        "reward_total_mean": mean(r["reward_total"] for r in rows),
        "reward_meter_mean": mean(r["reward_meter"] for r in rows),
        "reward_count_adherence_mean": mean(r["reward_count_adherence"] for r in rows),
        "reward_repeat_penalty_mean": mean(r["reward_repeat_penalty"] for r in rows),
        "reward_arabic_clean_mean": mean(r["reward_arabic_clean"] for r in rows),
    }
    save_json(summary, output_dir / "baseline_eval_summary.json")

    render_grouped_bar(
        [row["base_meter"] for row in per_meter],
        {
            "total": [row["reward_total_mean"] for row in per_meter],
            "meter": [row["reward_meter_mean"] for row in per_meter],
            "count": [row["reward_count_adherence_mean"] for row in per_meter],
            "repeat": [row["reward_repeat_penalty_mean"] for row in per_meter],
        },
        output_dir / "baseline_per_meter_components.png",
        "Shaer-adapters Before GRPO: Eval Per-Meter Components",
    )
    render_grouped_bar(
        [row["length_bucket"] for row in by_length],
        {
            "total": [row["reward_total_mean"] for row in by_length],
            "meter": [row["reward_meter_mean"] for row in by_length],
            "count": [row["reward_count_adherence_mean"] for row in by_length],
            "repeat": [row["reward_repeat_penalty_mean"] for row in by_length],
        },
        output_dir / "baseline_by_length_bucket_components.png",
        "Shaer-adapters Before GRPO: Eval By Length Bucket",
    )
    render_histograms(
        {
            "total reward": [row["reward_total"] for row in rows],
            "meter": [row["reward_meter"] for row in rows],
            "count adherence": [row["reward_count_adherence"] for row in rows],
            "anti-repeat": [row["reward_repeat_penalty"] for row in rows],
            "arabic clean": [row["reward_arabic_clean"] for row in rows],
        },
        output_dir / "baseline_reward_histograms.png",
        "Shaer-adapters Before GRPO: Reward Distributions",
    )

    after_csv = run_dir / "final_plots" / "best_checkpoint_eval_per_meter.csv"
    if after_csv.exists():
        after_rows = []
        with after_csv.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                after_rows.append(row)
        after_index = {row["base_meter"]: row for row in after_rows}
        compare_rows = []
        for row in per_meter:
            meter = row["base_meter"]
            after = after_index.get(meter)
            if not after:
                continue
            compare_rows.append(
                {
                    "base_meter": meter,
                    "before_total": row["reward_total_mean"],
                    "after_total": float(after["reward_total_mean"]),
                    "delta_total": float(after["reward_total_mean"]) - row["reward_total_mean"],
                    "before_meter": row["reward_meter_mean"],
                    "after_meter": float(after["reward_meter_mean"]),
                    "delta_meter": float(after["reward_meter_mean"]) - row["reward_meter_mean"],
                    "before_count": row["reward_count_adherence_mean"],
                    "after_count": float(after["reward_count_adherence_mean"]),
                    "delta_count": float(after["reward_count_adherence_mean"]) - row["reward_count_adherence_mean"],
                    "before_repeat": row["reward_repeat_penalty_mean"],
                    "after_repeat": float(after["reward_repeat_penalty_mean"]),
                    "delta_repeat": float(after["reward_repeat_penalty_mean"]) - row["reward_repeat_penalty_mean"],
                }
            )
        write_csv(
            output_dir / "baseline_vs_best_checkpoint_per_meter.csv",
            compare_rows,
            ["base_meter", "before_total", "after_total", "delta_total", "before_meter", "after_meter", "delta_meter", "before_count", "after_count", "delta_count", "before_repeat", "after_repeat", "delta_repeat"],
        )
        render_grouped_bar(
            [row["base_meter"] for row in compare_rows],
            {
                "before total": [row["before_total"] for row in compare_rows],
                "after total": [row["after_total"] for row in compare_rows],
            },
            output_dir / "before_vs_after_total_by_meter.png",
            "Before vs After GRPO: Total Reward By Meter",
        )
        render_grouped_bar(
            [row["base_meter"] for row in compare_rows],
            {
                "before meter": [row["before_meter"] for row in compare_rows],
                "after meter": [row["after_meter"] for row in compare_rows],
            },
            output_dir / "before_vs_after_meter_by_meter.png",
            "Before vs After GRPO: Meter Score By Meter",
        )

    best_clean = {}
    for row in rows:
        meter = row["base_meter"]
        score_tuple = (
            1 if row["reward_arabic_clean"] >= 1.0 else 0,
            1 if row["reward_repeat_penalty"] >= 0.95 else 0,
            1 if row["reward_count_adherence"] >= 0.95 else 0,
            float(row["reward_total"]),
            float(row["reward_meter"]),
        )
        if meter not in best_clean or score_tuple > best_clean[meter][0]:
            best_clean[meter] = (score_tuple, row)
    build_poem_markdown(
        {meter: payload[1] for meter, payload in best_clean.items()},
        output_dir / "best_clean_poem_per_meter.md",
        "Shaer-adapters Before GRPO: One Clean Strong Poem Per Meter",
    )

    readme_lines = [
        "# Before-GRPO Baseline Bundle",
        "",
        "This folder evaluates `Shaer-AI/Shaer-adapters` on the same balanced GRPO eval bank used by the finished GRPO run.",
        "",
        f"- adapter repo: `{adapter_repo}`",
        f"- base model: `{base_model_id}`",
        f"- dataset: `{source_dataset_id}` `{eval_split}`",
        f"- selected eval rows: `{len(rows)}`",
        f"- backend: `{args.backend}`",
        "",
        "## Summary",
        "",
        f"- total: `{summary['reward_total_mean']:.4f}`",
        f"- meter: `{summary['reward_meter_mean']:.4f}`",
        f"- count adherence: `{summary['reward_count_adherence_mean']:.4f}`",
        f"- repeat penalty: `{summary['reward_repeat_penalty_mean']:.4f}`",
        f"- arabic clean: `{summary['reward_arabic_clean_mean']:.4f}`",
        "",
        "## Main Files",
        "",
        "- `baseline_eval_generations.jsonl`",
        "- `baseline_eval_per_meter.csv`",
        "- `baseline_eval_by_length_bucket.csv`",
        "- `baseline_reward_histograms.png`",
        "- `baseline_per_meter_components.png`",
        "- `baseline_by_length_bucket_components.png`",
        "- `before_vs_after_total_by_meter.png`",
        "- `before_vs_after_meter_by_meter.png`",
        "- `baseline_vs_best_checkpoint_per_meter.csv`",
        "- `best_clean_poem_per_meter.md`",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
