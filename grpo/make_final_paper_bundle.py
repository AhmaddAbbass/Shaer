import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_live_rewards import (
    build_plot_specs,
    load_arabic_gate_series,
    load_generation_spreads,
    load_meter_by_meter_series,
    load_metrics_for_runs,
    render_arabic_gate_plot,
    render_component_panels,
    render_kl_plot,
    render_meter_by_meter_plot,
    render_plots,
    select_component_metrics,
    write_chain_artifacts,
)
from rewards.common import ensure_dir


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


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


def determine_best_eval_row(metrics_rows):
    eval_rows = [row for row in metrics_rows if str(row.get("mode", "")) == "eval" and row.get("eval_reward_total_mean") is not None]
    if not eval_rows:
        raise RuntimeError("No eval rows found in metrics.jsonl")
    return max(eval_rows, key=lambda row: float(row.get("eval_reward_total_mean", float("-inf"))))


def iter_best_eval_generations(path: Path, step: int):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            if str(row.get("mode", "")) != "eval":
                continue
            try:
                if int(row.get("global_step", -1) or -1) != int(step):
                    continue
            except Exception:
                continue
            yield row


def render_grouped_bar(labels, series_map, output_path: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(labels) + 4), 6))
    metric_names = list(series_map.keys())
    width = 0.8 / max(1, len(metric_names))
    xs = list(range(len(labels)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, metric_name in enumerate(metric_names):
        vals = series_map[metric_name]
        offset = (idx - (len(metric_names) - 1) / 2) * width
        ax.bar([x + offset for x in xs], vals, width=width, label=metric_name, color=colors[idx % len(colors)])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(5, len(metric_names)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_histograms(values_map, output_path: Path, title: str):
    keys = list(values_map.keys())
    ncols = 2
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows))
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for ax, key, color in zip(axes, keys, colors * 10):
        vals = [float(v) for v in values_map[key]]
        ax.hist(vals, bins=12, color=color, alpha=0.82, edgecolor="white")
        ax.set_title(key)
        ax.set_xlim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.2)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_boxplot_by_meter(rows, metric_key: str, output_path: Path, title: str):
    meters = sorted({row["base_meter"] for row in rows})
    series = [[float(row[metric_key]) for row in rows if row["base_meter"] == meter] for meter in meters]
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(meters) + 4), 6))
    ax.boxplot(series, tick_labels=meters, patch_artist=True, boxprops={"facecolor": "#9ecae1"})
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_meter_histograms_by_meter(rows, output_path: Path, title: str):
    meters = sorted({row["base_meter"] for row in rows})
    ncols = 3
    nrows = math.ceil(len(meters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.8 * nrows), sharex=True, sharey=True)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    for ax, meter in zip(axes, meters):
        vals = [float(row["reward_meter"]) for row in rows if row["base_meter"] == meter]
        ax.hist(vals, bins=8, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.set_title(meter)
        ax.set_xlim(0.0, 1.02)
        ax.grid(True, axis="y", alpha=0.2)
    for ax in axes[len(meters):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_scatter(rows, x_key: str, y_key: str, output_path: Path, title: str):
    bucket_colors = {
        "1-3": "#1f77b4",
        "4-6": "#ff7f0e",
        "7-10": "#2ca02c",
        "11-20": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for bucket, color in bucket_colors.items():
        points = [row for row in rows if str(row.get("length_bucket", "")) == bucket]
        if not points:
            continue
        ax.scatter(
            [float(row[x_key]) for row in points],
            [float(row[y_key]) for row in points],
            alpha=0.75,
            s=36,
            color=color,
            label=bucket,
        )
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, title="length bucket")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def build_one_poem_markdown(best_rows, output_path: Path, best_step: int):
    lines = [f"# Best Checkpoint {best_step}: One Eval Poem Per Meter", ""]
    lines.append("These are the strongest single eval samples per meter at the best checkpoint, selected by highest `reward_total` for that meter.")
    lines.append("")
    for meter in sorted(best_rows):
        row = best_rows[meter]
        lines.append(f"## {meter}")
        lines.append("")
        lines.append(f"- total: `{float(row.get('reward_total', 0.0)):.4f}`")
        lines.append(f"- meter: `{float(row.get('reward_meter', 0.0)):.4f}`")
        lines.append(f"- count: `{float(row.get('reward_count_adherence', row.get('reward_exact_count_bonus', 0.0))):.4f}`")
        lines.append(f"- repeat: `{float(row.get('reward_repeat_penalty', 0.0)):.4f}`")
        lines.append("")
        lines.append("```text")
        lines.append(str(row.get("completion_text", "")).strip())
        lines.append("```")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_readme(output_path: Path, run_dir: Path, best_eval_row: dict, per_meter_rows, by_length_rows):
    best_step = int(best_eval_row.get("global_step", 0) or 0)
    lines = [
        "# Final Paper Plot Bundle",
        "",
        f"- run dir: `{run_dir}`",
        f"- best checkpoint step: `{best_step}`",
        f"- best eval total: `{float(best_eval_row.get('eval_reward_total_mean', 0.0)):.4f}`",
        f"- best eval meter: `{float(best_eval_row.get('eval_reward_meter_mean', 0.0)):.4f}`",
        f"- best eval count adherence: `{float(best_eval_row.get('eval_reward_count_adherence_mean', best_eval_row.get('eval_reward_exact_count_bonus_mean', 0.0))):.4f}`",
        f"- best eval repeat penalty: `{float(best_eval_row.get('eval_reward_repeat_penalty_mean', 0.0)):.4f}`",
        f"- best eval arabic clean: `{float(best_eval_row.get('eval_reward_arabic_clean_mean', 0.0)):.4f}`",
        "",
        "## Connected Run Plots",
        "",
        "- `reward_panels_train_connected.png`",
        "- `reward_panels_eval_connected.png`",
        "- `reward_curves_connected.png`",
        "- `kl_connected.png`",
        "- `arabic_gate_connected.png`",
        "- `meter_by_meter_connected.png`",
        "",
        "These use the resume chain as one continuous run and intentionally omit the dashed transition marker.",
        "",
        "## Best Checkpoint Analysis",
        "",
        "- `best_checkpoint_per_meter_components.png`",
        "- `best_checkpoint_reward_histograms.png`",
        "- `best_checkpoint_meter_histograms_by_meter.png`",
        "- `best_checkpoint_total_boxplot_by_meter.png`",
        "- `best_checkpoint_length_bucket_components.png`",
        "- `best_checkpoint_total_vs_repeat_scatter.png`",
        "- `best_checkpoint_eval_per_meter.csv`",
        "- `best_checkpoint_eval_by_length_bucket.csv`",
        "- `best_checkpoint_3100_one_poem_per_meter.md`",
        "",
        "## Best Checkpoint Per-Meter Summary",
        "",
    ]
    for row in per_meter_rows:
        lines.append(
            f"- {row['base_meter']}: total `{float(row['reward_total_mean']):.3f}`, meter `{float(row['reward_meter_mean']):.3f}`, "
            f"count `{float(row['reward_count_adherence_mean']):.3f}`, repeat `{float(row['reward_repeat_penalty_mean']):.3f}`"
        )
    lines.append("")
    lines.append("## Best Checkpoint By Length Bucket")
    lines.append("")
    for row in by_length_rows:
        lines.append(
            f"- {row['length_bucket']}: total `{float(row['reward_total_mean']):.3f}`, meter `{float(row['reward_meter_mean']):.3f}`, "
            f"count `{float(row['reward_count_adherence_mean']):.3f}`, repeat `{float(row['reward_repeat_penalty_mean']):.3f}`"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build publication-ready final plot and best-checkpoint bundle for a completed GRPO run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else run_dir / "final_plots"
    ensure_dir(output_dir)

    chain_specs = build_plot_specs(run_dir)
    chain_rows = load_metrics_for_runs(chain_specs)
    chain_spreads = load_generation_spreads(chain_specs)
    chain_gate_series = load_arabic_gate_series(chain_specs)
    chain_meter_by_meter = load_meter_by_meter_series(chain_specs)
    train_metric_specs = select_component_metrics(chain_rows, "train")
    eval_metric_specs = select_component_metrics(chain_rows, "eval")

    render_component_panels(
        chain_rows,
        chain_spreads,
        output_dir,
        "reward_panels_train_connected.png",
        "Connected Train Rewards",
        "train",
        train_metric_specs,
        boundaries=None,
    )
    render_component_panels(
        chain_rows,
        chain_spreads,
        output_dir,
        "reward_panels_eval_connected.png",
        "Connected Eval Rewards",
        "eval",
        eval_metric_specs,
        boundaries=None,
    )
    render_plots(chain_rows, output_dir, "reward_curves_connected.png", "Connected Reward Curves", boundaries=None)
    render_kl_plot(chain_rows, output_dir, "kl_connected.png", "Connected KL", boundaries=None)
    render_arabic_gate_plot(chain_gate_series, output_dir, "arabic_gate_connected.png", "Connected Arabic Gate", boundaries=None)
    render_meter_by_meter_plot(chain_meter_by_meter, output_dir, "meter_by_meter_connected.png", "Connected Meter By Meter", boundaries=None)
    write_chain_artifacts(run_dir, chain_specs, chain_rows, output_dir)

    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    best_eval_row = determine_best_eval_row(metrics_rows)
    best_step = int(best_eval_row.get("global_step", 0) or 0)

    best_eval_rows = []
    best_by_meter = {}
    for row in iter_best_eval_generations(run_dir / "all_generations.jsonl", best_step):
        base_meter = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
        length_bucket = str(row.get("length_bucket", "")).strip()
        record = {
            "base_meter": base_meter,
            "length_bucket": length_bucket,
            "reward_total": float(row.get("reward_total", 0.0) or 0.0),
            "reward_meter": float(row.get("reward_meter", 0.0) or 0.0),
            "reward_count_adherence": float(row.get("reward_count_adherence", row.get("reward_exact_count_bonus", 0.0)) or 0.0),
            "reward_repeat_penalty": float(row.get("reward_repeat_penalty", 0.0) or 0.0),
            "reward_arabic_clean": float(row.get("reward_arabic_clean", 0.0) or 0.0),
        }
        best_eval_rows.append(record)
        if base_meter and (base_meter not in best_by_meter or float(row.get("reward_total", 0.0) or 0.0) > float(best_by_meter[base_meter].get("reward_total", 0.0) or 0.0)):
            best_by_meter[base_meter] = row

    if not best_eval_rows:
        raise RuntimeError(f"No eval generations found for best step {best_step}")

    meter_buckets = defaultdict(list)
    length_buckets = defaultdict(list)
    for row in best_eval_rows:
        meter_buckets[row["base_meter"]].append(row)
        length_buckets[row["length_bucket"]].append(row)

    per_meter_rows = []
    for meter, rows in sorted(meter_buckets.items(), key=lambda item: mean(r["reward_total"] for r in item[1]), reverse=True):
        per_meter_rows.append(
            {
                "base_meter": meter,
                "n": len(rows),
                "reward_total_mean": mean(r["reward_total"] for r in rows),
                "reward_meter_mean": mean(r["reward_meter"] for r in rows),
                "reward_count_adherence_mean": mean(r["reward_count_adherence"] for r in rows),
                "reward_repeat_penalty_mean": mean(r["reward_repeat_penalty"] for r in rows),
                "reward_arabic_clean_mean": mean(r["reward_arabic_clean"] for r in rows),
            }
        )

    by_length_rows = []
    for bucket in ["1-3", "4-6", "7-10", "11-20"]:
        rows = length_buckets.get(bucket, [])
        if not rows:
            continue
        by_length_rows.append(
            {
                "length_bucket": bucket,
                "n": len(rows),
                "reward_total_mean": mean(r["reward_total"] for r in rows),
                "reward_meter_mean": mean(r["reward_meter"] for r in rows),
                "reward_count_adherence_mean": mean(r["reward_count_adherence"] for r in rows),
                "reward_repeat_penalty_mean": mean(r["reward_repeat_penalty"] for r in rows),
                "reward_arabic_clean_mean": mean(r["reward_arabic_clean"] for r in rows),
            }
        )

    write_csv(
        output_dir / "best_checkpoint_eval_per_meter.csv",
        per_meter_rows,
        ["base_meter", "n", "reward_total_mean", "reward_meter_mean", "reward_count_adherence_mean", "reward_repeat_penalty_mean", "reward_arabic_clean_mean"],
    )
    write_csv(
        output_dir / "best_checkpoint_eval_by_length_bucket.csv",
        by_length_rows,
        ["length_bucket", "n", "reward_total_mean", "reward_meter_mean", "reward_count_adherence_mean", "reward_repeat_penalty_mean", "reward_arabic_clean_mean"],
    )

    summary = {
        "run_id": run_dir.name,
        "best_step": best_step,
        "best_eval_total": float(best_eval_row.get("eval_reward_total_mean", 0.0) or 0.0),
        "best_eval_meter": float(best_eval_row.get("eval_reward_meter_mean", 0.0) or 0.0),
        "best_eval_count_adherence": float(best_eval_row.get("eval_reward_count_adherence_mean", best_eval_row.get("eval_reward_exact_count_bonus_mean", 0.0)) or 0.0),
        "best_eval_repeat_penalty": float(best_eval_row.get("eval_reward_repeat_penalty_mean", 0.0) or 0.0),
        "best_eval_arabic_clean": float(best_eval_row.get("eval_reward_arabic_clean_mean", 0.0) or 0.0),
        "num_eval_generations": len(best_eval_rows),
    }
    (output_dir / "best_checkpoint_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    labels = [row["base_meter"] for row in per_meter_rows]
    render_grouped_bar(
        labels,
        {
            "total": [row["reward_total_mean"] for row in per_meter_rows],
            "meter": [row["reward_meter_mean"] for row in per_meter_rows],
            "count": [row["reward_count_adherence_mean"] for row in per_meter_rows],
            "repeat": [row["reward_repeat_penalty_mean"] for row in per_meter_rows],
        },
        output_dir / "best_checkpoint_per_meter_components.png",
        f"Best Checkpoint {best_step}: Per-Meter Components",
        "Average score",
    )
    render_grouped_bar(
        [row["length_bucket"] for row in by_length_rows],
        {
            "total": [row["reward_total_mean"] for row in by_length_rows],
            "meter": [row["reward_meter_mean"] for row in by_length_rows],
            "count": [row["reward_count_adherence_mean"] for row in by_length_rows],
            "repeat": [row["reward_repeat_penalty_mean"] for row in by_length_rows],
        },
        output_dir / "best_checkpoint_length_bucket_components.png",
        f"Best Checkpoint {best_step}: By Length Bucket",
        "Average score",
    )
    render_histograms(
        {
            "total reward": [row["reward_total"] for row in best_eval_rows],
            "meter": [row["reward_meter"] for row in best_eval_rows],
            "count adherence": [row["reward_count_adherence"] for row in best_eval_rows],
            "anti-repeat": [row["reward_repeat_penalty"] for row in best_eval_rows],
            "arabic clean": [row["reward_arabic_clean"] for row in best_eval_rows],
        },
        output_dir / "best_checkpoint_reward_histograms.png",
        f"Best Checkpoint {best_step}: Reward Distributions",
    )
    render_meter_histograms_by_meter(
        best_eval_rows,
        output_dir / "best_checkpoint_meter_histograms_by_meter.png",
        f"Best Checkpoint {best_step}: Meter Score Histograms By Meter",
    )
    render_boxplot_by_meter(
        best_eval_rows,
        "reward_total",
        output_dir / "best_checkpoint_total_boxplot_by_meter.png",
        f"Best Checkpoint {best_step}: Total Reward By Meter",
    )
    render_scatter(
        best_eval_rows,
        "reward_repeat_penalty",
        "reward_total",
        output_dir / "best_checkpoint_total_vs_repeat_scatter.png",
        f"Best Checkpoint {best_step}: Total vs Anti-Repeat",
    )

    poem_md_path = output_dir / f"best_checkpoint_{best_step}_one_poem_per_meter.md"
    build_one_poem_markdown(best_by_meter, poem_md_path, best_step)
    build_readme(output_dir / "README.md", run_dir, best_eval_row, per_meter_rows, by_length_rows)


if __name__ == "__main__":
    main()
