import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "outputs" / "sanity_check_model"
OUTPUT_ROOT = ROOT / "outputs" / "sanity_check_model_base_clean"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def suggest_action(row: pd.Series) -> str:
    if (
        row["median_score"] >= 0.55
        and row["pass_rate_0_7"] >= 0.50
        and row["mean_prompt_best_score"] >= 0.70
    ):
        return "keep"
    if (
        row["median_score"] < 0.15
        and row["pass_rate_0_5"] <= 0.15
        and row["mean_prompt_best_score"] < 0.35
    ):
        return "candidate_for_removal_or_extra_filtering"
    return "keep_but_flag_weak"


def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.bar(df[x_col], df[y_col], color="#2f5aa8")
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grouped_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    ordered = df["base_meter"].tolist()
    x = range(len(ordered))
    width = 0.25
    plt.figure(figsize=(15, 7))
    plt.bar([i - width for i in x], df["pass_rate_0_5"], width=width, label="pass@0.5", color="#7db7ff")
    plt.bar(x, df["pass_rate_0_7"], width=width, label="pass@0.7", color="#2f5aa8")
    plt.bar([i + width for i in x], df["pass_rate_0_9"], width=width, label="pass@0.9", color="#0f2b5b")
    plt.xticks(list(x), ordered, rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("Fraction of generations")
    plt.title("Pass Rates by Base Meter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_boxplot_by_base_meter(df: pd.DataFrame, ordered_bases: list[str], out_path: Path) -> None:
    data = [df.loc[df["scored_base_meter"] == meter, "meter_score"].tolist() for meter in ordered_bases]
    plt.figure(figsize=(15, 7))
    plt.boxplot(data, tick_labels=ordered_bases, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Meter score")
    plt.title("Meter Score Distribution by Base Meter")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_histogram(values: pd.Series, title: str, xlabel: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=20, color="#2f5aa8", edgecolor="black", alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prompt_mix(pivot_df: pd.DataFrame, out_path: Path) -> None:
    ax = pivot_df.plot(
        kind="bar",
        stacked=True,
        figsize=(15, 7),
        colormap="tab20",
    )
    ax.set_xlabel("Base meter")
    ax.set_ylabel("Number of generations")
    ax.set_title("Requested Label Mix Feeding Each Base Meter")
    ax.legend(title="Requested label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_grpo_signal(df: pd.DataFrame, out_path: Path) -> None:
    ordered = df["base_meter"].tolist()
    x = range(len(ordered))
    width = 0.25
    plt.figure(figsize=(15, 7))
    plt.bar([i - width for i in x], df["mean_prompt_best_score"], width=width, label="mean prompt best", color="#1b9e77")
    plt.bar(x, df["mean_prompt_worst_score"], width=width, label="mean prompt worst", color="#d95f02")
    plt.bar([i + width for i in x], df["mean_prompt_gap"], width=width, label="mean prompt gap", color="#7570b3")
    plt.xticks(list(x), ordered, rotation=90)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Approximate GRPO Signal by Base Meter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def build_prompt_groups(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["scored_base_meter", "requested_meter_label", "prompt_group_id"]
    grouped = df.sort_values("generation_index").groupby(group_cols, as_index=False)
    prompt_groups = grouped.agg(
        prompt_preview=("prompt_preview", "first"),
        requested_bayts=("requested_bayts", "first"),
        requested_lines=("requested_lines", "first"),
        num_candidates=("meter_score", "size"),
        mean_score=("meter_score", "mean"),
        best_score=("meter_score", "max"),
        worst_score=("meter_score", "min"),
        min_score=("meter_score", "min"),
        max_score=("meter_score", "max"),
        mean_scored_bayts=("num_scored_bayts", "mean"),
        mean_output_lines=("num_output_lines", "mean"),
        any_unpaired_last_line=("has_unpaired_last_line", "max"),
    )
    prompt_groups["score_gap"] = prompt_groups["best_score"] - prompt_groups["worst_score"]
    prompt_groups["has_good_candidate_0_5"] = (prompt_groups["best_score"] >= 0.5).astype(float)
    prompt_groups["has_good_candidate_0_7"] = (prompt_groups["best_score"] >= 0.7).astype(float)
    prompt_groups["has_good_candidate_0_9"] = (prompt_groups["best_score"] >= 0.9).astype(float)
    prompt_groups["has_clear_preference_gap_0_2"] = (prompt_groups["score_gap"] >= 0.2).astype(float)
    return prompt_groups


def build_base_meter_summary(df: pd.DataFrame, prompt_groups: pd.DataFrame) -> pd.DataFrame:
    gen_summary = (
        df.groupby("scored_base_meter", as_index=False)
        .agg(
            num_generations=("meter_score", "size"),
            mean_score=("meter_score", "mean"),
            median_score=("meter_score", "median"),
            std_score=("meter_score", "std"),
            min_score=("meter_score", "min"),
            max_score=("meter_score", "max"),
            pass_rate_0_5=("above_0_5", "mean"),
            pass_rate_0_7=("above_0_7", "mean"),
            pass_rate_0_9=("above_0_9", "mean"),
            mean_requested_bayts=("requested_bayts", "mean"),
            mean_output_line_pairs=("num_output_line_pairs", "mean"),
            mean_scored_bayts=("num_scored_bayts", "mean"),
            odd_tail_rate=("has_unpaired_last_line", "mean"),
            requested_label_count=("requested_meter_label", "nunique"),
        )
        .rename(columns={"scored_base_meter": "base_meter"})
    )

    prompt_summary = (
        prompt_groups.groupby("scored_base_meter", as_index=False)
        .agg(
            prompt_groups=("prompt_group_id", "nunique"),
            mean_prompt_best_score=("best_score", "mean"),
            mean_prompt_worst_score=("worst_score", "mean"),
            mean_prompt_gap=("score_gap", "mean"),
            prompts_with_good_candidate_0_5=("has_good_candidate_0_5", "mean"),
            prompts_with_good_candidate_0_7=("has_good_candidate_0_7", "mean"),
            prompts_with_good_candidate_0_9=("has_good_candidate_0_9", "mean"),
            prompts_with_clear_preference_gap_0_2=("has_clear_preference_gap_0_2", "mean"),
        )
        .rename(columns={"scored_base_meter": "base_meter"})
    )

    summary = gen_summary.merge(prompt_summary, on="base_meter", how="left")
    summary = summary.sort_values(["mean_score", "median_score"], ascending=[False, False]).reset_index(drop=True)
    summary["rank"] = summary.index + 1
    summary["suggested_action"] = summary.apply(suggest_action, axis=1)
    return summary


def build_requested_label_mix(df: pd.DataFrame, prompt_groups: pd.DataFrame) -> pd.DataFrame:
    gen_counts = (
        df.groupby(["scored_base_meter", "requested_meter_label"], as_index=False)
        .size()
        .rename(columns={"size": "num_generations"})
    )
    prompt_counts = (
        prompt_groups.groupby(["scored_base_meter", "requested_meter_label"], as_index=False)
        .size()
        .rename(columns={"size": "num_prompt_groups"})
    )
    mix = gen_counts.merge(prompt_counts, on=["scored_base_meter", "requested_meter_label"], how="left")
    total_by_base = mix.groupby("scored_base_meter")["num_generations"].transform("sum")
    mix["generation_share"] = mix["num_generations"] / total_by_base
    mix = mix.rename(columns={"scored_base_meter": "base_meter"})
    mix = mix.sort_values(["base_meter", "num_generations", "requested_meter_label"], ascending=[True, False, True])
    return mix


def plot_requested_label_breakdown(label_summary: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(label_summary["requested_meter_label"], label_summary["mean_score"], color="#2f5aa8")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Mean score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_prompt_group_scores(prompt_groups: pd.DataFrame, title: str, out_path: Path) -> None:
    ordered = prompt_groups.sort_values("best_score", ascending=False).reset_index(drop=True)
    x = range(len(ordered))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], ordered["best_score"], width=width, label="best candidate", color="#1b9e77")
    plt.bar([i + width / 2 for i in x], ordered["worst_score"], width=width, label="worst candidate", color="#d95f02")
    plt.xticks(list(x), [f"P{i+1}" for i in x], rotation=0)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_top_level_readme(
    report: dict,
    summary: pd.DataFrame,
    prompt_mix: pd.DataFrame,
    output_root: Path,
) -> None:
    strongest = summary.head(5)[["base_meter", "mean_score"]].values.tolist()
    weakest = summary.tail(5)[["base_meter", "mean_score"]].values.tolist()
    keep_list = summary.loc[summary["suggested_action"] == "keep", "base_meter"].tolist()
    weak_list = summary.loc[summary["suggested_action"] == "keep_but_flag_weak", "base_meter"].tolist()
    remove_list = summary.loc[
        summary["suggested_action"] == "candidate_for_removal_or_extra_filtering", "base_meter"
    ].tolist()
    lines = [
        "# Sanity Check Model Base-Clean View",
        "",
        "This folder is a clean reporting layer derived from [`outputs/sanity_check_model`](../sanity_check_model/).",
        "No new model run was executed to build it. The goal is to make the existing study readable for group review and paper writing.",
        "",
        "## What This Folder Measures",
        "",
        "- Prompts may still mention form-specific requested labels such as `مجزوء البسيط` or `مخلع البسيط`.",
        "- The meter reward in this study is base-meter-first, so every generation here was scored against its `base_meter`.",
        "- In this run, each prompt had `2` sampled candidates.",
        "- That means a weak result for a requested form label should be read as: the model missed even the easier base-meter target, not that the evaluator punished form-specific mismatch.",
        "",
        "## Folder Layout",
        "",
        "- [`report.json`](report.json): machine-readable run summary.",
        "- [`base_meter_summary.csv`](base_meter_summary.csv): the main table, one row per base meter.",
        "- [`prompt_mix_by_base_meter.csv`](prompt_mix_by_base_meter.csv): which requested labels fed each base meter.",
        "- [`all_generations_clean.csv`](all_generations_clean.csv): cleaned row-level table with clearer column names.",
        "- [`plots/`](plots/): small set of top-level plots.",
        "- [`per_meter/`](per_meter/): one folder per base meter, each with its own README, CSVs, and PNGs.",
        "",
        "## Main Table Columns",
        "",
        "- `mean_score`: average meter score across all generations for that base meter.",
        "- `median_score`: middle score. This is often safer than the mean when a few generations are extremely good or extremely bad.",
        "- `std_score`: score spread. High spread means instability.",
        "- `min_score`: worst observed generation for that base meter.",
        "- `max_score`: best observed generation for that base meter.",
        "- `pass_rate_0_5`, `pass_rate_0_7`, `pass_rate_0_9`: fraction of generations that crossed those quality thresholds.",
        "- `mean_prompt_best_score`: for each prompt, take the best candidate among its sampled generations, then average across prompts.",
        "- `mean_prompt_worst_score`: same idea, but with the worst candidate per prompt.",
        "- `mean_prompt_gap`: average difference between best and worst candidates per prompt. This is an approximate GRPO preference-signal measure.",
        "- `prompts_with_good_candidate_0_7`: fraction of prompts where at least one candidate reached `0.7`.",
        "- `prompts_with_clear_preference_gap_0_2`: fraction of prompts where the within-prompt score gap was at least `0.2`.",
        "- `odd_tail_rate`: fraction of generations with an unpaired final line. Those generations are still scored on complete line pairs only.",
        "- `suggested_action`: simple heuristic bucket to help review, not a hard rule.",
        "",
        "## How To Read Each Meter Folder",
        "",
        "- `summary.json`: machine-readable summary for that base meter.",
        "- `README.md`: short human guide and current interpretation for that meter.",
        "- `all_rows.csv`: every generation for that base meter.",
        "- `prompt_groups.csv`: one row per prompt, summarizing the best and worst candidate scores. This is the most GRPO-relevant CSV.",
        "- `requested_label_summary.csv`: performance split by requested label feeding this base meter.",
        "- `examples_best.csv`: highest-scoring examples.",
        "- `examples_worst.csv`: lowest-scoring examples.",
        "- `plots/score_histogram.png`: score distribution for that base meter.",
        "- `plots/requested_label_breakdown.png`: mean score by requested label within that base meter.",
        "- `plots/prompt_group_scores.png`: best vs worst candidate score for each prompt group.",
        "",
        "## Analysis",
        "",
        f"- Base meters evaluated: `{int(summary['base_meter'].nunique())}`",
        f"- Total generations evaluated: `{int(report['num_generations_total'])}`",
        f"- Overall mean score: `{report['overall_mean_score']:.4f}`",
        f"- Overall median score: `{report['overall_median_score']:.4f}`",
        "",
        "### Strongest Base Meters In This Run",
        "",
    ]
    for meter, score in strongest:
        lines.append(f"- `{meter}`: mean score `{score:.4f}`")
    lines += [
        "",
        "### Weakest Base Meters In This Run",
        "",
    ]
    for meter, score in weakest:
        lines.append(f"- `{meter}`: mean score `{score:.4f}`")
    lines += [
        "",
        "### How Min and Max Should Be Interpreted",
        "",
        "- A high `max_score` means the model can sometimes produce a strong candidate on that meter.",
        "- A very low `min_score` means failures can still be severe, even if the meter sometimes looks good.",
        "- For GRPO, a high `max_score` alone is not enough. We also want a usable within-prompt ranking signal.",
        "",
        "### Why The Prompt-Level Best and Worst Scores Matter For GRPO",
        "",
        "- GRPO compares candidates generated for the same prompt.",
        "- If a meter has a strong `mean_prompt_best_score` and a healthy `mean_prompt_gap`, then GRPO can often prefer one candidate over another and produce a useful update signal.",
        "- If both `mean_prompt_best_score` and `mean_prompt_worst_score` are near zero, then even ranking candidates becomes weak because all candidates are bad.",
        "- If `mean_prompt_best_score` is decent but `mean_prompt_worst_score` is much lower, then GRPO may still have room to learn because the reward can separate better from worse outputs.",
        "",
        "### Heuristic Review Buckets",
        "",
        f"- `keep`: {', '.join(keep_list) if keep_list else 'none'}",
        f"- `keep_but_flag_weak`: {', '.join(weak_list) if weak_list else 'none'}",
        f"- `candidate_for_removal_or_extra_filtering`: {', '.join(remove_list) if remove_list else 'none'}",
        "",
        "These buckets are only review aids. They are not automatic policy decisions.",
        "",
        "### Requested-Label Mixing",
        "",
        "- Some base meters are fed by more than one requested label.",
        "- Example: `البسيط` in this run includes prompts from `البسيط`, `مجزوء البسيط`, and `مخلع البسيط`.",
        "- Use [`prompt_mix_by_base_meter.csv`](prompt_mix_by_base_meter.csv) and the per-meter `requested_label_summary.csv` files to see whether weakness comes from one requested label or from the base meter more broadly.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def write_meter_readme(meter_row: pd.Series, label_summary: pd.DataFrame, meter_dir: Path) -> None:
    strongest_label = label_summary.sort_values("mean_score", ascending=False).iloc[0]
    weakest_label = label_summary.sort_values("mean_score", ascending=True).iloc[0]
    lines = [
        f"# {meter_row['base_meter']}",
        "",
        "This folder contains the base-meter-clean view for this meter only.",
        "",
        "## Headline Stats",
        "",
        f"- Mean score: `{meter_row['mean_score']:.4f}`",
        f"- Median score: `{meter_row['median_score']:.4f}`",
        f"- Std score: `{meter_row['std_score']:.4f}`",
        f"- Min score: `{meter_row['min_score']:.4f}`",
        f"- Max score: `{meter_row['max_score']:.4f}`",
        f"- Pass@0.5: `{meter_row['pass_rate_0_5']:.4f}`",
        f"- Pass@0.7: `{meter_row['pass_rate_0_7']:.4f}`",
        f"- Pass@0.9: `{meter_row['pass_rate_0_9']:.4f}`",
        f"- Mean prompt best score: `{meter_row['mean_prompt_best_score']:.4f}`",
        f"- Mean prompt worst score: `{meter_row['mean_prompt_worst_score']:.4f}`",
        f"- Mean prompt gap: `{meter_row['mean_prompt_gap']:.4f}`",
        f"- Suggested action: `{meter_row['suggested_action']}`",
        "",
        "## Requested Label View",
        "",
        f"- Strongest requested label feeding this base meter: `{strongest_label['requested_meter_label']}` with mean `{strongest_label['mean_score']:.4f}`",
        f"- Weakest requested label feeding this base meter: `{weakest_label['requested_meter_label']}` with mean `{weakest_label['mean_score']:.4f}`",
        "",
        "## How To Read This Meter",
        "",
        "- Start with `summary.json` and this README for the headline picture.",
        "- Use `prompt_groups.csv` to judge whether GRPO will get a useful within-prompt ranking signal.",
        "- Use `requested_label_summary.csv` to see whether any requested label is dragging the base meter down.",
        "- Use `examples_best.csv` and `examples_worst.csv` to inspect real outputs instead of relying only on averages.",
        "",
        "## Interpretation Guide",
        "",
        "- High `max_score` with low `min_score` means the meter sometimes works but is unstable.",
        "- High `mean_prompt_best_score` with a noticeable `mean_prompt_gap` means GRPO may be able to prefer stronger candidates.",
        "- Low `mean_prompt_best_score` and low `mean_prompt_gap` together mean even the best candidate per prompt is weak, which makes RL harder.",
    ]
    (meter_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    all_generations = pd.read_csv(SOURCE_ROOT / "all_generations.csv")
    source_report = json.loads((SOURCE_ROOT / "report.json").read_text())

    ensure_clean_dir(OUTPUT_ROOT)
    plots_root = OUTPUT_ROOT / "plots"
    per_meter_root = OUTPUT_ROOT / "per_meter"
    plots_root.mkdir(parents=True, exist_ok=True)
    per_meter_root.mkdir(parents=True, exist_ok=True)

    cleaned = all_generations.rename(
        columns={
            "meter_label": "requested_meter_label",
            "base_meter": "scored_base_meter",
            "target_meter_used": "reward_target_meter",
            "meter_reward": "meter_score",
            "generated_num_lines": "num_output_lines",
            "generated_complete_bayts": "num_output_line_pairs",
            "num_valid_bayts": "num_scored_bayts",
            "num_skipped_bayts": "num_unscored_bayts",
            "has_odd_tail": "has_unpaired_last_line",
            "odd_tail_line": "unpaired_last_line_text",
            "meter_reward_mean": "per_poem_mean_score",
            "meter_reward_logmean": "per_poem_logmean_score",
            "meter_reward_std": "per_poem_score_std",
            "meter_reward_min": "per_poem_min_score",
            "meter_reward_max": "per_poem_max_score",
        }
    ).copy()
    cleaned["prompt_group_id"] = (
        cleaned["requested_meter_label"].astype(str)
        + "::"
        + cleaned["source_index"].astype(str)
        + "::"
        + cleaned["prompt_rank_within_meter"].astype(str)
    )

    ordered_columns = [
        "scored_base_meter",
        "requested_meter_label",
        "reward_target_meter",
        "target_resolution",
        "prompt_group_id",
        "source_index",
        "prompt_rank_within_meter",
        "generation_index",
        "requested_bayts",
        "requested_lines",
        "prompt_has_odd_tail",
        "meter_score",
        "per_poem_mean_score",
        "per_poem_logmean_score",
        "per_poem_score_std",
        "per_poem_min_score",
        "per_poem_max_score",
        "num_scored_bayts",
        "num_unscored_bayts",
        "num_output_lines",
        "num_output_line_pairs",
        "has_unpaired_last_line",
        "unpaired_last_line_text",
        "above_0_3",
        "above_0_5",
        "above_0_7",
        "above_0_9",
        "prompt_preview",
        "generated_text",
    ]
    cleaned = cleaned[ordered_columns]

    prompt_groups = build_prompt_groups(cleaned)
    base_summary = build_base_meter_summary(cleaned, prompt_groups)
    prompt_mix = build_requested_label_mix(cleaned, prompt_groups)

    report = {
        "source_output_root": str(SOURCE_ROOT.relative_to(ROOT)),
        "clean_output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "dataset_id": source_report["dataset_id"],
        "generation_backend": source_report["generation_backend"],
        "num_generations_total": int(len(cleaned)),
        "num_base_meters": int(cleaned["scored_base_meter"].nunique()),
        "num_requested_labels": int(cleaned["requested_meter_label"].nunique()),
        "overall_mean_score": float(cleaned["meter_score"].mean()),
        "overall_median_score": float(cleaned["meter_score"].median()),
        "overall_std_score": float(cleaned["meter_score"].std()),
        "overall_min_score": float(cleaned["meter_score"].min()),
        "overall_max_score": float(cleaned["meter_score"].max()),
        "target_resolution_counts": {
            key: int(value) for key, value in cleaned["target_resolution"].value_counts().to_dict().items()
        },
        "strongest_base_meters_by_mean": base_summary.head(5)["base_meter"].tolist(),
        "weakest_base_meters_by_mean": base_summary.tail(5)["base_meter"].tolist(),
    }

    cleaned.to_csv(OUTPUT_ROOT / "all_generations_clean.csv", index=False)
    base_summary.to_csv(OUTPUT_ROOT / "base_meter_summary.csv", index=False)
    prompt_mix.to_csv(OUTPUT_ROOT / "prompt_mix_by_base_meter.csv", index=False)
    write_json(OUTPUT_ROOT / "report.json", report)

    ordered_bases = base_summary["base_meter"].tolist()
    plot_bar(base_summary, "base_meter", "mean_score", "Mean Meter Score by Base Meter", "Mean score", plots_root / "base_meter_mean_score.png")
    plot_grouped_pass_rates(base_summary, plots_root / "base_meter_pass_rates.png")
    plot_boxplot_by_base_meter(cleaned, ordered_bases, plots_root / "base_meter_boxplot.png")
    plot_histogram(cleaned["meter_score"], "Overall Meter Score Distribution", "Meter score", plots_root / "overall_score_histogram.png")
    prompt_mix_pivot = (
        prompt_mix.pivot(index="base_meter", columns="requested_meter_label", values="num_generations")
        .fillna(0)
        .reindex(ordered_bases)
    )
    plot_prompt_mix(prompt_mix_pivot, plots_root / "prompt_mix_by_base_meter.png")
    plot_grpo_signal(base_summary, plots_root / "grpo_signal_by_base_meter.png")

    for _, meter_row in base_summary.iterrows():
        base_meter = meter_row["base_meter"]
        meter_rank = int(meter_row["rank"])
        meter_dir = per_meter_root / f"{meter_rank:02d}_{base_meter}"
        meter_plots_dir = meter_dir / "plots"
        meter_dir.mkdir(parents=True, exist_ok=True)
        meter_plots_dir.mkdir(parents=True, exist_ok=True)

        meter_rows = cleaned.loc[cleaned["scored_base_meter"] == base_meter].copy()
        meter_prompt_groups = prompt_groups.loc[prompt_groups["scored_base_meter"] == base_meter].copy()
        meter_label_summary = (
            meter_rows.groupby("requested_meter_label", as_index=False)
            .agg(
                num_generations=("meter_score", "size"),
                mean_score=("meter_score", "mean"),
                median_score=("meter_score", "median"),
                std_score=("meter_score", "std"),
                min_score=("meter_score", "min"),
                max_score=("meter_score", "max"),
                pass_rate_0_5=("above_0_5", "mean"),
                pass_rate_0_7=("above_0_7", "mean"),
                pass_rate_0_9=("above_0_9", "mean"),
                mean_scored_bayts=("num_scored_bayts", "mean"),
            )
            .sort_values(["mean_score", "median_score"], ascending=[False, False])
        )
        meter_examples_best = meter_rows.sort_values("meter_score", ascending=False).head(5)
        meter_examples_worst = meter_rows.sort_values("meter_score", ascending=True).head(5)

        meter_rows.to_csv(meter_dir / "all_rows.csv", index=False)
        meter_prompt_groups.to_csv(meter_dir / "prompt_groups.csv", index=False)
        meter_label_summary.to_csv(meter_dir / "requested_label_summary.csv", index=False)
        meter_examples_best.to_csv(meter_dir / "examples_best.csv", index=False)
        meter_examples_worst.to_csv(meter_dir / "examples_worst.csv", index=False)

        meter_summary_json = {
            "base_meter": base_meter,
            "rank": meter_rank,
            "num_generations": int(meter_row["num_generations"]),
            "prompt_groups": int(meter_row["prompt_groups"]),
            "requested_label_count": int(meter_row["requested_label_count"]),
            "mean_score": float(meter_row["mean_score"]),
            "median_score": float(meter_row["median_score"]),
            "std_score": float(meter_row["std_score"]) if not math.isnan(meter_row["std_score"]) else 0.0,
            "min_score": float(meter_row["min_score"]),
            "max_score": float(meter_row["max_score"]),
            "pass_rate_0_5": float(meter_row["pass_rate_0_5"]),
            "pass_rate_0_7": float(meter_row["pass_rate_0_7"]),
            "pass_rate_0_9": float(meter_row["pass_rate_0_9"]),
            "mean_prompt_best_score": float(meter_row["mean_prompt_best_score"]),
            "mean_prompt_worst_score": float(meter_row["mean_prompt_worst_score"]),
            "mean_prompt_gap": float(meter_row["mean_prompt_gap"]),
            "prompts_with_good_candidate_0_7": float(meter_row["prompts_with_good_candidate_0_7"]),
            "prompts_with_clear_preference_gap_0_2": float(meter_row["prompts_with_clear_preference_gap_0_2"]),
            "suggested_action": meter_row["suggested_action"],
        }
        write_json(meter_dir / "summary.json", meter_summary_json)
        write_meter_readme(meter_row, meter_label_summary, meter_dir)

        plot_histogram(
            meter_rows["meter_score"],
            f"Score Distribution: {base_meter}",
            "Meter score",
            meter_plots_dir / "score_histogram.png",
        )
        plot_requested_label_breakdown(
            meter_label_summary,
            f"Requested Label Mean Scores Feeding {base_meter}",
            meter_plots_dir / "requested_label_breakdown.png",
        )
        plot_prompt_group_scores(
            meter_prompt_groups,
            f"Best vs Worst Candidate per Prompt: {base_meter}",
            meter_plots_dir / "prompt_group_scores.png",
        )

    write_top_level_readme(report, base_summary, prompt_mix, OUTPUT_ROOT)
    print(f"saved clean folder: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
