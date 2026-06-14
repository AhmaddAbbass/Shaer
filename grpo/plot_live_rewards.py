import argparse
from collections import defaultdict
import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rewards.common import ensure_dir


DEFAULT_TRAIN_COMPONENT_METRICS = [
    ("reward_total_mean", "reward_total", "total reward"),
    ("reward_meter_mean", "reward_meter", "meter"),
    ("reward_count_adherence_mean", "reward_count_adherence", "count adherence"),
    ("reward_hard_gate_mean", "reward_hard_gate", "hard gate"),
    ("reward_judge_quality_mean", "reward_judge_quality", "judge quality"),
    ("reward_judge_meaning_fit_mean", "reward_judge_meaning_fit", "judge meaning fit"),
    ("reward_judge_naturalness_mean", "reward_judge_naturalness", "judge naturalness"),
    ("reward_repeat_soft_mean", "reward_repeat_soft", "repeat soft"),
    ("loss", None, "train loss"),
]

DEFAULT_EVAL_COMPONENT_METRICS = [
    ("eval_reward_total_mean", "reward_total", "total reward"),
    ("eval_reward_meter_mean", "reward_meter", "meter"),
    ("eval_reward_count_adherence_mean", "reward_count_adherence", "count adherence"),
    ("eval_reward_hard_gate_mean", "reward_hard_gate", "hard gate"),
    ("eval_reward_judge_quality_mean", "reward_judge_quality", "judge quality"),
    ("eval_reward_judge_meaning_fit_mean", "reward_judge_meaning_fit", "judge meaning fit"),
    ("eval_reward_judge_naturalness_mean", "reward_judge_naturalness", "judge naturalness"),
    ("eval_reward_repeat_soft_mean", "reward_repeat_soft", "repeat soft"),
    ("eval_loss", None, "eval loss"),
]

ARABIC_GATE_METRICS = [
    ("exact_count_rate", "exact-count rate"),
    ("hard_gate_failure_rate", "hard-gate failure rate"),
    ("contamination_fail_rate", "contamination fail rate"),
    ("arabic_clean_ok_rate", "arabic_clean_ok rate"),
    ("contains_latin_rate", "latin contamination rate"),
    ("missing_arabic_rate", "missing Arabic rate"),
    ("contains_digits_rate", "digit leakage rate"),
    ("contamination_ratio_mean", "contamination ratio"),
    ("artifact_free_rate", "artifact-free rate"),
    ("lexical_plausibility_mean", "lexical plausibility"),
]

TEMPLATE_QUALITY_METRICS = [
    ("exact_repeat_score_mean", "exact-repeat score"),
    ("near_duplicate_score_mean", "near-duplicate score"),
    ("opening_diversity_score_mean", "opening diversity"),
    ("distinct_2_score_mean", "distinct-2 score"),
    ("near_duplicate_pair_fraction_mean", "near-dup pair frac"),
    ("opening_dominance_fraction_mean", "opening dominance"),
]

AGGREGATE_COLOR = "#d62728"
TRAIN_SMOOTH_WINDOW = 25
EVAL_SMOOTH_WINDOW = 5
GENERIC_SMOOTH_WINDOW = 15


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def load_csv_rows(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def parse_utc_timestamp(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except Exception:
        return 0.0


def read_lineage(run_dir: Path):
    for name in ["lineage.json", "run_summary.json"]:
        data = read_json(run_dir / name)
        if data:
            return data
    return {"run_id": run_dir.name, "run_dir": str(run_dir), "chain_id": run_dir.name, "run_sequence_index": 0}


def discover_chain_runs(run_dir: Path, follow_chain: bool):
    lineage = read_lineage(run_dir)
    if not follow_chain:
        return [(run_dir, lineage)]
    chain_id = str(lineage.get("chain_id", run_dir.name))
    runs = []
    for child in sorted(run_dir.parent.iterdir()):
        if not child.is_dir():
            continue
        child_lineage = read_lineage(child)
        if str(child_lineage.get("chain_id", "")) != chain_id:
            continue
        runs.append((child, child_lineage))
    runs.sort(
        key=lambda item: (
            int(item[1].get("run_sequence_index", 0) or 0),
            parse_utc_timestamp(str(item[1].get("created_at_utc", ""))),
            item[0].name,
        )
    )
    return runs


def parse_checkpoint_step(path_value):
    text = str(path_value or "").strip()
    if not text:
        return None
    name = Path(text).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("checkpoint-", 1)[1])
        except Exception:
            return None
    return None


def build_plot_specs(run_dir: Path):
    specs = []
    current_dir = Path(run_dir).resolve()
    current_lineage = read_lineage(current_dir)

    specs.append(
        {
            "run_dir": current_dir,
            "lineage": current_lineage,
            "min_step": None,
            "max_step": None,
            "repeat_penalty_visible": True,
        }
    )

    child_dir = current_dir
    child_lineage = current_lineage
    while child_lineage and str(child_lineage.get("parent_run_dir", "")).strip():
        parent_dir = Path(str(child_lineage["parent_run_dir"])).resolve()
        resume_decision = read_json(child_dir / "resume_decision.json") or {}
        cutoff = parse_checkpoint_step(resume_decision.get("local_resume_path"))
        specs.append(
            {
                "run_dir": parent_dir,
                "lineage": read_lineage(parent_dir),
                "min_step": None,
                "max_step": cutoff,
                "repeat_penalty_visible": False,
            }
        )
        child_dir = parent_dir
        child_lineage = read_lineage(child_dir)

    specs.reverse()
    return specs


def discover_baseline_eval_per_meter(run_dir: Path):
    candidates = []
    direct = run_dir / "plots_before" / "baseline_eval_per_meter.csv"
    if direct.exists():
        candidates.append(direct)
    train_root = run_dir.parent
    for path in train_root.glob("*/plots_before/baseline_eval_per_meter.csv"):
        if path.exists():
            candidates.append(path)
    if not candidates:
        return {}
    candidates.sort(key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)
    chosen = candidates[0]
    baselines = {}
    for row in load_csv_rows(chosen):
        meter = str(row.get("base_meter", "")).strip()
        value = row.get("reward_meter_mean")
        if not meter or value in (None, ""):
            continue
        try:
            baselines[meter] = float(value)
        except Exception:
            continue
    return baselines


def discover_trained_meter_filter(run_dir: Path):
    summary = read_json(run_dir / "dataset_summary.json") or {}
    counts = summary.get("train_base_meter_counts") or {}
    meters = set()
    for meter_name, count in counts.items():
        meter = str(meter_name or "").strip()
        if not meter:
            continue
        try:
            if int(count or 0) <= 0:
                continue
        except Exception:
            continue
        meters.add(meter)
    return meters


def run_step_extent(spec):
    run_dir = Path(spec["run_dir"])
    min_allowed = spec.get("min_step")
    max_allowed = spec.get("max_step")
    min_step = None
    max_step = None
    for row in read_jsonl(run_dir / "metrics.jsonl"):
        step = row.get("global_step")
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            continue
        if min_allowed is not None and step < int(min_allowed):
            continue
        if max_allowed is not None and step > int(max_allowed):
            continue
        min_step = step if min_step is None else min(min_step, step)
        max_step = step if max_step is None else max(max_step, step)
    if max_step is None:
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            step = generation_step(row)
            if step is None:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            min_step = step if min_step is None else min(min_step, step)
            max_step = step if max_step is None else max(max_step, step)
    return min_step, max_step


def compute_run_offsets(run_specs):
    offsets = {}
    chain_cursor = 0
    for idx, spec in enumerate(run_specs):
        run_dir = Path(spec["run_dir"])
        min_step, max_step = run_step_extent(spec)
        if idx == 0:
            offset = 0
        elif min_step is None:
            offset = chain_cursor
        else:
            offset = chain_cursor - int(min_step) + 1
        offsets[str(run_dir)] = offset
        if max_step is not None:
            chain_cursor = offset + int(max_step)
    return offsets


def load_metrics_for_runs(run_specs):
    offsets = compute_run_offsets(run_specs)
    rows = []
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        lineage = spec["lineage"]
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        show_repeat_penalty = bool(spec.get("repeat_penalty_visible", True))
        for row in read_jsonl(run_dir / "metrics.jsonl"):
            rec = dict(row)
            step = rec.get("global_step")
            try:
                step_int = int(step) if step is not None else None
            except Exception:
                step_int = None
            if step_int is not None:
                if min_allowed is not None and step_int < int(min_allowed):
                    continue
                if max_allowed is not None and step_int > int(max_allowed):
                    continue
            rec.setdefault("run_id", str(lineage.get("run_id", run_dir.name)))
            rec.setdefault("run_sequence_index", int(lineage.get("run_sequence_index", 0) or 0))
            try:
                rec["_plot_step"] = float(offset + int(step_int)) if step_int is not None else None
            except Exception:
                rec["_plot_step"] = None
            if not show_repeat_penalty:
                rec.pop("reward_repeat_penalty_mean", None)
                rec.pop("reward_repeat_penalty_std", None)
                rec.pop("eval_reward_repeat_penalty_mean", None)
                rec.pop("eval_reward_repeat_penalty_std", None)
            rows.append(rec)
    rows.sort(
        key=lambda row: (
            int(row.get("run_sequence_index", 0) or 0),
            float(row.get("_plot_step", 0) or 0),
            0 if str(row.get("mode", "train")) == "train" else 1,
            parse_utc_timestamp(str(row.get("timestamp_utc", ""))),
        )
    )
    return rows


def series_for(rows, metric_name, step_key="_plot_step"):
    xs, ys = [], []
    for row in rows:
        value = row.get(metric_name)
        step = row.get(step_key)
        if value is None or step is None:
            continue
        try:
            xs.append(float(step))
            ys.append(float(value))
        except Exception:
            continue
    return xs, ys


def rolling_mean(values, window: int):
    if not values:
        return []
    window = max(1, min(int(window), len(values)))
    out = []
    running = 0.0
    for idx, value in enumerate(values):
        running += float(value)
        if idx >= window:
            running -= float(values[idx - window])
        denom = min(idx + 1, window)
        out.append(running / float(denom))
    return out


def draw_series_with_aggregate(
    ax,
    xs,
    ys,
    *,
    raw_label=None,
    raw_color="#1f77b4",
    raw_marker="o",
    raw_linewidth=2.0,
    raw_markersize=3,
    aggregate_window=GENERIC_SMOOTH_WINDOW,
    aggregate_label=None,
    aggregate_linestyle="-",
):
    if not xs:
        return
    ax.plot(
        xs,
        ys,
        marker=raw_marker,
        linewidth=raw_linewidth,
        markersize=raw_markersize,
        color=raw_color,
        alpha=0.45,
        label=raw_label,
    )
    agg = rolling_mean(ys, aggregate_window)
    ax.plot(
        xs,
        agg,
        linewidth=max(2.4, raw_linewidth + 0.4),
        color=AGGREGATE_COLOR,
        linestyle=aggregate_linestyle,
        alpha=0.95,
        label=aggregate_label,
    )


def generation_step(row):
    mode = str(row.get("mode", "train"))
    if mode == "train" and row.get("logical_step") is not None:
        return row.get("logical_step")
    return row.get("global_step")


def summarize_values(values):
    clean = []
    for value in values:
        try:
            clean.append(float(value))
        except Exception:
            continue
    if not clean:
        return None
    clean.sort()
    if len(clean) == 1:
        q1 = q3 = clean[0]
    else:
        q1, _, q3 = statistics.quantiles(clean, n=4, method="inclusive")
    return {
        "min": clean[0],
        "max": clean[-1],
        "q1": q1,
        "q3": q3,
    }


def load_generation_spreads(run_specs, eval_meter_filter=None):
    offsets = compute_run_offsets(run_specs)
    reward_keys = {
        "reward_total",
        "reward_meter",
        "reward_count_adherence",
        "reward_arabic_clean",
        "reward_hard_gate",
        "reward_repeat_soft",
        "reward_repeat_penalty",
        "reward_total_composite",
        "reward_exact_count_bonus",
        "reward_meter_count_clean",
    }
    buckets = {}
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        show_repeat_penalty = bool(spec.get("repeat_penalty_visible", True))
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            mode = str(row.get("mode", "train"))
            meter = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
            step = generation_step(row)
            if step is None:
                continue
            if mode == "eval" and eval_meter_filter and meter not in eval_meter_filter:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            plot_step = offset + step
            for reward_key in reward_keys:
                if reward_key == "reward_repeat_penalty" and not show_repeat_penalty:
                    continue
                value = row.get(reward_key)
                if value is not None:
                    buckets.setdefault((mode, plot_step), {key: [] for key in reward_keys})[reward_key].append(value)

    spreads = {}
    for bucket_key, reward_values in buckets.items():
        spreads[bucket_key] = {}
        for reward_key, values in reward_values.items():
            summary = summarize_values(values)
            if summary:
                spreads[bucket_key][reward_key] = summary
    return spreads


def load_filtered_eval_metric_rows(run_specs, eval_meter_filter=None):
    offsets = compute_run_offsets(run_specs)
    rows = []
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        lineage = spec["lineage"]
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        buckets = {}
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            if str(row.get("mode", "")) != "eval":
                continue
            meter = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
            if eval_meter_filter and meter not in eval_meter_filter:
                continue
            step = generation_step(row)
            if step is None:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            bucket = buckets.setdefault(
                step,
                {
                    "timestamp_utc": row.get("timestamp_utc"),
                    "mode": "eval",
                    "global_step": step,
                    "run_id": str(lineage.get("run_id", run_dir.name)),
                    "run_sequence_index": int(lineage.get("run_sequence_index", 0) or 0),
                    "_plot_step": float(offset + step),
                    "_values": defaultdict(list),
                },
            )
            for key, value in row.items():
                if not str(key).startswith("reward_"):
                    continue
                if key.endswith("_mean") or key.endswith("_std"):
                    continue
                try:
                    bucket["_values"][str(key)].append(float(value))
                except Exception:
                    continue
        for step in sorted(buckets):
            bucket = buckets[step]
            out = {k: v for k, v in bucket.items() if k != "_values"}
            for reward_key, values in bucket["_values"].items():
                if not values:
                    continue
                out[f"eval_{reward_key}_mean"] = statistics.mean(values)
                out[f"eval_{reward_key}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
            rows.append(out)
    rows.sort(
        key=lambda row: (
            int(row.get("run_sequence_index", 0) or 0),
            float(row.get("_plot_step", 0) or 0),
            parse_utc_timestamp(str(row.get("timestamp_utc", ""))),
        )
    )
    return rows


def select_component_metrics(rows, mode_name: str):
    metric_specs = []
    if mode_name == "train":
        prefix = ""
        row_has = lambda key: any(row.get(key) is not None for row in rows if str(row.get("mode", "train")) == "train")
        metric_specs.append(("reward_total_mean", "reward_total", "total reward"))
    else:
        prefix = "eval_"
        row_has = lambda key: any(row.get(key) is not None for row in rows if str(row.get("mode", "")) == "eval")
        metric_specs.append(("eval_reward_total_mean", "reward_total", "total reward"))

    if row_has(f"{prefix}reward_meter_mean"):
        metric_specs.append((f"{prefix}reward_meter_mean", "reward_meter", "meter"))
    elif row_has(f"{prefix}reward_meter_count_clean_mean"):
        metric_specs.append((f"{prefix}reward_meter_count_clean_mean", "reward_meter_count_clean", "optimized composite"))

    if row_has(f"{prefix}reward_count_adherence_mean"):
        metric_specs.append((f"{prefix}reward_count_adherence_mean", "reward_count_adherence", "count adherence"))
    elif row_has(f"{prefix}reward_exact_count_bonus_mean"):
        metric_specs.append((f"{prefix}reward_exact_count_bonus_mean", "reward_exact_count_bonus", "exact-count rate"))

    if row_has(f"{prefix}reward_arabic_clean_mean"):
        metric_specs.append((f"{prefix}reward_arabic_clean_mean", "reward_arabic_clean", "arabic-clean gate"))
    if row_has(f"{prefix}reward_hard_gate_mean"):
        metric_specs.append((f"{prefix}reward_hard_gate_mean", "reward_hard_gate", "hard gate"))
    if row_has(f"{prefix}reward_arabic_floor_mean"):
        metric_specs.append((f"{prefix}reward_arabic_floor_mean", "reward_arabic_floor", "arabic floor"))
    if row_has(f"{prefix}reward_count_floor_mean"):
        metric_specs.append((f"{prefix}reward_count_floor_mean", "reward_count_floor", "count floor"))
    if row_has(f"{prefix}reward_lexical_plausibility_mean"):
        metric_specs.append((f"{prefix}reward_lexical_plausibility_mean", "reward_lexical_plausibility", "lexical plausibility"))
    if row_has(f"{prefix}reward_judge_quality_mean"):
        metric_specs.append((f"{prefix}reward_judge_quality_mean", "reward_judge_quality", "judge quality"))
    if row_has(f"{prefix}reward_judge_meaning_fit_mean"):
        metric_specs.append((f"{prefix}reward_judge_meaning_fit_mean", "reward_judge_meaning_fit", "judge meaning fit"))
    if row_has(f"{prefix}reward_judge_naturalness_mean"):
        metric_specs.append((f"{prefix}reward_judge_naturalness_mean", "reward_judge_naturalness", "judge naturalness"))

    if row_has(f"{prefix}reward_repeat_soft_mean"):
        metric_specs.append((f"{prefix}reward_repeat_soft_mean", "reward_repeat_soft", "repeat soft"))
    if row_has(f"{prefix}reward_repeat_penalty_mean"):
        metric_specs.append((f"{prefix}reward_repeat_penalty_mean", "reward_repeat_penalty", "anti-repeat"))
    if row_has(f"{prefix}reward_repeat_floor_mean"):
        metric_specs.append((f"{prefix}reward_repeat_floor_mean", "reward_repeat_floor", "repeat floor"))
    if row_has(f"{prefix}reward_near_duplicate_penalty_mean"):
        metric_specs.append((f"{prefix}reward_near_duplicate_penalty_mean", "reward_near_duplicate_penalty", "near-duplicate penalty"))
    if row_has(f"{prefix}reward_opening_diversity_mean"):
        metric_specs.append((f"{prefix}reward_opening_diversity_mean", "reward_opening_diversity", "opening diversity"))
    if row_has(f"{prefix}reward_distinct_2_mean"):
        metric_specs.append((f"{prefix}reward_distinct_2_mean", "reward_distinct_2", "distinct-2"))

    if row_has(f"{prefix}loss"):
        metric_specs.append((f"{prefix}loss", None, "train loss" if mode_name == "train" else "eval loss"))
    return metric_specs or (DEFAULT_TRAIN_COMPONENT_METRICS if mode_name == "train" else DEFAULT_EVAL_COMPONENT_METRICS)


def compute_chain_boundaries(rows):
    starts = {}
    for row in rows:
        seq = row.get("run_sequence_index")
        step = row.get("_plot_step")
        if seq is None or step is None:
            continue
        try:
            seq = int(seq)
            step = float(step)
        except Exception:
            continue
        if seq <= 0:
            continue
        current = starts.get(seq)
        if current is None or step < current:
            starts[seq] = step
    boundaries = []
    for seq in sorted(starts):
        label = "drop trio + anti-repeat" if seq == 1 else f"run {seq + 1}"
        boundaries.append((starts[seq], label))
    return boundaries


def draw_boundaries(ax, boundaries):
    for step, label in boundaries or []:
        ax.axvline(step, linestyle="--", linewidth=1.2, color="#555555", alpha=0.9)
        ax.text(
            step,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#555555",
            backgroundcolor="white",
        )


def load_arabic_gate_series(run_specs, eval_meter_filter=None):
    offsets = compute_run_offsets(run_specs)
    buckets = {}
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            extra = row.get("arabic_clean_extra")
            if not isinstance(extra, dict):
                extra = row.get("total_composite_extra")
            if not isinstance(extra, dict):
                extra = row.get("meter_count_clean_extra")
            if not isinstance(extra, dict):
                continue
            mode = str(row.get("mode", "train"))
            meter = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
            step = generation_step(row)
            if step is None:
                continue
            if mode == "eval" and eval_meter_filter and meter not in eval_meter_filter:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            plot_step = offset + step
            bucket = buckets.setdefault(
                (mode, plot_step),
                {
                    "exact_count": [],
                    "hard_gate_failure": [],
                    "contamination_fail": [],
                    "arabic_clean_ok": [],
                    "contains_latin": [],
                    "missing_arabic": [],
                    "contains_digits": [],
                    "contamination_ratio": [],
                    "artifact_free": [],
                    "lexical_plausibility": [],
                },
            )
            has_arabic = bool(extra.get("has_arabic", False))
            contains_latin = bool(extra.get("contains_latin", False))
            arabic_clean_ok = bool(extra.get("arabic_clean_ok", False))
            contains_digits = bool(extra.get("contains_digits", False))
            contains_forbidden_artifacts = bool(extra.get("contains_forbidden_artifacts", False))
            exact_count = bool(extra.get("exact_count", False))
            hard_gate_failure = bool(extra.get("hard_gate_blocked", False))
            contamination_ratio = float(extra.get("contamination_ratio", 0.0) or 0.0)
            artifact_free = float(extra.get("artifact_free_score", 0.0) or 0.0)
            lexical_plausibility = float(extra.get("lexical_plausibility_score", 0.0) or 0.0)
            bucket["exact_count"].append(1.0 if exact_count else 0.0)
            bucket["hard_gate_failure"].append(1.0 if hard_gate_failure else 0.0)
            bucket["contamination_fail"].append(1.0 if ((not has_arabic) or contains_latin or contains_digits or contains_forbidden_artifacts) else 0.0)
            bucket["arabic_clean_ok"].append(1.0 if arabic_clean_ok else 0.0)
            bucket["contains_latin"].append(1.0 if contains_latin else 0.0)
            bucket["missing_arabic"].append(0.0 if has_arabic else 1.0)
            bucket["contains_digits"].append(1.0 if contains_digits else 0.0)
            bucket["contamination_ratio"].append(contamination_ratio)
            bucket["artifact_free"].append(artifact_free)
            bucket["lexical_plausibility"].append(lexical_plausibility)

    series = {"train": {}, "eval": {}}
    for (mode, step), values in buckets.items():
        mode_series = series.setdefault(mode, {})
        mode_series[step] = {
            "exact_count_rate": statistics.mean(values["exact_count"]) if values["exact_count"] else None,
            "hard_gate_failure_rate": statistics.mean(values["hard_gate_failure"]) if values["hard_gate_failure"] else None,
            "contamination_fail_rate": statistics.mean(values["contamination_fail"]) if values["contamination_fail"] else None,
            "arabic_clean_ok_rate": statistics.mean(values["arabic_clean_ok"]) if values["arabic_clean_ok"] else None,
            "contains_latin_rate": statistics.mean(values["contains_latin"]) if values["contains_latin"] else None,
            "missing_arabic_rate": statistics.mean(values["missing_arabic"]) if values["missing_arabic"] else None,
            "contains_digits_rate": statistics.mean(values["contains_digits"]) if values["contains_digits"] else None,
            "contamination_ratio_mean": statistics.mean(values["contamination_ratio"]) if values["contamination_ratio"] else None,
            "artifact_free_rate": statistics.mean(values["artifact_free"]) if values["artifact_free"] else None,
            "lexical_plausibility_mean": statistics.mean(values["lexical_plausibility"]) if values["lexical_plausibility"] else None,
        }
    return series


def load_template_quality_series(run_specs, eval_meter_filter=None):
    offsets = compute_run_offsets(run_specs)
    buckets = {}
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            extra = row.get("repeat_penalty_extra")
            if not isinstance(extra, dict):
                extra = row.get("total_composite_extra")
            if not isinstance(extra, dict):
                extra = row.get("meter_count_clean_extra")
            if not isinstance(extra, dict):
                continue
            mode = str(row.get("mode", "train"))
            meter = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
            step = generation_step(row)
            if step is None:
                continue
            if mode == "eval" and eval_meter_filter and meter not in eval_meter_filter:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            plot_step = offset + step
            bucket = buckets.setdefault(
                (mode, plot_step),
                {
                    "exact_repeat_score": [],
                    "near_duplicate_score": [],
                    "opening_diversity_score": [],
                    "distinct_2_score": [],
                    "near_duplicate_pair_fraction": [],
                    "opening_dominance_fraction": [],
                },
            )
            bucket["exact_repeat_score"].append(float(extra.get("exact_repeat_score", 0.0) or 0.0))
            bucket["near_duplicate_score"].append(float(extra.get("near_duplicate_score", 0.0) or 0.0))
            bucket["opening_diversity_score"].append(float(extra.get("opening_diversity_score", 0.0) or 0.0))
            bucket["distinct_2_score"].append(float(extra.get("distinct_2_score", 0.0) or 0.0))
            bucket["near_duplicate_pair_fraction"].append(float(extra.get("near_duplicate_pair_fraction", 0.0) or 0.0))
            bucket["opening_dominance_fraction"].append(float(extra.get("opening_dominance_fraction", 0.0) or 0.0))
    series = {"train": {}, "eval": {}}
    for (mode, step), values in buckets.items():
        mode_series = series.setdefault(mode, {})
        mode_series[step] = {
            "exact_repeat_score_mean": statistics.mean(values["exact_repeat_score"]) if values["exact_repeat_score"] else None,
            "near_duplicate_score_mean": statistics.mean(values["near_duplicate_score"]) if values["near_duplicate_score"] else None,
            "opening_diversity_score_mean": statistics.mean(values["opening_diversity_score"]) if values["opening_diversity_score"] else None,
            "distinct_2_score_mean": statistics.mean(values["distinct_2_score"]) if values["distinct_2_score"] else None,
            "near_duplicate_pair_fraction_mean": statistics.mean(values["near_duplicate_pair_fraction"]) if values["near_duplicate_pair_fraction"] else None,
            "opening_dominance_fraction_mean": statistics.mean(values["opening_dominance_fraction"]) if values["opening_dominance_fraction"] else None,
        }
    return series


def load_meter_by_meter_series(run_specs, eval_meter_filter=None):
    offsets = compute_run_offsets(run_specs)
    buckets = {"train": defaultdict(lambda: defaultdict(list)), "eval": defaultdict(lambda: defaultdict(list))}
    for spec in run_specs:
        run_dir = Path(spec["run_dir"])
        offset = offsets.get(str(run_dir), 0)
        min_allowed = spec.get("min_step")
        max_allowed = spec.get("max_step")
        for row in read_jsonl(run_dir / "all_generations.jsonl"):
            if bool(row.get("incomplete_reward_batch", False)):
                continue
            meter = str(row.get("base_meter", "")).strip()
            mode = str(row.get("mode", "train"))
            step = generation_step(row)
            score = row.get("reward_meter")
            if not meter or score is None or step is None:
                continue
            if mode == "eval" and eval_meter_filter and meter not in eval_meter_filter:
                continue
            try:
                step = int(step)
            except Exception:
                continue
            if min_allowed is not None and step < int(min_allowed):
                continue
            if max_allowed is not None and step > int(max_allowed):
                continue
            buckets.setdefault(mode, defaultdict(lambda: defaultdict(list)))[meter][offset + step].append(float(score))
    series = {"train": {}, "eval": {}}
    for mode, meter_steps in buckets.items():
        mode_series = {}
        for meter, step_values in meter_steps.items():
            points = []
            for step, values in sorted(step_values.items()):
                if not values:
                    continue
                points.append(
                    {
                        "step": step,
                        "mean": statistics.mean(values),
                        "count": len(values),
                    }
                )
            if points:
                mode_series[meter] = points
        series[mode] = mode_series
    return series


def render_component_panels(rows, generation_spreads, output_dir: Path, filename: str, title: str, mode_name: str, metric_specs, boundaries=None):
    ensure_dir(output_dir)
    ncols = 2 if len(metric_specs) > 1 else 1
    nrows = max(1, math.ceil(len(metric_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows), sharex=False)
    if hasattr(axes, "flatten"):
        axes = list(axes.flatten())
    else:
        axes = [axes]

    for ax, (metric_name, generation_metric_name, label) in zip(axes, metric_specs):
        xs, ys = series_for(rows, metric_name)
        if xs:
            draw_series_with_aggregate(
                ax,
                xs,
                ys,
                raw_label="raw",
                aggregate_label=f"aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW if mode_name == 'train' else EVAL_SMOOTH_WINDOW, len(ys)))}-step mean)",
                aggregate_window=TRAIN_SMOOTH_WINDOW if mode_name == "train" else EVAL_SMOOTH_WINDOW,
            )
            color = "#1f77b4"
            if generation_metric_name:
                q1_vals, q3_vals = [], []
                for step_value in xs:
                    spread = generation_spreads.get((mode_name, int(round(step_value))), {}).get(generation_metric_name)
                    if not spread:
                        q1_vals.append(math.nan)
                        q3_vals.append(math.nan)
                        continue
                    q1_vals.append(spread["q1"])
                    q3_vals.append(spread["q3"])
                if any(not math.isnan(v) for v in q1_vals):
                    ax.fill_between(xs, q1_vals, q3_vals, color=color, alpha=0.16)
        else:
            ax.text(0.5, 0.5, "No metrics yet", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(label)
        ax.set_xlabel("Chain step")
        ax.set_ylabel("Value")
        if generation_metric_name:
            if generation_metric_name == "reward_total":
                ax.set_ylim(-0.02, 1.22)
            else:
                ax.set_ylim(-0.02, 1.02)
        draw_boundaries(ax, boundaries)
        ax.grid(True, alpha=0.25)
        if xs:
            ax.legend()
    for ax in axes[len(metric_specs):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_plots(rows, output_dir: Path, filename: str, title: str, boundaries=None):
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    train_xs, train_ys = series_for([row for row in rows if str(row.get("mode", "train")) == "train"], "reward_total_mean")
    eval_xs, eval_ys = series_for([row for row in rows if str(row.get("mode", "")) == "eval"], "eval_reward_total_mean")
    if train_xs:
        draw_series_with_aggregate(
            ax,
            train_xs,
            train_ys,
            raw_label="train raw",
            aggregate_label=f"train aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW, len(train_ys)))}-step mean)",
            aggregate_window=TRAIN_SMOOTH_WINDOW,
        )
    if eval_xs:
        draw_series_with_aggregate(
            ax,
            eval_xs,
            eval_ys,
            raw_label="eval raw",
            raw_color="#ff7f0e",
            raw_marker="s",
            aggregate_label=f"eval aggregate ({max(1, min(EVAL_SMOOTH_WINDOW, len(eval_ys)))}-step mean)",
            aggregate_window=EVAL_SMOOTH_WINDOW,
            aggregate_linestyle="--",
        )
    if not train_xs and not eval_xs:
        ax.text(0.5, 0.5, "No metrics yet", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Chain step")
    ax.set_ylabel("Total reward")
    draw_boundaries(ax, boundaries)
    ax.grid(True, alpha=0.25)
    if train_xs or eval_xs:
        ax.legend()
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_kl_plot(rows, output_dir: Path, filename: str, title: str, boundaries=None):
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    train_xs, train_ys = series_for([row for row in rows if str(row.get("mode", "train")) == "train"], "kl")
    eval_xs, eval_ys = series_for([row for row in rows if str(row.get("mode", "")) == "eval"], "eval_kl")
    if train_xs:
        draw_series_with_aggregate(
            ax,
            train_xs,
            train_ys,
            raw_label="train raw",
            aggregate_label=f"train aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW, len(train_ys)))}-step mean)",
            aggregate_window=TRAIN_SMOOTH_WINDOW,
        )
    if eval_xs:
        draw_series_with_aggregate(
            ax,
            eval_xs,
            eval_ys,
            raw_label="eval raw",
            raw_color="#ff7f0e",
            raw_marker="s",
            aggregate_label=f"eval aggregate ({max(1, min(EVAL_SMOOTH_WINDOW, len(eval_ys)))}-step mean)",
            aggregate_window=EVAL_SMOOTH_WINDOW,
            aggregate_linestyle="--",
        )
    if not train_xs and not eval_xs:
        ax.text(0.5, 0.5, "No metrics yet", transform=ax.transAxes, ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Chain step")
    ax.set_ylabel("KL")
    draw_boundaries(ax, boundaries)
    ax.grid(True, alpha=0.25)
    if train_xs or eval_xs:
        ax.legend()
    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_arabic_gate_plot(gate_series, output_dir: Path, filename: str, title: str, boundaries=None):
    ensure_dir(output_dir)
    ncols = 3
    nrows = math.ceil(len(ARABIC_GATE_METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.3 * nrows), sharex=False)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    train_series = gate_series.get("train", {})
    eval_series = gate_series.get("eval", {})

    for ax, (metric_name, label) in zip(axes, ARABIC_GATE_METRICS):
        train_steps = sorted(step for step, values in train_series.items() if values.get(metric_name) is not None)
        eval_steps = sorted(step for step, values in eval_series.items() if values.get(metric_name) is not None)
        train_vals = [train_series[step][metric_name] for step in train_steps]
        eval_vals = [eval_series[step][metric_name] for step in eval_steps]
        if train_steps:
            draw_series_with_aggregate(
                ax,
                train_steps,
                train_vals,
                raw_label="train raw",
                aggregate_label=f"train aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW, len(train_vals)))}-step mean)",
                aggregate_window=TRAIN_SMOOTH_WINDOW,
            )
        if eval_steps:
            draw_series_with_aggregate(
                ax,
                eval_steps,
                eval_vals,
                raw_label="eval raw",
                raw_color="#ff7f0e",
                raw_marker="s",
                aggregate_label=f"eval aggregate ({max(1, min(EVAL_SMOOTH_WINDOW, len(eval_vals)))}-step mean)",
                aggregate_window=EVAL_SMOOTH_WINDOW,
                aggregate_linestyle="--",
            )
        if not train_steps and not eval_steps:
            ax.text(0.5, 0.5, "No gate data yet", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(label)
        ax.set_xlabel("Chain step")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.02, 1.02)
        draw_boundaries(ax, boundaries)
        ax.grid(True, alpha=0.25)
        if train_steps or eval_steps:
            ax.legend()

    for ax in axes[len(ARABIC_GATE_METRICS):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_template_quality_plot(series, output_dir: Path, filename: str, title: str, boundaries=None):
    ensure_dir(output_dir)
    ncols = 3
    nrows = math.ceil(len(TEMPLATE_QUALITY_METRICS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.3 * nrows), sharex=False)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]
    train_series = series.get("train", {})
    eval_series = series.get("eval", {})

    for ax, (metric_name, label) in zip(axes, TEMPLATE_QUALITY_METRICS):
        train_steps = sorted(step for step, values in train_series.items() if values.get(metric_name) is not None)
        eval_steps = sorted(step for step, values in eval_series.items() if values.get(metric_name) is not None)
        train_vals = [train_series[step][metric_name] for step in train_steps]
        eval_vals = [eval_series[step][metric_name] for step in eval_steps]
        if train_steps:
            draw_series_with_aggregate(
                ax,
                train_steps,
                train_vals,
                raw_label="train raw",
                aggregate_label=f"train aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW, len(train_vals)))}-step mean)",
                aggregate_window=TRAIN_SMOOTH_WINDOW,
            )
        if eval_steps:
            draw_series_with_aggregate(
                ax,
                eval_steps,
                eval_vals,
                raw_label="eval raw",
                raw_color="#ff7f0e",
                raw_marker="s",
                aggregate_label=f"eval aggregate ({max(1, min(EVAL_SMOOTH_WINDOW, len(eval_vals)))}-step mean)",
                aggregate_window=EVAL_SMOOTH_WINDOW,
                aggregate_linestyle="--",
            )
        if not train_steps and not eval_steps:
            ax.text(0.5, 0.5, "No template data yet", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(label)
        ax.set_xlabel("Chain step")
        ax.set_ylabel("Value")
        ax.set_ylim(-0.02, 1.02)
        draw_boundaries(ax, boundaries)
        ax.grid(True, alpha=0.25)
        if train_steps or eval_steps:
            ax.legend()
    for ax in axes[len(TEMPLATE_QUALITY_METRICS):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def render_meter_by_meter_plot(series, output_dir: Path, filename: str, title: str, boundaries=None, baseline_eval_per_meter=None):
    ensure_dir(output_dir)
    train_summary = series.get("train", {})
    eval_summary = series.get("eval", {})
    baseline_eval_per_meter = baseline_eval_per_meter or {}
    meters = sorted(
        set(train_summary) | set(eval_summary),
        key=lambda meter: (
            -(
                eval_summary.get(meter, [{}])[-1].get("mean", -1.0)
                if eval_summary.get(meter)
                else -1.0
            ),
            -(
                train_summary.get(meter, [{}])[-1].get("mean", -1.0)
                if train_summary.get(meter)
                else -1.0
            ),
            meter,
        ),
    )
    if not meters:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No per-meter data yet", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        fig.suptitle(title)
        out_path = output_dir / filename
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return out_path

    ncols = 3
    nrows = math.ceil(len(meters) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.6 * nrows), sharex=False, sharey=True)
    axes = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for ax, meter in zip(axes, meters):
        train_points = train_summary.get(meter, [])
        eval_points = eval_summary.get(meter, [])
        baseline_value = baseline_eval_per_meter.get(meter)

        all_steps = [point["step"] for point in train_points] + [point["step"] for point in eval_points]
        if all_steps:
            min_step = min(all_steps)
            max_step = max(all_steps)
            baseline_step = max(0.0, float(min_step) - max(10.0, 0.05 * max(1.0, float(max_step) - float(min_step) + 1.0)))
        else:
            baseline_step = 0.0

        if train_points:
            train_steps = [point["step"] for point in train_points]
            train_means = [point["mean"] for point in train_points]
            draw_series_with_aggregate(
                ax,
                train_steps,
                train_means,
                raw_label="train raw",
                aggregate_label=f"train aggregate ({max(1, min(TRAIN_SMOOTH_WINDOW, len(train_means)))}-step mean)",
                aggregate_window=TRAIN_SMOOTH_WINDOW,
                raw_linewidth=1.5,
            )
        if eval_points:
            eval_steps = [point["step"] for point in eval_points]
            eval_means = [point["mean"] for point in eval_points]
            draw_series_with_aggregate(
                ax,
                eval_steps,
                eval_means,
                raw_label="eval raw",
                raw_color="#ff7f0e",
                raw_marker="s",
                aggregate_label=f"eval aggregate ({max(1, min(EVAL_SMOOTH_WINDOW, len(eval_means)))}-step mean)",
                aggregate_window=EVAL_SMOOTH_WINDOW,
                aggregate_linestyle="--",
                raw_linewidth=1.5,
            )
        if baseline_value is not None:
            ax.scatter(
                [baseline_step],
                [baseline_value],
                marker="*",
                s=90,
                color="#d62728",
                edgecolors="black",
                linewidths=0.4,
                zorder=5,
                label="SFT before GRPO",
            )

        if not train_points and not eval_points:
            ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes, ha="center", va="center")

        ax.set_title(meter)
        ax.set_xlabel("Chain step")
        ax.set_ylim(-0.02, 1.02)
        draw_boundaries(ax, boundaries)
        ax.grid(True, alpha=0.25)

    for idx, ax in enumerate(axes):
        if idx >= len(meters):
            ax.axis("off")
            continue
        if idx % ncols == 0:
            ax.set_ylabel("Average reward_meter")

    legend_handles = []
    legend_labels = []
    if meters:
        first_ax = axes[0]
        legend_handles, legend_labels = first_ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_chain_artifacts(run_dir: Path, run_specs, rows, output_dir: Path):
    chain_runs_path = output_dir / "chain_runs.json"
    chain_metrics_path = output_dir / "chain_metrics.jsonl"
    with chain_runs_path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "run_id": str(spec["lineage"].get("run_id", Path(spec["run_dir"]).name)),
                    "run_dir": str(spec["run_dir"]),
                    "run_sequence_index": int(spec["lineage"].get("run_sequence_index", 0) or 0),
                    "chain_id": str(spec["lineage"].get("chain_id", "")),
                    "min_step": spec.get("min_step"),
                    "max_step": spec.get("max_step"),
                    "repeat_penalty_visible": bool(spec.get("repeat_penalty_visible", True)),
                }
                for spec in run_specs
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    with chain_metrics_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Render GRPO reward plots from one run or a whole resume chain")
    parser.add_argument("--run-dir", required=True, help="GRPO run directory")
    parser.add_argument("--output-dir", default="", help="Optional output directory; defaults to <run_dir>/plots")
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--follow-chain", action="store_true")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Render only train-focused plots and skip eval-derived panels",
    )
    parser.add_argument("--suppress-boundaries", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else run_dir / "plots"

    def render_once():
        ensure_dir(output_dir)
        single_specs = build_plot_specs(run_dir)
        single_rows = load_metrics_for_runs(single_specs)
        single_spreads = load_generation_spreads(single_specs)
        chain_specs = build_plot_specs(run_dir) if args.follow_chain else single_specs
        chain_rows = load_metrics_for_runs(chain_specs)
        chain_spreads = load_generation_spreads(chain_specs)
        chain_boundaries = [] if args.suppress_boundaries else compute_chain_boundaries(chain_rows)
        train_metric_specs = select_component_metrics(single_rows, "train")
        chain_train_metric_specs = select_component_metrics(chain_rows, "train")
        chain_title_prefix = "Chain" if args.follow_chain else "Run"

        run_train_plot = render_component_panels(
            single_rows,
            single_spreads,
            output_dir,
            "reward_panels_train_run.png",
            "Run Train Rewards",
            "train",
            train_metric_specs,
        )
        chain_train_plot = render_component_panels(
            chain_rows,
            chain_spreads,
            output_dir,
            "reward_panels_train_chain.png",
            f"{chain_title_prefix} Train Rewards",
            "train",
            chain_train_metric_specs,
            chain_boundaries,
        )
        run_kl_plot = render_kl_plot(
            single_rows,
            output_dir,
            "kl_run.png",
            "Run KL",
        )
        chain_kl_plot = render_kl_plot(
            chain_rows,
            output_dir,
            "kl_chain.png",
            f"{chain_title_prefix} KL",
            chain_boundaries,
        )

        outputs = [
            run_train_plot,
            chain_train_plot,
            run_kl_plot,
            chain_kl_plot,
        ]

        if not args.train_only:
            baseline_eval_per_meter = discover_baseline_eval_per_meter(run_dir)
            trained_meter_filter = discover_trained_meter_filter(run_dir)
            single_eval_spreads = load_generation_spreads(single_specs, eval_meter_filter=trained_meter_filter)
            single_gate_series = load_arabic_gate_series(single_specs, eval_meter_filter=trained_meter_filter)
            single_template_series = load_template_quality_series(single_specs, eval_meter_filter=trained_meter_filter)
            single_meter_by_meter = load_meter_by_meter_series(single_specs, eval_meter_filter=trained_meter_filter)
            single_eval_rows = load_filtered_eval_metric_rows(single_specs, eval_meter_filter=trained_meter_filter)
            chain_eval_spreads = load_generation_spreads(chain_specs, eval_meter_filter=trained_meter_filter)
            chain_gate_series = load_arabic_gate_series(chain_specs, eval_meter_filter=trained_meter_filter)
            chain_template_series = load_template_quality_series(chain_specs, eval_meter_filter=trained_meter_filter)
            chain_meter_by_meter = load_meter_by_meter_series(chain_specs, eval_meter_filter=trained_meter_filter)
            chain_eval_rows = load_filtered_eval_metric_rows(chain_specs, eval_meter_filter=trained_meter_filter)
            eval_metric_specs = select_component_metrics(single_eval_rows, "eval")
            chain_eval_metric_specs = select_component_metrics(chain_eval_rows, "eval")

            run_eval_plot = render_component_panels(
                single_eval_rows,
                single_eval_spreads,
                output_dir,
                "reward_panels_eval_run.png",
                "Run Eval Rewards (trained meters only)",
                "eval",
                eval_metric_specs,
            )
            chain_eval_plot = render_component_panels(
                chain_eval_rows,
                chain_eval_spreads,
                output_dir,
                "reward_panels_eval_chain.png",
                f"{chain_title_prefix} Eval Rewards (trained meters only)",
                "eval",
                chain_eval_metric_specs,
                chain_boundaries,
            )
            run_arabic_gate_plot = render_arabic_gate_plot(
                single_gate_series,
                output_dir,
                "arabic_gate_run.png",
                "Run Arabic Gate",
            )
            chain_arabic_gate_plot = render_arabic_gate_plot(
                chain_gate_series,
                output_dir,
                "arabic_gate_chain.png",
                f"{chain_title_prefix} Arabic Gate",
                chain_boundaries,
            )
            run_template_quality_plot = render_template_quality_plot(
                single_template_series,
                output_dir,
                "template_quality_run.png",
                "Run Template Quality",
            )
            chain_template_quality_plot = render_template_quality_plot(
                chain_template_series,
                output_dir,
                "template_quality_chain.png",
                f"{chain_title_prefix} Template Quality",
                chain_boundaries,
            )
            run_meter_by_meter_plot = render_meter_by_meter_plot(
                single_meter_by_meter,
                output_dir,
                "meter_by_meter_run.png",
                "Run Meter By Meter (trained meters)",
                baseline_eval_per_meter=baseline_eval_per_meter,
            )
            chain_meter_by_meter_plot = render_meter_by_meter_plot(
                chain_meter_by_meter,
                output_dir,
                "meter_by_meter_chain.png",
                f"{chain_title_prefix} Meter By Meter (trained meters)",
                chain_boundaries,
                baseline_eval_per_meter=baseline_eval_per_meter,
            )
            outputs.extend(
                [
                    run_eval_plot,
                    chain_eval_plot,
                    run_arabic_gate_plot,
                    chain_arabic_gate_plot,
                    run_template_quality_plot,
                    chain_template_quality_plot,
                    run_meter_by_meter_plot,
                    chain_meter_by_meter_plot,
                ]
            )

        write_chain_artifacts(run_dir, chain_specs, chain_rows, output_dir)
        return outputs

    if args.watch:
        while True:
            outputs = render_once()
            for plot_path in outputs:
                print(f"[plot_live_rewards] updated {plot_path}")
            time.sleep(max(1, args.interval_seconds))
    else:
        outputs = render_once()
        for plot_path in outputs:
            print(plot_path)


if __name__ == "__main__":
    main()
