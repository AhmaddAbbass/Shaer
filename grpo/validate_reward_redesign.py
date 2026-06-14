#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

from rewards.common import (
    ensure_dir,
    extract_text,
    row_to_description,
    row_to_poem_text,
    score_arabic_cleanliness,
    score_count_adherence,
    score_minimal_hard_gate,
    score_repeat_soft_signal,
    score_repetition_penalty,
    score_weighted_reward_train,
)
from rewards.judge_quality import batch_score_judge_quality
from rewards.meter import score_meter_poem


ROOT = Path(__file__).resolve().parent
DEFAULT_HACKED_RUN_DIR = ROOT / "outputs" / "train" / "shaer_grpo_20260411_223409"
DEFAULT_HACKED_STEP = 3100


def load_cfg() -> dict[str, Any]:
    with open(ROOT / "grpo_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def summarize_values(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(row[key]) for row in rows)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "min": float(values[0]),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": float(values[-1]),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "reward_total": summarize_values(rows, "reward_total"),
        "meter": summarize_values(rows, "meter"),
        "count_adherence": summarize_values(rows, "count_adherence"),
        "judge_quality": summarize_values(rows, "judge_quality"),
        "repeat_soft": summarize_values(rows, "repeat_soft"),
        "hard_gate_pass_rate": float(sum(1.0 if row["hard_gate"] >= 1.0 else 0.0 for row in rows) / max(1, len(rows))),
        "exact_count_rate": float(sum(1.0 if row["exact_count"] else 0.0 for row in rows) / max(1, len(rows))),
        "latin_leakage_rate": float(sum(1.0 if row["contains_latin"] else 0.0 for row in rows) / max(1, len(rows))),
        "digit_leakage_rate": float(sum(1.0 if row["contains_digits"] else 0.0 for row in rows) / max(1, len(rows))),
        "artifact_leakage_rate": float(sum(1.0 if row["contains_forbidden_artifacts"] else 0.0 for row in rows) / max(1, len(rows))),
    }


def health_decision(golden_rows: list[dict[str, Any]], hacked_rows: list[dict[str, Any]]) -> dict[str, Any]:
    golden = summarize_group(golden_rows)
    hacked = summarize_group(hacked_rows)
    checks = {
        "golden_total_gt_hacked_total_by_0.15": (
            golden["reward_total"]["mean"] - hacked["reward_total"]["mean"]
        ) >= 0.15,
        "golden_judge_gt_hacked_judge_by_0.15": (
            golden["judge_quality"]["mean"] - hacked["judge_quality"]["mean"]
        ) >= 0.15,
        "golden_judge_not_too_harsh": golden["judge_quality"]["mean"] >= 0.50,
        "golden_meter_not_broken": golden["meter"]["mean"] >= 0.70,
        "hacked_total_sufficiently_low": hacked["reward_total"]["mean"] <= 0.45,
    }
    notes = []
    for label, passed in checks.items():
        if not passed:
            notes.append(label)
    return {
        "healthy": all(checks.values()),
        "checks": checks,
        "failed_checks": notes,
    }


def score_rows(rows: list[dict[str, Any]], *, prompt_file: Path, cache_dir: Path, cache_namespace: str) -> list[dict[str, Any]]:
    descriptions = [str(row["description"]) for row in rows]
    poems = [str(row["text"]) for row in rows]
    judge_rows = batch_score_judge_quality(
        descriptions=descriptions,
        poems=poems,
        prompt_file=str(prompt_file),
        cache_dir=str(cache_dir),
        cache_namespace=cache_namespace,
        max_workers=4,
    )

    scored: list[dict[str, Any]] = []
    for row, judge_out in zip(rows, judge_rows):
        meter_out = score_meter_poem(row["text"], row["meter_label"], base_meter=row["base_meter"], aggregator="logmean")
        count_out = score_count_adherence(int(row["requested_bayts"] or 0), row["text"])
        clean_out = score_arabic_cleanliness(row["text"])
        repeat_out = score_repetition_penalty(row["text"])
        hard_gate_out = score_minimal_hard_gate(
            generated_poem=row["text"],
            generated_bayts=int(count_out.get("generated_bayts", 0) or 0),
            clean_out=clean_out,
        )
        repeat_soft = score_repeat_soft_signal(
            exact_repeat_score=float(repeat_out["exact_repeat_score"]),
            near_duplicate_score=float(repeat_out["near_duplicate_score"]),
            opening_diversity_score=float(repeat_out["opening_diversity_score"]),
            distinct_2_score=float(repeat_out["distinct_2_score"]),
        )
        reward_out = score_weighted_reward_train(
            meter_score=float(meter_out["score"]),
            count_adherence_score=float(count_out["score"]),
            judge_quality_score=float(judge_out["score"]),
            repeat_soft_score=repeat_soft,
            hard_gate_score=float(hard_gate_out["hard_gate_score"]),
        )
        scored.append(
            {
                **row,
                "reward_total": float(reward_out["total_score"]),
                "reward_weighted_sum": float(reward_out["weighted_sum"]),
                "meter": float(meter_out["score"]),
                "count_adherence": float(count_out["score"]),
                "judge_quality": float(judge_out["score"]),
                "repeat_soft": float(repeat_soft),
                "repeat_penalty": float(repeat_out["score"]),
                "hard_gate": float(hard_gate_out["hard_gate_score"]),
                "hard_gate_reason": str(hard_gate_out["hard_gate_reason"]),
                "exact_count": bool(int(count_out.get("generated_bayts", -1) or -1) == int(row["requested_bayts"] or 0)),
                "generated_bayts": int(count_out.get("generated_bayts", 0) or 0),
                "contains_latin": bool(clean_out["contains_latin"]),
                "contains_digits": bool(clean_out["contains_digits"]),
                "contains_forbidden_artifacts": bool(clean_out["contains_forbidden_artifacts"]),
                "artifact_free_score": float(clean_out["artifact_free_score"]),
                "contamination_ratio": float(hard_gate_out["contamination_ratio"]),
                "exact_repeat_score": float(repeat_out["exact_repeat_score"]),
                "near_duplicate_score": float(repeat_out["near_duplicate_score"]),
                "opening_diversity_score": float(repeat_out["opening_diversity_score"]),
                "distinct_2_score": float(repeat_out["distinct_2_score"]),
            }
        )
    return scored


def select_hacked_rows(run_dir: Path, step: int, per_meter: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", row.get("dataset_split", ""))) != "eval":
            continue
        if int(row.get("global_step", -1) or -1) != int(step):
            continue
        meter = str(row.get("base_meter", "")).strip()
        if not meter:
            continue
        buckets[meter].append(row)
    selected: list[dict[str, Any]] = []
    for meter in sorted(buckets):
        ordered = sorted(buckets[meter], key=lambda row: float(row.get("reward_total", 0.0) or 0.0), reverse=True)
        for row in ordered[:per_meter]:
            selected.append(
                {
                    "group": "hacked",
                    "source": f"{run_dir.name}:step{step}",
                    "base_meter": meter,
                    "meter_label": str(row.get("meter_label", meter)),
                    "requested_bayts": int(row.get("requested_bayts", 0) or 0),
                    "description": str(row.get("description_preview", "") or ""),
                    "text": extract_text(row.get("completion_text", "")),
                }
            )
    return selected


def select_golden_rows(source_dataset_id: str, manifest_path: Path, per_meter: int) -> list[dict[str, Any]]:
    manifest_rows = read_manifest_rows(manifest_path)
    ordered_indices: list[int] = []
    for row in manifest_rows:
        try:
            ordered_indices.append(int(row.get("source_index", 0) or 0))
        except Exception:
            continue
    needed = set(ordered_indices)
    ds = load_dataset(source_dataset_id, split="train")
    by_index: dict[int, dict[str, Any]] = {}
    for row in ds:
        try:
            source_index = int(row.get("source_index", 0) or 0)
        except Exception:
            continue
        if source_index in needed:
            by_index[source_index] = dict(row)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index in ordered_indices:
        row = by_index.get(source_index)
        if not row:
            continue
        meter = str(row.get("base_meter", "") or "").strip()
        if not meter or len(buckets[meter]) >= per_meter:
            continue
        buckets[meter].append(
            {
                "group": "golden",
                "source": f"{source_dataset_id}:train:{source_index}",
                "base_meter": meter,
                "meter_label": str(row.get("meter_label", meter)),
                "requested_bayts": int(row.get("requested_bayts", 0) or 0),
                "description": row_to_description(row),
                "text": row_to_poem_text(row),
            }
        )
    selected: list[dict[str, Any]] = []
    for meter in sorted(buckets):
        selected.extend(buckets[meter])
    return selected


def representative_sections(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    subset = [row for row in rows if row["group"] == group]
    if not subset:
        return []
    high = sorted(subset, key=lambda row: row["reward_total"], reverse=True)[:3]
    low = sorted(subset, key=lambda row: row["reward_total"])[:3]
    seen = set()
    out = []
    for row in high + low:
        key = (row["base_meter"], row["text"][:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def format_example(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"- meter: `{row['base_meter']}`",
            f"- total: `{row['reward_total']:.4f}`",
            f"- weighted sum pre-gate: `{row['reward_weighted_sum']:.4f}`",
            f"- meter: `{row['meter']:.4f}`",
            f"- count adherence: `{row['count_adherence']:.4f}`",
            f"- judge quality: `{row['judge_quality']:.4f}`",
            f"- repeat soft: `{row['repeat_soft']:.4f}`",
            f"- hard gate: `{row['hard_gate']:.4f}`",
            f"- hard gate reason: `{row['hard_gate_reason']}`",
            "",
            "Description:",
            "```text",
            str(row["description"]).strip()[:1200],
            "```",
            "",
            "Poem:",
            "```text",
            str(row["text"]).strip()[:1200],
            "```",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the new weighted GRPO reward on golden and hacked poems.")
    parser.add_argument("--hacked-run-dir", default=str(DEFAULT_HACKED_RUN_DIR))
    parser.add_argument("--hacked-step", type=int, default=DEFAULT_HACKED_STEP)
    parser.add_argument("--golden-per-meter", type=int, default=3)
    parser.add_argument("--hacked-per-meter", type=int, default=3)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env", override=False)
    cfg = load_cfg()
    manifest_path = Path(cfg["dataset"]["train_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if str(args.prompt_file).strip():
        prompt_file = Path(args.prompt_file).resolve()
    else:
        prompt_file = Path(cfg["phase1"]["judge_quality_prompt_file"])
        if not prompt_file.is_absolute():
            prompt_file = ROOT / prompt_file
    cache_dir = Path(cfg["phase1"]["judge_quality_cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    hacked_run_dir = Path(args.hacked_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else hacked_run_dir / "reward_redesign_validation"
    ensure_dir(output_dir)

    golden_rows = select_golden_rows(str(cfg["dataset"]["source_dataset_id"]), manifest_path, args.golden_per_meter)
    hacked_rows = select_hacked_rows(hacked_run_dir, args.hacked_step, args.hacked_per_meter)

    scored_golden = score_rows(golden_rows, prompt_file=prompt_file, cache_dir=cache_dir, cache_namespace="judge_quality_validation_golden")
    scored_hacked = score_rows(hacked_rows, prompt_file=prompt_file, cache_dir=cache_dir, cache_namespace="judge_quality_validation_hacked")
    all_rows = scored_golden + scored_hacked

    golden_summary = summarize_group(scored_golden)
    hacked_summary = summarize_group(scored_hacked)
    health = health_decision(scored_golden, scored_hacked)
    summary = {
        "hacked_run_dir": str(hacked_run_dir),
        "hacked_step": int(args.hacked_step),
        "train_manifest_path": str(manifest_path),
        "prompt_file": str(prompt_file),
        "golden_summary": golden_summary,
        "hacked_summary": hacked_summary,
        "health": health,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "scored_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Reward Redesign Validation",
        "",
        f"- hacked run dir: `{hacked_run_dir}`",
        f"- hacked step: `{args.hacked_step}`",
        f"- train manifest: `{manifest_path}`",
        f"- judge prompt: `{prompt_file}`",
        "",
        "## Health Decision",
        "",
        f"- healthy: `{health['healthy']}`",
        f"- failed checks: `{', '.join(health['failed_checks']) if health['failed_checks'] else 'none'}`",
        "",
        "## Golden Summary",
        "",
        f"- total mean / median: `{golden_summary['reward_total']['mean']:.4f}` / `{golden_summary['reward_total']['median']:.4f}`",
        f"- meter mean: `{golden_summary['meter']['mean']:.4f}`",
        f"- count mean: `{golden_summary['count_adherence']['mean']:.4f}`",
        f"- judge mean: `{golden_summary['judge_quality']['mean']:.4f}`",
        f"- repeat soft mean: `{golden_summary['repeat_soft']['mean']:.4f}`",
        f"- hard-gate pass rate: `{golden_summary['hard_gate_pass_rate']:.4f}`",
        f"- exact-count rate: `{golden_summary['exact_count_rate']:.4f}`",
        "",
        "## Hacked Summary",
        "",
        f"- total mean / median: `{hacked_summary['reward_total']['mean']:.4f}` / `{hacked_summary['reward_total']['median']:.4f}`",
        f"- meter mean: `{hacked_summary['meter']['mean']:.4f}`",
        f"- count mean: `{hacked_summary['count_adherence']['mean']:.4f}`",
        f"- judge mean: `{hacked_summary['judge_quality']['mean']:.4f}`",
        f"- repeat soft mean: `{hacked_summary['repeat_soft']['mean']:.4f}`",
        f"- hard-gate pass rate: `{hacked_summary['hard_gate_pass_rate']:.4f}`",
        f"- exact-count rate: `{hacked_summary['exact_count_rate']:.4f}`",
        "",
        "## Representative Golden Examples",
        "",
    ]
    for row in representative_sections(all_rows, "golden"):
        lines.append(f"### {row['base_meter']}")
        lines.append("")
        lines.append(format_example(row))
        lines.append("")
    lines.append("## Representative Hacked Examples")
    lines.append("")
    for row in representative_sections(all_rows, "hacked"):
        lines.append(f"### {row['base_meter']}")
        lines.append("")
        lines.append(format_example(row))
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
