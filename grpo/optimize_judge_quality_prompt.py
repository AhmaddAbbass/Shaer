from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from rewards.common import ensure_dir, row_to_description, row_to_poem_text
from rewards.judge_quality import batch_score_judge_quality


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
DEFAULT_HACKED_RUN_DIR = ROOT / "outputs" / "train" / "shaer_grpo_20260411_223409"
DEFAULT_HACKED_STEP = 3100


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_hacked_examples(run_dir: Path, step: int):
    rows = []
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", "")) != "eval":
            continue
        if int(row.get("global_step", -1) or -1) != int(step):
            continue
        rows.append(row)

    per_meter_best = {}
    for row in rows:
        meter = str(row.get("base_meter", "")).strip()
        if not meter:
            continue
        current = per_meter_best.get(meter)
        if current is None or float(row.get("reward_total", 0.0) or 0.0) > float(current.get("reward_total", 0.0) or 0.0):
            per_meter_best[meter] = row

    examples = []
    for meter in sorted(per_meter_best):
        row = per_meter_best[meter]
        examples.append(
            {
                "label": "hacked",
                "meter": meter,
                "description": str(row.get("description_preview", "") or ""),
                "poem": str(row.get("completion_text", "") or ""),
                "source": f"{run_dir.name}:step{step}",
                "reward_total": float(row.get("reward_total", 0.0) or 0.0),
            }
        )
    return examples


def select_golden_examples(dataset_id: str, per_meter: int):
    ds = load_dataset(dataset_id, split="train")
    buckets = defaultdict(list)
    for row in ds:
        meter = str(row.get("base_meter", "")).strip()
        poem = row_to_poem_text(row)
        description = row_to_description(row)
        if not meter or not poem or not description:
            continue
        if len(buckets[meter]) >= per_meter:
            continue
        buckets[meter].append(
            {
                "label": "golden",
                "meter": meter,
                "description": description,
                "poem": poem,
                "source": dataset_id,
                "reward_total": None,
            }
        )
    examples = []
    for meter in sorted(buckets):
        examples.extend(buckets[meter])
    return examples


def evaluate_prompt(prompt_file: Path, examples: list[dict], cache_dir: Path, max_workers: int):
    descriptions = [row["description"] for row in examples]
    poems = [row["poem"] for row in examples]
    scored = batch_score_judge_quality(
        descriptions=descriptions,
        poems=poems,
        prompt_file=str(prompt_file),
        cache_dir=str(cache_dir),
        cache_namespace=f"judge_quality_search_{prompt_file.stem}",
        max_workers=max_workers,
    )
    rows = []
    for example, score_out in zip(examples, scored):
        rows.append(
            {
                **example,
                "judge_quality": float(score_out["score"]),
                "failure_mode": score_out["failure_mode"],
                "notes": score_out["notes"],
                "cache_hit": bool(score_out["cache_hit"]),
                "latency_sec": score_out["latency_sec"],
                "error": score_out["error"],
                "prompt_name": score_out["prompt_name"],
                "prompt_file": score_out["prompt_file"],
            }
        )
    return rows


def summarize(rows: list[dict]):
    golden = [row["judge_quality"] for row in rows if row["label"] == "golden"]
    hacked = [row["judge_quality"] for row in rows if row["label"] == "hacked"]
    accuracy_at_05 = sum(
        1
        for row in rows
        if (row["judge_quality"] >= 0.5 and row["label"] == "golden")
        or (row["judge_quality"] < 0.5 and row["label"] == "hacked")
    ) / max(1, len(rows))
    golden_accept_rate = sum(1 for value in golden if value >= 0.7) / max(1, len(golden))
    hacked_reject_rate = sum(1 for value in hacked if value <= 0.3) / max(1, len(hacked))
    golden_mean = sum(golden) / max(1, len(golden))
    hacked_mean = sum(hacked) / max(1, len(hacked))
    separation_gap = golden_mean - hacked_mean
    objective = separation_gap + 0.25 * accuracy_at_05 + 0.15 * golden_accept_rate + 0.15 * hacked_reject_rate
    return {
        "count": len(rows),
        "golden_count": len(golden),
        "hacked_count": len(hacked),
        "golden_mean": golden_mean,
        "hacked_mean": hacked_mean,
        "separation_gap": separation_gap,
        "accuracy_at_0.5": accuracy_at_05,
        "golden_accept_rate_at_0.7": golden_accept_rate,
        "hacked_reject_rate_at_0.3": hacked_reject_rate,
        "objective": objective,
    }


def write_csv(path: Path, rows: list[dict]):
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-id", default=DEFAULT_SOURCE_DATASET_ID)
    parser.add_argument("--hacked-run-dir", default=str(DEFAULT_HACKED_RUN_DIR))
    parser.add_argument("--hacked-step", type=int, default=DEFAULT_HACKED_STEP)
    parser.add_argument("--golden-per-meter", type=int, default=3)
    parser.add_argument("--prompt-glob", default="prompts/judge_quality_v*.yaml")
    parser.add_argument("--output-root", default="outputs/judge_quality_prompt_search")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    output_root = ensure_dir(ROOT / args.output_root)
    cache_dir = ensure_dir(output_root / "cache")

    hacked = select_hacked_examples(Path(args.hacked_run_dir), args.hacked_step)
    golden = select_golden_examples(args.source_dataset_id, args.golden_per_meter)
    examples = hacked + golden

    prompt_files = sorted(ROOT.glob(args.prompt_glob))
    if not prompt_files:
        raise SystemExit("No prompt files matched")

    all_summaries = []
    best_prompt = None
    best_summary = None
    best_rows = None

    for prompt_file in prompt_files:
        rows = evaluate_prompt(prompt_file, examples, cache_dir, args.max_workers)
        summary = summarize(rows)
        summary["prompt_file"] = str(prompt_file)
        all_summaries.append(summary)
        write_csv(output_root / f"{prompt_file.stem}_rows.csv", rows)
        (output_root / f"{prompt_file.stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if best_summary is None or summary["objective"] > best_summary["objective"]:
            best_prompt = prompt_file
            best_summary = summary
            best_rows = rows

    selected_path = ROOT / "prompts" / "judge_quality_selected.yaml"
    shutil.copyfile(best_prompt, selected_path)
    write_csv(output_root / "selected_rows.csv", best_rows)
    (output_root / "prompt_summaries.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "selected_summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    example_good = next((row for row in best_rows if row["label"] == "golden"), None)
    example_bad = next((row for row in best_rows if row["label"] == "hacked"), None)
    lines = []
    lines.append(f"# Judge Quality Prompt Search")
    lines.append("")
    lines.append(f"- selected prompt: `{best_prompt.name}`")
    lines.append(f"- copied to: `{selected_path}`")
    lines.append(f"- golden count: `{best_summary['golden_count']}`")
    lines.append(f"- hacked count: `{best_summary['hacked_count']}`")
    lines.append(f"- golden mean: `{best_summary['golden_mean']:.4f}`")
    lines.append(f"- hacked mean: `{best_summary['hacked_mean']:.4f}`")
    lines.append(f"- separation gap: `{best_summary['separation_gap']:.4f}`")
    lines.append(f"- accuracy@0.5: `{best_summary['accuracy_at_0.5']:.4f}`")
    lines.append(f"- golden accept rate@0.7: `{best_summary['golden_accept_rate_at_0.7']:.4f}`")
    lines.append(f"- hacked reject rate@0.3: `{best_summary['hacked_reject_rate_at_0.3']:.4f}`")
    lines.append("")
    if example_good:
        lines.append("## Example Golden")
        lines.append("")
        lines.append(f"- meter: `{example_good['meter']}`")
        lines.append(f"- score: `{example_good['judge_quality']:.4f}`")
        lines.append(f"- failure_mode: `{example_good['failure_mode']}`")
        lines.append(f"- notes: `{example_good['notes']}`")
        lines.append("")
        lines.append("```text")
        lines.append(example_good["poem"][:1200])
        lines.append("```")
        lines.append("")
    if example_bad:
        lines.append("## Example Hacked")
        lines.append("")
        lines.append(f"- meter: `{example_bad['meter']}`")
        lines.append(f"- score: `{example_bad['judge_quality']:.4f}`")
        lines.append(f"- failure_mode: `{example_bad['failure_mode']}`")
        lines.append(f"- notes: `{example_bad['notes']}`")
        lines.append("")
        lines.append("```text")
        lines.append(example_bad["poem"][:1200])
        lines.append("```")
        lines.append("")

    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2))
    print(f"selected_prompt={best_prompt}")
    print(f"selected_copy={selected_path}")


if __name__ == "__main__":
    main()
