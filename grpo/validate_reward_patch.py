import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv

from rewards.common import (
    ensure_dir,
    extract_text,
    load_and_prepare_dataset,
    save_json,
    score_arabic_cleanliness,
    score_count_adherence,
    score_friend_reward_floors,
    score_repetition_penalty,
)
from rewards.judge_quality import batch_score_judge_quality
from rewards.meter import score_meter_poem


ROOT = Path(__file__).resolve().parent


def load_cfg():
    with open(ROOT / "grpo_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def score_candidate(
    text: str,
    description: str,
    meter_label: str,
    base_meter: str,
    requested_bayts: int,
    *,
    judge_prompt_file: str,
    judge_cache_dir: str,
):
    meter_out = score_meter_poem(text, meter_label, base_meter=base_meter, aggregator="logmean")
    count_out = score_count_adherence(int(requested_bayts or 0), text)
    clean_out = score_arabic_cleanliness(text)
    repeat_out = score_repetition_penalty(text)
    judge_out = batch_score_judge_quality(
        descriptions=[str(description or "")],
        poems=[text],
        prompt_file=judge_prompt_file,
        cache_dir=judge_cache_dir,
        cache_namespace="judge_quality_validation",
        max_workers=1,
    )[0]
    floor_out = score_friend_reward_floors(
        generated_poem=text,
        generated_bayts=int(count_out.get("generated_bayts", 0) or 0),
        count_adherence_score=float(count_out["score"]),
        near_duplicate_score=float(repeat_out["near_duplicate_score"]),
        distinct_2_score=float(repeat_out["distinct_2_score"]),
        opening_diversity_score=float(repeat_out["opening_diversity_score"]),
    )
    total = (
        float(floor_out["hard_gate_score"])
        * float(meter_out["score"])
        * float(judge_out["score"])
        * float(floor_out["arabic_floor_score"])
        * float(floor_out["count_floor_score"])
        * float(floor_out["repeat_floor_score"])
    )
    return {
        "reward_total": total,
        "reward_meter": float(meter_out["score"]),
        "reward_count_adherence": float(count_out["score"]),
        "reward_arabic_clean": float(clean_out["score"]),
        "reward_repeat_penalty": float(repeat_out["score"]),
        "reward_judge_quality": float(judge_out["score"]),
        "reward_hard_gate": float(floor_out["hard_gate_score"]),
        "reward_arabic_floor": float(floor_out["arabic_floor_score"]),
        "reward_count_floor": float(floor_out["count_floor_score"]),
        "reward_repeat_floor": float(floor_out["repeat_floor_score"]),
        "reward_repeat_soft": float(floor_out["repeat_soft_score"]),
        "lexical_plausibility_score": float(clean_out["lexical_plausibility_score"]),
        "artifact_free_score": float(clean_out["artifact_free_score"]),
        "contains_digits": bool(clean_out["contains_digits"]),
        "contains_latin": bool(clean_out["contains_latin"]),
        "hard_gate_reason": str(floor_out["hard_gate_reason"]),
        "contamination_ratio": float(floor_out["contamination_ratio"]),
        "non_arabic_non_punct_count": int(floor_out["non_arabic_non_punct_count"]),
        "near_duplicate_score": float(repeat_out["near_duplicate_score"]),
        "opening_diversity_score": float(repeat_out["opening_diversity_score"]),
        "distinct_2_score": float(repeat_out["distinct_2_score"]),
        "near_duplicate_pair_fraction": float(repeat_out["near_duplicate_pair_fraction"]),
        "opening_dominance_fraction": float(repeat_out["opening_dominance_fraction"]),
        "line_pair_similarity_mean": float(repeat_out["line_pair_similarity_mean"]),
    }


def pick_hacked_rows(run_dir: Path, step: int):
    rows = []
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("dataset_split", row.get("mode", ""))) != "eval":
            continue
        if int(row.get("global_step", -1) or -1) != int(step):
            continue
        rows.append(row)
    best_by_meter = {}
    for row in rows:
        meter = str(row.get("base_meter", "")).strip()
        if not meter:
            continue
        score = float(row.get("reward_total", 0.0) or 0.0)
        current = best_by_meter.get(meter)
        if current is None or score > float(current.get("reward_total", 0.0) or 0.0):
            best_by_meter[meter] = row
    return [best_by_meter[meter] for meter in sorted(best_by_meter)]


def pick_golden_rows(dataset_id: str, *, per_meter: int, hf_token: str | None):
    prepared = list(load_and_prepare_dataset(dataset_id=dataset_id, split="train", max_bayts=20, allowed_meters=None, hf_token=hf_token))
    buckets = defaultdict(list)
    for row in prepared:
        meter = str(row.get("base_meter", "")).strip()
        if meter:
            buckets[meter].append(row)
    selected = []
    for meter in sorted(buckets):
        selected.extend(buckets[meter][:per_meter])
    return selected


def summarize(rows):
    return {
        "count": len(rows),
        "reward_total_mean": mean(row["reward_total"] for row in rows),
        "reward_meter_mean": mean(row["reward_meter"] for row in rows),
        "reward_count_adherence_mean": mean(row["reward_count_adherence"] for row in rows),
        "reward_arabic_clean_mean": mean(row["reward_arabic_clean"] for row in rows),
        "reward_hard_gate_mean": mean(row["reward_hard_gate"] for row in rows),
        "reward_arabic_floor_mean": mean(row["reward_arabic_floor"] for row in rows),
        "reward_count_floor_mean": mean(row["reward_count_floor"] for row in rows),
        "reward_repeat_floor_mean": mean(row["reward_repeat_floor"] for row in rows),
        "reward_repeat_penalty_mean": mean(row["reward_repeat_penalty"] for row in rows),
        "reward_judge_quality_mean": mean(row["reward_judge_quality"] for row in rows),
        "lexical_plausibility_mean": mean(row["lexical_plausibility_score"] for row in rows),
        "artifact_free_mean": mean(row["artifact_free_score"] for row in rows),
        "contamination_ratio_mean": mean(row["contamination_ratio"] for row in rows),
        "contains_digits_rate": mean(1.0 if row["contains_digits"] else 0.0 for row in rows),
        "contains_latin_rate": mean(1.0 if row["contains_latin"] else 0.0 for row in rows),
        "near_duplicate_score_mean": mean(row["near_duplicate_score"] for row in rows),
        "opening_diversity_mean": mean(row["opening_diversity_score"] for row in rows),
        "distinct_2_mean": mean(row["distinct_2_score"] for row in rows),
    }


def row_markdown(row):
    return "\n".join(
        [
            f"- meter: `{row['base_meter']}`",
            f"- total: `{row['reward_total']:.4f}`",
            f"- meter score: `{row['reward_meter']:.4f}`",
            f"- count adherence: `{row['reward_count_adherence']:.4f}`",
            f"- arabic clean: `{row['reward_arabic_clean']:.4f}`",
            f"- hard gate: `{row['reward_hard_gate']:.4f}`",
            f"- arabic floor: `{row['reward_arabic_floor']:.4f}`",
            f"- count floor: `{row['reward_count_floor']:.4f}`",
            f"- repeat floor: `{row['reward_repeat_floor']:.4f}`",
            f"- judge quality: `{row['reward_judge_quality']:.4f}`",
            f"- lexical plausibility: `{row['lexical_plausibility_score']:.4f}`",
            f"- repeat penalty: `{row['reward_repeat_penalty']:.4f}`",
            f"- near-duplicate score: `{row['near_duplicate_score']:.4f}`",
            f"- opening diversity: `{row['opening_diversity_score']:.4f}`",
            f"- distinct-2: `{row['distinct_2_score']:.4f}`",
            f"- contamination ratio: `{row['contamination_ratio']:.4f}`",
            f"- contains digits: `{row['contains_digits']}`",
            f"- hard gate reason: `{row['hard_gate_reason']}`",
            "",
            "```text",
            str(row["text"]).strip(),
            "```",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Validate patched reward on hacked generations vs golden poems.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, default=3100)
    parser.add_argument("--golden-per-meter", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(ROOT.parent / ".env", override=False)
    hf_token = (Path(ROOT.parent / ".env").exists() and __import__("os").getenv("HF_TOKEN", "").strip()) or None
    cfg = load_cfg()
    judge_prompt_file_cfg = str(cfg["phase1"].get("judge_quality_prompt_file", "prompts/judge_quality_selected.yaml")).strip()
    judge_prompt_file = Path(judge_prompt_file_cfg)
    if not judge_prompt_file.is_absolute():
        judge_prompt_file = ROOT / judge_prompt_file
    judge_cache_dir_cfg = str(cfg["phase1"].get("judge_quality_cache_dir", ROOT / "outputs" / "judge_quality_cache")).strip()
    judge_cache_dir = Path(judge_cache_dir_cfg)
    if not judge_cache_dir.is_absolute():
        judge_cache_dir = ROOT / judge_cache_dir
    run_dir = Path(args.run_dir).resolve()
    out_dir = run_dir / "reward_patch_validation"
    ensure_dir(out_dir)

    dataset_id = str(cfg["dataset"]["source_dataset_id"])
    hacked = []
    for raw in pick_hacked_rows(run_dir, args.step):
        text = extract_text(raw.get("completion_text", ""))
        scored = score_candidate(
            text=text,
            description=str(raw.get("description_preview", "") or ""),
            meter_label=str(raw.get("meter_label", "")),
            base_meter=str(raw.get("base_meter", "")),
            requested_bayts=int(raw.get("requested_bayts", 0) or 0),
            judge_prompt_file=str(judge_prompt_file),
            judge_cache_dir=str(judge_cache_dir),
        )
        hacked.append(
            {
                "group": "hacked_step_3100",
                "base_meter": str(raw.get("base_meter", "")),
                "meter_label": str(raw.get("meter_label", "")),
                "requested_bayts": int(raw.get("requested_bayts", 0) or 0),
                "text": text,
                **scored,
            }
        )

    golden = []
    for raw in pick_golden_rows(dataset_id, per_meter=args.golden_per_meter, hf_token=hf_token):
        text = extract_text(raw.get("poem_text", ""))
        scored = score_candidate(
            text=text,
            description=str(raw.get("description", "") or ""),
            meter_label=str(raw.get("meter_label", "")),
            base_meter=str(raw.get("base_meter", "")),
            requested_bayts=int(raw.get("requested_bayts", 0) or 0),
            judge_prompt_file=str(judge_prompt_file),
            judge_cache_dir=str(judge_cache_dir),
        )
        golden.append(
            {
                "group": "golden_dataset",
                "base_meter": str(raw.get("base_meter", "")),
                "meter_label": str(raw.get("meter_label", "")),
                "requested_bayts": int(raw.get("requested_bayts", 0) or 0),
                "text": text,
                **scored,
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "step": int(args.step),
        "dataset_id": dataset_id,
        "hacked_summary": summarize(hacked),
        "golden_summary": summarize(golden),
    }
    save_json(summary, out_dir / "summary.json")
    with (out_dir / "scored_rows.jsonl").open("w", encoding="utf-8") as f:
        for row in hacked + golden:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Reward Patch Validation",
        "",
        f"- run dir: `{run_dir}`",
        f"- hacked eval step: `{args.step}`",
        f"- golden source dataset: `{dataset_id}`",
        "",
        "## Summary",
        "",
        f"- hacked total mean: `{summary['hacked_summary']['reward_total_mean']:.4f}`",
        f"- hacked arabic clean mean: `{summary['hacked_summary']['reward_arabic_clean_mean']:.4f}`",
        f"- hacked repeat mean: `{summary['hacked_summary']['reward_repeat_penalty_mean']:.4f}`",
        f"- hacked digit leakage rate: `{summary['hacked_summary']['contains_digits_rate']:.4f}`",
        "",
        f"- golden total mean: `{summary['golden_summary']['reward_total_mean']:.4f}`",
        f"- golden arabic clean mean: `{summary['golden_summary']['reward_arabic_clean_mean']:.4f}`",
        f"- golden repeat mean: `{summary['golden_summary']['reward_repeat_penalty_mean']:.4f}`",
        f"- golden digit leakage rate: `{summary['golden_summary']['contains_digits_rate']:.4f}`",
        "",
        "## Hacked Examples",
        "",
    ]
    for row in hacked[:6]:
        lines.append(f"### {row['base_meter']}")
        lines.append("")
        lines.append(row_markdown(row))
        lines.append("")
    lines.append("## Golden Examples")
    lines.append("")
    golden_best = sorted(golden, key=lambda row: row["reward_total"], reverse=True)
    for row in golden_best[:6]:
        lines.append(f"### {row['base_meter']}")
        lines.append("")
        lines.append(row_markdown(row))
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
