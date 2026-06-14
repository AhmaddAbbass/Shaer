#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from judge_common import load_env


DEFAULT_METRICS = [
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all frozen judge metrics on the reference calibration bank.")
    parser.add_argument("--input-jsonl", default="evaluation/outputs/judge_reference_calibration/reference_rows.jsonl")
    parser.add_argument("--output-dir", default="evaluation/outputs/judge_reference_calibration")
    parser.add_argument("--poem-field", default="reference_poem")
    parser.add_argument("--description-field", default="enhanced_description")
    parser.add_argument("--id-field", default="calibration_id")
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4.6")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric in DEFAULT_METRICS:
        output_jsonl = output_dir / f"{metric}.jsonl"
        cmd = [
            "python",
            "evaluation/judge_llm.py",
            "--input-jsonl",
            args.input_jsonl,
            "--output-jsonl",
            str(output_jsonl),
            "--metric",
            metric,
            "--poem-field",
            args.poem_field,
            "--description-field",
            args.description_field,
            "--id-field",
            args.id_field,
            "--judge-model",
            args.judge_model,
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        run_checked(cmd)
        if not args.dry_run:
            summary_json = output_dir / f"{metric}.summary.json"
            run_checked(
                [
                    "python",
                    "evaluation/summarize_judge_scores.py",
                    "--input-jsonl",
                    str(output_jsonl),
                    "--output-json",
                    str(summary_json),
                ]
            )
    return 0


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
