#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from common import OUTPUTS_ROOT, read_jsonl, utc_now_iso
from model_registry import parse_model_list


DEFAULT_MODELS = "fanar_2_diwan_prefix,ashaar_model,gpt2_small_arabic_poetry,gpt2_medium_arabic_poetry"
DEFAULT_MANIFEST = OUTPUTS_ROOT / "full_baseline_manifest_unique_prompts.jsonl"


@dataclass(frozen=True)
class RunpodModelConfig:
    batch_size: int
    trust_remote_code: bool = False


A40_MODEL_CONFIGS = {
    "fanar_2_diwan_prefix": RunpodModelConfig(batch_size=4, trust_remote_code=True),
    "ashaar_model": RunpodModelConfig(batch_size=16),
    "gpt2_small_arabic_poetry": RunpodModelConfig(batch_size=32),
    "gpt2_medium_arabic_poetry": RunpodModelConfig(batch_size=16),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A40-safe continuation baseline batch on RunPod.")
    parser.add_argument("--subset-jsonl", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--samples-per-row", type=int, default=1)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument(
        "--batch-sizes",
        default="",
        help="Optional comma-separated overrides, e.g. fanar_2_diwan_prefix=8,gpt2_small_arabic_poetry=64",
    )
    parser.add_argument(
        "--batch-multiplier",
        type=float,
        default=1.0,
        help="Multiply the selected/default per-model batch sizes, useful when tuning a larger GPU.",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    subset_jsonl = Path(args.subset_jsonl)
    if not subset_jsonl.is_absolute():
        subset_jsonl = project_root / subset_jsonl
    output_root = resolve_output_root(args.output_root)

    ensure_manifest(script_dir, project_root, subset_jsonl, bool(args.rebuild_manifest))
    manifest_rows = len(read_jsonl(subset_jsonl))
    expected_source_rows = int(args.limit_rows) if args.limit_rows else manifest_rows
    model_specs = parse_model_list(args.models)
    selected_models = [spec.name for spec in model_specs if spec.group == "continuation"]
    if not selected_models:
        raise RuntimeError("No continuation models were selected.")

    print(f"subset_jsonl={subset_jsonl}")
    print(f"output_root={output_root}")
    print(f"expected_source_rows={expected_source_rows}")
    print(f"models={','.join(selected_models)}")
    print("batch_sizes=" + ",".join(f"{name}={resolve_batch_size(args, name)}" for name in selected_models))

    scored_paths: list[Path] = []
    failures: list[tuple[str, int]] = []
    for model_name in selected_models:
        try:
            scored_paths.append(run_one_model(args, script_dir, project_root, subset_jsonl, output_root, model_name, expected_source_rows))
        except subprocess.CalledProcessError as exc:
            failures.append((model_name, int(exc.returncode)))
            print(f"model_failed={model_name} returncode={exc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                raise

    if scored_paths:
        combine_outputs(script_dir, project_root, output_root, scored_paths)
    if failures:
        return 1
    return 0


def resolve_output_root(output_root: str) -> Path:
    if output_root:
        path = Path(output_root)
    else:
        timestamp = utc_now_iso().replace(":", "").replace("-", "")
        path = OUTPUTS_ROOT / f"runpod_continuation_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_manifest(script_dir: Path, project_root: Path, subset_jsonl: Path, rebuild: bool) -> None:
    if subset_jsonl.exists() and not rebuild:
        return
    subset_jsonl.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(script_dir / "build_full_baseline_manifest.py"),
            "--continuation-policy",
            "auto",
            "--per-source",
            "first",
            "--scored-repo",
            "",
            "--output-jsonl",
            str(subset_jsonl),
        ],
        project_root,
    )


def resolve_batch_size(args: argparse.Namespace, model_name: str) -> int:
    overrides = parse_batch_size_overrides(str(getattr(args, "batch_sizes", "") or ""))
    if model_name in overrides:
        return max(1, overrides[model_name])
    config = A40_MODEL_CONFIGS.get(model_name, RunpodModelConfig(batch_size=1))
    multiplier = max(0.01, float(getattr(args, "batch_multiplier", 1.0) or 1.0))
    return max(1, int(round(config.batch_size * multiplier)))


def parse_batch_size_overrides(raw_value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --batch-sizes item '{item}'. Expected model_name=batch_size.")
        model_name, value = item.split("=", 1)
        model_name = model_name.strip()
        if not model_name:
            raise ValueError(f"Invalid --batch-sizes item '{item}'. Missing model name.")
        overrides[model_name] = int(value.strip())
    return overrides


def run_one_model(
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    subset_jsonl: Path,
    output_root: Path,
    model_name: str,
    expected_source_rows: int,
) -> Path:
    config = A40_MODEL_CONFIGS.get(model_name, RunpodModelConfig(batch_size=1))
    batch_size = resolve_batch_size(args, model_name)
    run_dir = output_root / model_name
    generation_path = run_dir / "continuation_generations.jsonl"
    validation_path = run_dir / "continuation_generations.validation.json"
    scored_path = run_dir / "continuation_generations_scored.jsonl"
    aggregate_path = run_dir / "continuation_generations_aggregate.json"

    generation_cmd = [
        sys.executable,
        str(script_dir / "generate_continuation_baselines.py"),
        "--subset-jsonl",
        str(subset_jsonl),
        "--models",
        model_name,
        "--run-dir",
        str(run_dir),
        "--samples-per-row",
        str(int(args.samples_per_row)),
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--temperature",
        str(float(args.temperature)),
        "--top-p",
        str(float(args.top_p)),
        "--repetition-penalty",
        str(float(args.repetition_penalty)),
    ]
    if args.limit_rows:
        generation_cmd.extend(["--limit-rows", str(int(args.limit_rows))])
    if config.trust_remote_code:
        generation_cmd.append("--trust-remote-code")
    run(generation_cmd, project_root)

    run(
        [
            sys.executable,
            str(script_dir / "validate_baseline_generations.py"),
            "--input-jsonl",
            str(generation_path),
            "--expected-models",
            model_name,
            "--expected-source-rows",
            str(expected_source_rows),
            "--samples-per-row",
            str(int(args.samples_per_row)),
            "--summary-json",
            str(validation_path),
        ],
        project_root,
    )

    if args.skip_score:
        return generation_path
    run(
        [
            sys.executable,
            str(script_dir / "score_baseline_meter_count.py"),
            "--input-jsonl",
            str(generation_path),
            "--output-jsonl",
            str(scored_path),
        ],
        project_root,
    )
    run(
        [
            sys.executable,
            str(script_dir / "aggregate_baseline_results.py"),
            "--input-jsonl",
            str(scored_path),
            "--output-json",
            str(aggregate_path),
        ],
        project_root,
    )
    return scored_path


def combine_outputs(script_dir: Path, project_root: Path, output_root: Path, scored_paths: list[Path]) -> None:
    run(
        [
            sys.executable,
            str(script_dir / "combine_model_results.py"),
            "--inputs",
            ",".join(str(path) for path in scored_paths),
            "--output-jsonl",
            str(output_root / "combined_continuation_models.jsonl"),
            "--output-csv",
            str(output_root / "combined_continuation_models.csv"),
            "--wide-csv",
            str(output_root / "combined_continuation_models_wide.csv"),
            "--summary-json",
            str(output_root / "combined_continuation_models.summary.json"),
        ],
        project_root,
    )


def run(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


if __name__ == "__main__":
    raise SystemExit(main())
