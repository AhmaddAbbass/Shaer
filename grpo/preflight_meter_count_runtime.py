#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from preprocess_meter_count_common import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_DATASET_ID,
    DEFAULT_SFT_ADAPTER_REPO,
    load_dotenv_if_present,
    utc_now_iso,
)


REQUIRED_IMPORTS = ["datasets", "huggingface_hub", "torch", "transformers", "peft", "vllm", "dotenv"]


def has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def gpu_snapshot() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    load_dotenv_if_present()
    imports = {name: has_import(name) for name in REQUIRED_IMPORTS}
    env_present = {
        "HF_TOKEN": bool(os.getenv("HF_TOKEN", "").strip()),
        "BASE_MODEL_ID": os.getenv("GRPO_MC_BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID),
        "SFT_ADAPTER_REPO": os.getenv("GRPO_MC_SFT_ADAPTER_REPO", DEFAULT_SFT_ADAPTER_REPO),
        "METER_MODEL_ID": bool(os.getenv("METER_MODEL_ID", "").strip()),
        "GRPO_PREPROCESS_DATASET_ID": os.getenv("GRPO_PREPROCESS_DATASET_ID", DEFAULT_DATASET_ID),
    }
    summary = {
        "timestamp": utc_now_iso(),
        "python": sys.executable,
        "imports": imports,
        "env": env_present,
        "gpus": gpu_snapshot(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    missing_imports = [name for name, ok in imports.items() if not ok]
    missing_env = [name for name in ["METER_MODEL_ID"] if not env_present[name]]
    if missing_imports:
        print(f"missing imports: {missing_imports}", file=sys.stderr)
        return 2
    if missing_env:
        print(f"missing required env: {missing_env}", file=sys.stderr)
        return 3
    if not summary["gpus"]:
        print("nvidia-smi is not available", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
