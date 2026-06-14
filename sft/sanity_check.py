from __future__ import annotations

import sys

from train_sft import run_training


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "sanity"])
    run_training()
