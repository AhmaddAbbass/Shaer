import copy
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from train_grpo import run_training


if __name__ == "__main__":
    run_training(mode="sanity")