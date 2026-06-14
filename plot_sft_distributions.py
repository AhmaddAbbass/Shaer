import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_PREFIX = "Shaer-AI/ashaar-v1-sft-ready-locked-prompt-2048-"


def resolve_dataset_id() -> str:
    explicit = os.getenv("SFT_DATASET_ID", "").strip()
    if explicit:
        return explicit

    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    api = HfApi(token=hf_token)
    items = list(api.list_datasets(author="Shaer-AI", sort="last_modified", limit=100, token=hf_token))
    for item in items:
        if item.id.startswith(DEFAULT_PREFIX):
            return item.id

    raise RuntimeError(
        "Could not resolve SFT dataset ID. Set SFT_DATASET_ID in env explicitly."
    )


def describe(values: np.ndarray, name: str) -> None:
    q = np.quantile(values, [0.5, 0.9, 0.95, 0.99, 1.0])
    print(f"\n{name} stats:")
    print(f"  count={len(values)}")
    print(f"  min={int(values.min())}")
    print(f"  p50={q[0]:.2f}")
    print(f"  p90={q[1]:.2f}")
    print(f"  p95={q[2]:.2f}")
    print(f"  p99={q[3]:.2f}")
    print(f"  max={int(q[4])}")


def make_hist(
    values: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int = 60,
) -> None:
    q95 = float(np.quantile(values, 0.95))
    q99 = float(np.quantile(values, 0.99))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(values, bins=bins, color="#1f77b4", edgecolor="white", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.axvline(q95, color="#ff7f0e", linestyle="--", linewidth=2, label=f"p95={q95:.1f}")
    ax.axvline(q99, color="#d62728", linestyle="--", linewidth=2, label=f"p99={q99:.1f}")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)

    dataset_id = resolve_dataset_id()
    print(f"Using dataset: {dataset_id}")

    ds = load_dataset(dataset_id, split="train")
    required = {"sft_num_lines", "sft_total_tokens"}
    missing = required.difference(set(ds.column_names))
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    num_lines = np.asarray(ds["sft_num_lines"], dtype=np.int32)
    total_tokens = np.asarray(ds["sft_total_tokens"], dtype=np.int32)

    describe(num_lines, "sft_num_lines")
    describe(total_tokens, "sft_total_tokens")

    lines_path = PROJECT_DIR / "sft_num_lines_distribution.png"
    tokens_path = PROJECT_DIR / "sft_total_tokens_distribution.png"

    make_hist(
        values=num_lines,
        title="SFT Num Lines Distribution",
        xlabel="sft_num_lines",
        output_path=lines_path,
        bins=80,
    )
    make_hist(
        values=total_tokens,
        title="SFT Total Tokens Distribution",
        xlabel="sft_total_tokens",
        output_path=tokens_path,
        bins=80,
    )

    print("\nSaved:")
    print(lines_path)
    print(tokens_path)


if __name__ == "__main__":
    main()
