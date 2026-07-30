#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
NON_WORD_RE = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


@dataclass
class SourceDoc:
    idx: int
    split: str
    row_id: str
    source_index: str
    base_meter: str
    form: str
    text: str
    norm_text: str
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Shaer generations for copying from train and held-out source poems.")
    parser.add_argument("--split-data-dir", default="", help="Directory containing train/eval/test parquet files from the SFT split dataset.")
    parser.add_argument("--shaer-eval-parquet", default="", help="Parquet file for Shaer-AI/shaer-sft-test.")
    parser.add_argument("--output-dir", default="evaluation/outputs/poem_copying_analysis")
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-posting-df", type=int, default=500)
    parser.add_argument("--limit-rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_data_dir = Path(args.split_data_dir) if args.split_data_dir else default_split_data_dir()
    shaer_eval_parquet = Path(args.shaer_eval_parquet) if args.shaer_eval_parquet else default_shaer_eval_parquet()

    train_docs = load_source_docs(split_data_dir, "train")
    heldout_test_docs = load_source_docs(split_data_dir, "test")
    heldout_eval_docs = load_source_docs(split_data_dir, "eval")
    shaer_df = pd.read_parquet(shaer_eval_parquet)
    if args.limit_rows:
        shaer_df = shaer_df.head(int(args.limit_rows)).copy()

    print(f"train_docs={len(train_docs)} heldout_test={len(heldout_test_docs)} heldout_eval={len(heldout_eval_docs)} shaer_rows={len(shaer_df)}", flush=True)
    index = build_ngram_index(train_docs, n=5, max_posting_df=int(args.max_posting_df))
    train_exact = build_exact_index(train_docs)
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for row_num, (_, row) in enumerate(shaer_df.iterrows(), start=1):
        if row_num % 250 == 0:
            print(f"processed={row_num}/{len(shaer_df)}", flush=True)
        result, example = analyze_generation(
            row=row.to_dict(),
            train_docs=train_docs,
            train_index=index,
            train_exact=train_exact,
            paired_test_doc=source_doc_from_eval_row(row.to_dict()),
            max_candidates=int(args.max_candidates),
        )
        rows.append(result)
        if example:
            examples.append(example)

    row_df = pd.DataFrame(rows)
    row_path = output_dir / "row_level_copying_metrics.csv"
    row_df.to_csv(row_path, index=False, encoding="utf-8-sig")

    examples = sorted(examples, key=lambda item: item["sort_score"], reverse=True)[:50]
    examples_path = output_dir / "top_copying_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as handle:
        for item in examples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = build_summary(row_df, train_docs, heldout_test_docs, heldout_eval_docs, args)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    make_plots(row_df, output_dir)

    print(f"wrote={row_path}", flush=True)
    print(f"wrote={summary_path}", flush=True)
    return 0


def default_split_data_dir() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Shaer-AI--ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits" / "snapshots"
    snapshots = sorted([p for p in base.glob("*") if (p / "data").exists()], key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        raise FileNotFoundError(f"No cached split dataset snapshots found under {base}")
    return snapshots[0] / "data"


def default_shaer_eval_parquet() -> Path:
    base = Path.home() / ".cache" / "huggingface" / "hub" / "datasets--Shaer-AI--shaer-sft-test" / "snapshots"
    files = sorted(base.glob("*/data/test-00000-of-00001.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No cached Shaer eval parquet found under {base}")
    return files[0]


def load_source_docs(data_dir: Path, split: str) -> list[SourceDoc]:
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files for split={split} in {data_dir}")
    docs: list[SourceDoc] = []
    for path in paths:
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            text = str(row.get("sft_completion") or "").strip()
            norm = normalize_arabic(text)
            docs.append(
                SourceDoc(
                    idx=len(docs),
                    split=split,
                    row_id=str(row.get("id") or ""),
                    source_index=str(row.get("source_index") or ""),
                    base_meter=str(row.get("base_meter") or ""),
                    form=str(row.get("form") or ""),
                    text=text,
                    norm_text=norm,
                    tokens=tokenize_norm(norm),
                )
            )
    return docs


def source_doc_from_eval_row(row: dict[str, Any]) -> SourceDoc | None:
    text = str(row.get("reference_completion") or "").strip()
    if not text:
        return None
    norm = normalize_arabic(text)
    return SourceDoc(
        idx=-1,
        split="paired_reference",
        row_id=str(row.get("id") or ""),
        source_index=str(row.get("source_id") or ""),
        base_meter=str(row.get("base_meter") or ""),
        form=str(row.get("form") or ""),
        text=text,
        norm_text=norm,
        tokens=tokenize_norm(norm),
    )


def normalize_arabic(text: str) -> str:
    text = str(text or "")
    text = ARABIC_DIACRITICS_RE.sub("", text)
    text = text.replace("\u0640", "")
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def tokenize_norm(norm_text: str) -> list[str]:
    return [token for token in str(norm_text or "").split() if token]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def char_ngrams(norm_text: str, n: int = 5) -> set[str]:
    compact = re.sub(r"\s+", "", norm_text)
    if len(compact) < n:
        return set()
    return {compact[i : i + n] for i in range(0, len(compact) - n + 1)}


def build_ngram_index(docs: list[SourceDoc], *, n: int, max_posting_df: int) -> dict[tuple[str, ...], list[int]]:
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for doc in docs:
        for gram in ngrams(doc.tokens, n):
            postings[gram].append(doc.idx)
    filtered = {gram: ids for gram, ids in postings.items() if len(ids) <= max_posting_df}
    print(f"word{n}_index_grams={len(filtered)} dropped_common_grams={len(postings) - len(filtered)}", flush=True)
    return filtered


def build_exact_index(docs: list[SourceDoc]) -> dict[str, list[int]]:
    exact: dict[str, list[int]] = defaultdict(list)
    for doc in docs:
        if doc.norm_text:
            exact[doc.norm_text].append(doc.idx)
    return exact


def analyze_generation(
    *,
    row: dict[str, Any],
    train_docs: list[SourceDoc],
    train_index: dict[tuple[str, ...], list[int]],
    train_exact: dict[str, list[int]],
    paired_test_doc: SourceDoc | None,
    max_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    generation_id = str(row.get("id") or row.get("generation_id") or "")
    generated_text = str(row.get("generated_text") or "").strip()
    norm = normalize_arabic(generated_text)
    tokens = tokenize_norm(norm)
    gen5 = ngrams(tokens, 5)
    gen8 = ngrams(tokens, 8)
    gen13 = ngrams(tokens, 13)
    gen_char5 = char_ngrams(norm, 5)

    candidate_counts: Counter[int] = Counter()
    for gram in gen5:
        candidate_counts.update(train_index.get(gram, []))
    for idx in train_exact.get(norm, []):
        candidate_counts[idx] += 10_000

    candidate_ids = [idx for idx, _ in candidate_counts.most_common(max_candidates)]
    best_train = empty_match()
    best_doc: SourceDoc | None = None
    for idx in candidate_ids:
        match = compare_to_doc(tokens, norm, gen5, gen8, gen13, gen_char5, train_docs[idx])
        if match["sort_score"] > best_train["sort_score"]:
            best_train = match
            best_doc = train_docs[idx]

    paired_match = empty_match()
    if paired_test_doc is not None:
        paired_match = compare_to_doc(tokens, norm, gen5, gen8, gen13, gen_char5, paired_test_doc)

    reasons: list[str] = []
    if best_train["normalized_exact"]:
        reasons.append("train_normalized_exact")
    if best_train["max_copied_word_span"] >= 30:
        reasons.append("train_span_ge_30")
    if best_train["shared_word13_count"] > 0:
        reasons.append("train_shared_13gram")
    if best_train["word5_jaccard"] >= 0.50:
        reasons.append("train_word5_jaccard_ge_0.50")
    if best_train["char5_jaccard"] >= 0.70:
        reasons.append("train_char5_jaccard_ge_0.70")

    paired_reasons: list[str] = []
    if paired_match["normalized_exact"]:
        paired_reasons.append("reference_normalized_exact")
    if paired_match["max_copied_word_span"] >= 30:
        paired_reasons.append("reference_span_ge_30")
    if paired_match["shared_word13_count"] > 0:
        paired_reasons.append("reference_shared_13gram")
    if paired_match["word5_jaccard"] >= 0.50:
        paired_reasons.append("reference_word5_jaccard_ge_0.50")
    if paired_match["char5_jaccard"] >= 0.70:
        paired_reasons.append("reference_char5_jaccard_ge_0.70")

    output = {
        "generation_id": generation_id,
        "base_meter": str(row.get("base_meter") or ""),
        "form": str(row.get("form") or ""),
        "generated_word_count": len(tokens),
        "candidate_train_docs": len(candidate_ids),
        "nearest_train_id": best_doc.row_id if best_doc else "",
        "nearest_train_source_index": best_doc.source_index if best_doc else "",
        "nearest_train_base_meter": best_doc.base_meter if best_doc else "",
        "nearest_train_form": best_doc.form if best_doc else "",
        "train_normalized_exact": bool(best_train["normalized_exact"]),
        "train_word5_jaccard": best_train["word5_jaccard"],
        "train_char5_jaccard": best_train["char5_jaccard"],
        "train_shared_word8_count": best_train["shared_word8_count"],
        "train_shared_word13_count": best_train["shared_word13_count"],
        "train_max_copied_word_span": best_train["max_copied_word_span"],
        "reference_normalized_exact": bool(paired_match["normalized_exact"]),
        "reference_word5_jaccard": paired_match["word5_jaccard"],
        "reference_char5_jaccard": paired_match["char5_jaccard"],
        "reference_shared_word8_count": paired_match["shared_word8_count"],
        "reference_shared_word13_count": paired_match["shared_word13_count"],
        "reference_max_copied_word_span": paired_match["max_copied_word_span"],
        "train_copy_flag": bool(reasons),
        "train_copy_flag_reasons": ";".join(reasons),
        "reference_copy_flag": bool(paired_reasons),
        "reference_copy_flag_reasons": ";".join(paired_reasons),
    }

    example = None
    if best_doc and (reasons or best_train["sort_score"] > 0.20):
        example = {
            "sort_score": best_train["sort_score"],
            "generation_id": generation_id,
            "train_copy_flag_reasons": output["train_copy_flag_reasons"],
            "train_metrics": best_train,
            "generated_text": generated_text,
            "nearest_train_id": best_doc.row_id,
            "nearest_train_source_index": best_doc.source_index,
            "nearest_train_text": best_doc.text,
        }
    return output, example


def empty_match() -> dict[str, Any]:
    return {
        "normalized_exact": False,
        "word5_jaccard": 0.0,
        "char5_jaccard": 0.0,
        "shared_word8_count": 0,
        "shared_word13_count": 0,
        "max_copied_word_span": 0,
        "sort_score": 0.0,
    }


def compare_to_doc(
    gen_tokens: list[str],
    gen_norm: str,
    gen5: set[tuple[str, ...]],
    gen8: set[tuple[str, ...]],
    gen13: set[tuple[str, ...]],
    gen_char5: set[str],
    doc: SourceDoc,
) -> dict[str, Any]:
    doc5 = ngrams(doc.tokens, 5)
    doc8 = ngrams(doc.tokens, 8)
    doc13 = ngrams(doc.tokens, 13)
    doc_char5 = char_ngrams(doc.norm_text, 5)
    word5_j = jaccard(gen5, doc5)
    char5_j = jaccard(gen_char5, doc_char5)
    shared8 = len(gen8 & doc8)
    shared13 = len(gen13 & doc13)
    span = longest_common_word_span(gen_tokens, doc.tokens)
    normalized_exact = bool(gen_norm and gen_norm == doc.norm_text)
    sort_score = max(word5_j, char5_j, min(1.0, span / 30.0), 1.0 if normalized_exact else 0.0, min(1.0, shared13 / 3.0))
    return {
        "normalized_exact": normalized_exact,
        "word5_jaccard": round(word5_j, 6),
        "char5_jaccard": round(char5_j, 6),
        "shared_word8_count": int(shared8),
        "shared_word13_count": int(shared13),
        "max_copied_word_span": int(span),
        "sort_score": round(sort_score, 6),
    }


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def longest_common_word_span(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


def build_summary(row_df: pd.DataFrame, train_docs: list[SourceDoc], heldout_test_docs: list[SourceDoc], heldout_eval_docs: list[SourceDoc], args: argparse.Namespace) -> dict[str, Any]:
    total = len(row_df)

    def count_bool(col: str) -> int:
        return int(row_df[col].fillna(False).astype(bool).sum())

    def pct(value: int) -> float:
        return round((value / total * 100.0) if total else 0.0, 4)

    train_flags = count_bool("train_copy_flag")
    ref_flags = count_bool("reference_copy_flag")
    return {
        "total_generations": total,
        "source_counts": {
            "train_rows": len(train_docs),
            "heldout_test_rows": len(heldout_test_docs),
            "heldout_eval_rows": len(heldout_eval_docs),
        },
        "parameters": {
            "max_candidates": int(args.max_candidates),
            "max_posting_df": int(args.max_posting_df),
        },
        "train": {
            "normalized_exact": count_bool("train_normalized_exact"),
            "any_shared_word13": int((row_df["train_shared_word13_count"] > 0).sum()),
            "span_ge_30_words": int((row_df["train_max_copied_word_span"] >= 30).sum()),
            "span_ge_20_words": int((row_df["train_max_copied_word_span"] >= 20).sum()),
            "span_ge_10_words": int((row_df["train_max_copied_word_span"] >= 10).sum()),
            "word5_jaccard_ge_050": int((row_df["train_word5_jaccard"] >= 0.50).sum()),
            "char5_jaccard_ge_070": int((row_df["train_char5_jaccard"] >= 0.70).sum()),
            "flagged_rows": train_flags,
            "flagged_pct": pct(train_flags),
            "mean_word5_jaccard": round(float(row_df["train_word5_jaccard"].mean()), 6),
            "median_word5_jaccard": round(float(row_df["train_word5_jaccard"].median()), 6),
            "max_word5_jaccard": round(float(row_df["train_word5_jaccard"].max()), 6),
            "mean_char5_jaccard": round(float(row_df["train_char5_jaccard"].mean()), 6),
            "median_char5_jaccard": round(float(row_df["train_char5_jaccard"].median()), 6),
            "max_char5_jaccard": round(float(row_df["train_char5_jaccard"].max()), 6),
            "mean_max_copied_word_span": round(float(row_df["train_max_copied_word_span"].mean()), 6),
            "median_max_copied_word_span": round(float(row_df["train_max_copied_word_span"].median()), 6),
            "max_copied_word_span": int(row_df["train_max_copied_word_span"].max()),
        },
        "heldout_reference": {
            "normalized_exact": count_bool("reference_normalized_exact"),
            "any_shared_word13": int((row_df["reference_shared_word13_count"] > 0).sum()),
            "span_ge_30_words": int((row_df["reference_max_copied_word_span"] >= 30).sum()),
            "span_ge_20_words": int((row_df["reference_max_copied_word_span"] >= 20).sum()),
            "span_ge_10_words": int((row_df["reference_max_copied_word_span"] >= 10).sum()),
            "word5_jaccard_ge_050": int((row_df["reference_word5_jaccard"] >= 0.50).sum()),
            "char5_jaccard_ge_070": int((row_df["reference_char5_jaccard"] >= 0.70).sum()),
            "flagged_rows": ref_flags,
            "flagged_pct": pct(ref_flags),
            "mean_word5_jaccard": round(float(row_df["reference_word5_jaccard"].mean()), 6),
            "median_word5_jaccard": round(float(row_df["reference_word5_jaccard"].median()), 6),
            "max_word5_jaccard": round(float(row_df["reference_word5_jaccard"].max()), 6),
            "mean_char5_jaccard": round(float(row_df["reference_char5_jaccard"].mean()), 6),
            "median_char5_jaccard": round(float(row_df["reference_char5_jaccard"].median()), 6),
            "max_char5_jaccard": round(float(row_df["reference_char5_jaccard"].max()), 6),
            "mean_max_copied_word_span": round(float(row_df["reference_max_copied_word_span"].mean()), 6),
            "median_max_copied_word_span": round(float(row_df["reference_max_copied_word_span"].median()), 6),
            "max_copied_word_span": int(row_df["reference_max_copied_word_span"].max()),
        },
    }


def make_plots(row_df: pd.DataFrame, output_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(row_df["train_word5_jaccard"], bins=40, color="#255c99", alpha=0.85)
    axes[0].set_title("Nearest training poem")
    axes[0].set_xlabel("Word 5-gram Jaccard")
    axes[0].set_ylabel("Generations")
    axes[1].hist(row_df["reference_word5_jaccard"], bins=40, color="#8f3f2a", alpha=0.85)
    axes[1].set_title("Paired held-out source poem")
    axes[1].set_xlabel("Word 5-gram Jaccard")
    axes[1].set_ylabel("Generations")
    fig.tight_layout()
    fig.savefig(output_dir / "word5_jaccard_hist.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].hist(row_df["train_max_copied_word_span"], bins=range(0, int(row_df["train_max_copied_word_span"].max()) + 2), color="#255c99", alpha=0.85)
    axes[0].set_title("Nearest training poem")
    axes[0].set_xlabel("Longest shared word span")
    axes[0].set_ylabel("Generations")
    axes[1].hist(row_df["reference_max_copied_word_span"], bins=range(0, int(row_df["reference_max_copied_word_span"].max()) + 2), color="#8f3f2a", alpha=0.85)
    axes[1].set_title("Paired held-out source poem")
    axes[1].set_xlabel("Longest shared word span")
    axes[1].set_ylabel("Generations")
    fig.tight_layout()
    fig.savefig(output_dir / "longest_span_hist.png", dpi=180)
    plt.close(fig)

    labels = ["Train", "Held-out source"]
    exact = [int(row_df["train_normalized_exact"].sum()), int(row_df["reference_normalized_exact"].sum())]
    shared13 = [int((row_df["train_shared_word13_count"] > 0).sum()), int((row_df["reference_shared_word13_count"] > 0).sum())]
    span30 = [int((row_df["train_max_copied_word_span"] >= 30).sum()), int((row_df["reference_max_copied_word_span"] >= 30).sum())]
    flagged = [int(row_df["train_copy_flag"].sum()), int(row_df["reference_copy_flag"].sum())]
    x = range(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, vals, label, color in [
        (-1.5 * width, exact, "Exact", "#1b4965"),
        (-0.5 * width, shared13, "Any 13-gram", "#2a9d8f"),
        (0.5 * width, span30, "Span >= 30", "#e76f51"),
        (1.5 * width, flagged, "Flagged", "#6d597a"),
    ]:
        ax.bar([i + offset for i in x], vals, width=width, label=label, color=color)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Generations")
    ax.set_title("Copying flags by comparison source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "copying_flags_bar.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
