#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DASH_SEPARATOR_RE = re.compile(r"\s+[ـ\-–—]+\s+")
MULTISPACE_RE = re.compile(r"[ \t\r\f\v]+")
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!؟?۔])\s+")
METHOD = "split_dash_sentence_merge_short_split_long_truncate_to_requested_lines"
MIN_WORDS_PER_SHATER = 3
MAX_WORDS_PER_SHATER = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GPT-2 continuation generations for meter/count scoring.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="gpt2_small_arabic_poetry")
    parser.add_argument("--min-candidates-before-sentence-split", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    processed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            if str(row.get("model_name") or "") == str(args.model_name):
                row = preprocess_row(
                    row,
                    min_candidates_before_sentence_split=int(args.min_candidates_before_sentence_split),
                )
                processed += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"rows={rows} processed={processed} output={output_path}")
    return 0


def preprocess_row(row: dict[str, Any], min_candidates_before_sentence_split: int = 2) -> dict[str, Any]:
    out = dict(row)
    original = str(out.get("raw_generated_text") or out.get("generated_text") or "").strip()
    if not original:
        return out

    requested_lines = requested_lines_from_row(out)
    candidates = extract_candidate_lines(
        original,
        requested_lines=requested_lines,
        min_candidates_before_sentence_split=min_candidates_before_sentence_split,
    )
    selected = candidates[:requested_lines] if requested_lines > 0 else candidates
    processed_text = "\n".join(selected).strip()

    out["raw_generated_text"] = original
    out["generated_text"] = processed_text
    out["gpt2_preprocessing_applied"] = True
    out["gpt2_preprocessing_method"] = METHOD
    out["raw_generated_num_lines"] = len(nonempty_lines(original))
    out["processed_generated_num_lines"] = len(nonempty_lines(processed_text))
    out["processed_candidate_num_lines"] = len(candidates)
    out["processed_truncated_extra_lines"] = max(0, len(candidates) - len(selected))
    out["processed_short_line_count"] = count_short_lines(processed_text)
    out["processed_max_words_per_line"] = max_words_per_line(processed_text)
    return out


def requested_lines_from_row(row: dict[str, Any]) -> int:
    for key in ("requested_num_lines", "continuation_requested_num_lines", "sft_num_lines", "requested_lines"):
        value = row.get(key)
        if value is not None and str(value).strip():
            try:
                return max(0, int(value))
            except Exception:
                pass
    bayts = row.get("requested_bayts")
    if bayts is not None and str(bayts).strip():
        try:
            return max(0, int(bayts) * 2)
        except Exception:
            pass
    return 0


def extract_candidate_lines(
    text: str,
    requested_lines: int = 0,
    min_candidates_before_sentence_split: int = 2,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    candidates: list[str] = []
    for raw_line in nonempty_lines(text):
        dash_parts = [part.strip() for part in DASH_SEPARATOR_RE.split(raw_line) if part.strip()]
        if len(dash_parts) >= min_candidates_before_sentence_split:
            candidates.extend(dash_parts)
        else:
            sentence_parts = split_sentence_artifacts(raw_line)
            candidates.extend(sentence_parts)

    candidates = [line for line in (clean_line(part) for part in candidates) if is_useful_line(line)]
    candidates = merge_short_to_next(candidates)
    candidates = split_all_balanced(candidates)
    candidates = merge_short_to_next(candidates)
    candidates = split_all_balanced(candidates)
    return [line for line in candidates if line]


def split_sentence_artifacts(text: str) -> list[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    return parts or [text.strip()]


def split_words_evenly(text: str, requested_lines: int) -> list[str]:
    words = [word.strip() for word in normalize_text(text).split(" ") if word.strip()]
    if requested_lines <= 1 or len(words) <= 1:
        return [normalize_text(text)]
    line_count = min(requested_lines, len(words))
    lines: list[str] = []
    for index in range(line_count):
        start = index * len(words) // line_count
        end = (index + 1) * len(words) // line_count
        line = " ".join(words[start:end]).strip()
        if line:
            lines.append(line)
    return lines


def merge_short_to_next(lines: list[str]) -> list[str]:
    out: list[str] = []
    pending: list[str] = []
    for line in lines:
        if word_count(line) < MIN_WORDS_PER_SHATER:
            pending.append(line)
            continue
        if pending:
            line = " ".join(pending + [line])
            pending = []
        out.append(line)
    if pending:
        if out:
            out[-1] = " ".join([out[-1]] + pending)
        else:
            out.append(" ".join(pending))
    return out


def split_all_balanced(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        out.extend(split_balanced(line))
    return out


def split_balanced(text: str) -> list[str]:
    words = [word for word in normalize_text(text).split(" ") if word.strip()]
    n_words = len(words)
    if n_words <= MAX_WORDS_PER_SHATER:
        return [" ".join(words)] if words else []

    chunk_count = max(1, (n_words + MAX_WORDS_PER_SHATER - 1) // MAX_WORDS_PER_SHATER)
    while chunk_count > 1 and (n_words // chunk_count) < MIN_WORDS_PER_SHATER:
        chunk_count -= 1

    lines: list[str] = []
    for index in range(chunk_count):
        start = index * n_words // chunk_count
        end = (index + 1) * n_words // chunk_count
        line = " ".join(words[start:end]).strip()
        if line:
            lines.append(line)
    return lines


def word_count(text: str) -> int:
    return len(str(text or "").split())


def count_short_lines(text: str) -> int:
    return sum(1 for line in nonempty_lines(text) if word_count(line) < MIN_WORDS_PER_SHATER)


def max_words_per_line(text: str) -> int:
    counts = [word_count(line) for line in nonempty_lines(text)]
    return max(counts) if counts else 0


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def clean_line(text: str) -> str:
    text = normalize_text(text)
    text = text.strip(" \t\n\r-–—ـ،,.;؛:")
    return text.strip()


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").replace("\r", "\n").split("\n") if line.strip()]


def is_useful_line(text: str) -> bool:
    return bool(text) and bool(ARABIC_RE.search(text))


if __name__ == "__main__":
    raise SystemExit(main())
