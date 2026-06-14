#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MIN_WORDS_PER_SHATER = 3
MAX_WORDS_PER_SHATER = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Fanar prefix-continuation generations for meter/count scoring.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="fanar_2_diwan_prefix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    normalized = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            if str(row.get("model_name") or "") == str(args.model_name):
                row = normalize_row(row)
                normalized += 1
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"rows={rows} normalized={normalized} output={output_path}")
    return 0


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    original = str(out.get("raw_generated_text") or out.get("generated_text") or "").strip()
    if not original:
        return out
    normalized = normalize_fanar_text(original)
    out["raw_generated_text"] = original
    out["generated_text"] = normalized
    out["fanar_prefix_generated_text_normalized"] = normalized
    out["fanar_prefix_normalization_applied"] = True
    out["fanar_prefix_normalization_method"] = "merge_short_lines_to_next_and_split_long_lines"
    out["fanar_prefix_raw_num_lines"] = len(nonempty_lines(original))
    out["fanar_prefix_normalized_num_lines"] = len(nonempty_lines(normalized))
    out["fanar_prefix_raw_short_line_count"] = count_short_lines(original)
    out["fanar_prefix_normalized_short_line_count"] = count_short_lines(normalized)
    return out


def normalize_fanar_text(text: str) -> str:
    lines = [clean_line(line) for line in str(text or "").replace("\r", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    lines = merge_short_to_next(lines)
    lines = split_all_balanced(lines)
    lines = merge_short_to_next(lines)
    lines = split_all_balanced(lines)
    return "\n".join(line for line in lines if line).strip()


def clean_line(line: str) -> str:
    return SPACE_RE.sub(" ", str(line or "").strip())


def nonempty_lines(text: str) -> list[str]:
    return [clean_line(line) for line in str(text or "").replace("\r", "\n").split("\n") if clean_line(line)]


def word_count(line: str) -> int:
    return len(str(line or "").split())


def count_short_lines(text: str) -> int:
    return sum(1 for line in nonempty_lines(text) if word_count(line) < MIN_WORDS_PER_SHATER)


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


def split_balanced(line: str) -> list[str]:
    words = [word for word in str(line or "").split() if word]
    n_words = len(words)
    if n_words <= MAX_WORDS_PER_SHATER:
        return [" ".join(words)] if words else []

    chunk_count = max(1, math.ceil(n_words / MAX_WORDS_PER_SHATER))
    while chunk_count > 1 and math.floor(n_words / chunk_count) < MIN_WORDS_PER_SHATER:
        chunk_count -= 1

    chunks: list[str] = []
    for index in range(chunk_count):
        start = math.floor(index * n_words / chunk_count)
        end = math.floor((index + 1) * n_words / chunk_count)
        chunk_words = words[start:end]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


if __name__ == "__main__":
    raise SystemExit(main())
