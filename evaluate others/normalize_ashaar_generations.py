#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
MULTISPACE_RE = re.compile(r"\s{2,}")
SINGLE_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
MAX_WORDS_PER_NORMALIZED_SHATER = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Ashaar character-spaced generations for scoring.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--model-name", default="ashaar_model")
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
    out["raw_generated_text"] = original
    normalized = normalize_ashaar_text(original, requested_lines=requested_lines_from_row(out))
    out["generated_text"] = normalized
    out["ashaar_generated_text_normalized"] = normalized
    out["ashaar_normalization_applied"] = True
    out["ashaar_normalization_method"] = "collapse_char_spaced_arabic_preserve_separators_or_length_guard"
    out["ashaar_raw_newline_count"] = int(original.count("\n"))
    out["ashaar_normalized_newline_count"] = int(normalized.count("\n"))
    return out


def requested_lines_from_row(row: dict[str, Any]) -> int:
    for key in ("requested_num_lines", "continuation_requested_num_lines", "sft_num_lines", "requested_lines"):
        value = row.get(key)
        if value is not None and str(value).strip():
            try:
                return max(1, int(value))
            except Exception:
                pass
    bayts = row.get("requested_bayts")
    if bayts is not None and str(bayts).strip():
        try:
            return max(1, int(bayts) * 2)
        except Exception:
            pass
    return 0


def normalize_ashaar_text(text: str, requested_lines: int = 0) -> str:
    text = str(text or "").replace("\r", "\n").strip()
    if not text:
        return ""
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(raw_lines) > 1:
        collapsed_lines = [collapse_char_spaced_line(line) for line in raw_lines]
        return "\n".join(line for line in collapsed_lines if line).strip()

    collapsed = collapse_char_spaced_line(text)
    words = [word for word in collapsed.split() if word]
    if not words:
        return collapsed
    if requested_lines > 1 and len(words) <= requested_lines * MAX_WORDS_PER_NORMALIZED_SHATER:
        return split_words_evenly(collapsed, requested_lines)
    return split_words_by_max_words(collapsed, MAX_WORDS_PER_NORMALIZED_SHATER)


def collapse_char_spaced_line(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    # Ashaar commonly emits single spaces between characters and 2+ spaces between words.
    chunks = [chunk for chunk in MULTISPACE_RE.split(text) if chunk.strip()]
    if len(chunks) > 1:
        words = [collapse_word_chunk(chunk) for chunk in chunks]
    else:
        words = [collapse_word_chunk(chunk) for chunk in SINGLE_SPACE_RE.split(text) if chunk.strip()]
    words = [word for word in words if word]
    return " ".join(words).strip()


def collapse_word_chunk(chunk: str) -> str:
    chunk = str(chunk or "")
    chunk = SINGLE_SPACE_RE.sub("", chunk)
    chunk = DIACRITICS_RE.sub("", chunk)
    chunk = chunk.replace("ـ", "")
    return chunk.strip()


def split_words_by_max_words(text: str, max_words_per_line: int) -> str:
    words = [word for word in text.split() if word]
    if not words:
        return text.strip()
    max_words = max(1, int(max_words_per_line))
    lines = [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]
    return "\n".join(line for line in lines if line).strip()


def split_words_evenly(text: str, requested_lines: int) -> str:
    words = [word for word in text.split() if word]
    if not words or requested_lines <= 1:
        return text.strip()
    line_count = min(requested_lines, len(words))
    lines: list[str] = []
    for index in range(line_count):
        start = math.floor(index * len(words) / line_count)
        end = math.floor((index + 1) * len(words) / line_count)
        line_words = words[start:end]
        if line_words:
            lines.append(" ".join(line_words))
    return "\n".join(lines).strip()


if __name__ == "__main__":
    raise SystemExit(main())
