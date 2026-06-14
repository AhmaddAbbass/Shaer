#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from common import OUTPUTS_ROOT, atomic_write_json, build_meter_label, json_ready, parse_poem_lines, utc_now_iso


DEFAULT_SOURCE_DATASET = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASHAAR_ROOT = PROJECT_ROOT.parent / "Ashaar"
DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Ashaar-native generation manifest from the Shaer test split.")
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=("controls", "native_prefix"), default="controls")
    parser.add_argument("--prefix-mode", choices=("first_shatr", "first_bayt", "first_two_bayts"), default="first_bayt")
    parser.add_argument("--ashaar-root", default=str(DEFAULT_ASHAAR_ROOT))
    parser.add_argument("--theme-fallback", default="null")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ashaar_root = Path(args.ashaar_root)
    meter_to_token = read_json(ashaar_root / "extra" / "meter_tokens.json")
    theme_to_token = read_json(ashaar_root / "extra" / "theme_tokens.json")
    rows = load_source_rows(args.source_dataset, args.split, limit_rows=int(args.limit_rows or 0))

    manifest_rows: list[dict[str, Any]] = []
    missing_meter = Counter()
    missing_theme = Counter()
    qafiyah_counts = Counter()
    for source_row_index, source_row in enumerate(rows):
        row = build_manifest_row(
            source_row=dict(source_row),
            source_row_index=source_row_index,
            meter_to_token=meter_to_token,
            theme_to_token=theme_to_token,
            prompt_mode=args.prompt_mode,
            prefix_mode=args.prefix_mode,
            theme_fallback=args.theme_fallback,
        )
        if row["ashaar_meter_token_missing"]:
            missing_meter[str(row.get("base_meter") or "")] += 1
        if row["ashaar_theme_token_missing"]:
            missing_theme[str(row.get("poem_theme") or "")] += 1
        qafiyah_counts[str(row.get("ashaar_qafiyah") or "")] += 1
        manifest_rows.append(row)

    output_jsonl = resolve_output_path(args.output_jsonl, args.prompt_mode)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")

    summary = {
        "created_at_utc": utc_now_iso(),
        "source_dataset": args.source_dataset,
        "split": args.split,
        "output_jsonl": str(output_jsonl),
        "rows": len(manifest_rows),
        "prompt_mode": args.prompt_mode,
        "prefix_mode": args.prefix_mode if args.prompt_mode == "native_prefix" else "",
        "missing_meter_token_rows": sum(missing_meter.values()),
        "missing_meter_token_values": dict(missing_meter),
        "missing_theme_token_rows": sum(missing_theme.values()),
        "missing_theme_token_values": dict(missing_theme),
        "qafiyah_counts": dict(qafiyah_counts.most_common()),
    }
    atomic_write_json(output_jsonl.with_suffix(".summary.json"), summary)
    print(f"rows={len(manifest_rows)} output={output_jsonl}")
    return 0


def read_json(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_source_rows(dataset_id: str, split: str, limit_rows: int = 0) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split=split)
        if limit_rows:
            ds = ds.select(range(min(int(limit_rows), len(ds))))
        return [dict(row) for row in ds]
    except Exception:
        return load_rows_from_dataset_server(dataset_id, split, limit_rows=limit_rows)


def load_rows_from_dataset_server(dataset_id: str, split: str, page_size: int = 100, limit_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    quoted_dataset = urllib.parse.quote(dataset_id, safe="")
    quoted_split = urllib.parse.quote(split, safe="")
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={quoted_dataset}&config=default&split={quoted_split}&offset={offset}&length={page_size}"
        )
        payload = fetch_json_with_retry(url)
        page_rows = payload.get("rows") or []
        if not page_rows:
            break
        rows.extend(dict(item["row"]) for item in page_rows)
        if limit_rows and len(rows) >= int(limit_rows):
            return rows[: int(limit_rows)]
        total = int(payload.get("num_rows_total") or 0)
        offset += len(page_rows)
        if offset >= total:
            break
    return rows


def fetch_json_with_retry(url: str, attempts: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    if last_error:
        raise last_error
    raise RuntimeError("fetch_json_with_retry failed without an exception")


def build_manifest_row(
    source_row: dict[str, Any],
    source_row_index: int,
    meter_to_token: dict[str, str],
    theme_to_token: dict[str, str],
    prompt_mode: str,
    prefix_mode: str,
    theme_fallback: str,
) -> dict[str, Any]:
    verses = normalize_verses(source_row.get("poem verses") or source_row.get("poem_verses") or [])
    reference_completion = "\n".join(verses).strip() or str(source_row.get("sft_completion") or "").strip()
    if not verses:
        verses = parse_poem_lines(reference_completion)

    base_meter = str(source_row.get("base_meter") or "").strip()
    form = str(source_row.get("form") or "").strip()
    poem_theme = str(source_row.get("poem theme") or source_row.get("poem_theme") or "").strip()
    theme_for_token = poem_theme if poem_theme in theme_to_token else str(theme_fallback or "null")
    meter_token = meter_to_token.get(base_meter, "")
    theme_token = theme_to_token.get(theme_for_token, "")
    qafiyah = infer_qafiyah(verses)
    controls_prompt = " ".join(part for part in (meter_token, qafiyah, theme_token) if part).strip()

    prefix_text = ""
    native_prefix_text = ""
    prompt = controls_prompt
    if prompt_mode == "native_prefix":
        prefix_lines_count = {"first_shatr": 1, "first_bayt": 2, "first_two_bayts": 4}[prefix_mode]
        prefix_lines = verses[: min(prefix_lines_count, len(verses))]
        prefix_text = "\n".join(prefix_lines).strip()
        native_prefix_text = format_native_prefix(prefix_lines)
        prompt = f"{controls_prompt} <|psep|> {native_prefix_text}".strip()

    requested_bayts = int(source_row.get("requested_bayts") or max(0, len(verses) // 2))
    requested_num_lines = int(source_row.get("sft_num_lines") or requested_bayts * 2 or len(verses))
    generation_id_base = f"ashaar_native_test_{source_row_index:05d}"
    return {
        "manifest_row_id": generation_id_base,
        "input_row_id": generation_id_base,
        "source_row_index": int(source_row_index),
        "source_id": str(source_row.get("id") or ""),
        "base_meter": base_meter,
        "form": form,
        "meter_label": build_meter_label(base_meter, form),
        "requested_bayts": requested_bayts,
        "requested_num_lines": requested_num_lines,
        "sft_num_lines": requested_num_lines,
        "poem_theme": poem_theme,
        "poem_meter": str(source_row.get("poem meter") or "").strip(),
        "poem_url": str(source_row.get("poem url") or "").strip(),
        "description": str(source_row.get("description") or ""),
        "enhanced_description": str(source_row.get("enhanced_description") or ""),
        "reference_completion": reference_completion,
        "reference_prefix_text": prefix_text,
        "ashaar_prompt_mode": prompt_mode,
        "ashaar_prefix_mode": prefix_mode if prompt_mode == "native_prefix" else "",
        "ashaar_meter_token": meter_token,
        "ashaar_meter_token_missing": not bool(meter_token),
        "ashaar_qafiyah": qafiyah,
        "ashaar_theme": theme_for_token,
        "ashaar_theme_token": theme_token,
        "ashaar_theme_token_missing": not bool(theme_token) or theme_for_token != poem_theme,
        "ashaar_controls_prompt": controls_prompt,
        "ashaar_native_prefix_text": native_prefix_text,
        "ashaar_native_prompt": prompt,
        "manifest_created_at_utc": utc_now_iso(),
    }


def normalize_verses(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return parse_poem_lines(str(value or ""))


def infer_qafiyah(verses: list[str]) -> str:
    try:
        from bohour.qafiah import get_qafiyah

        bayts = [" # ".join(verses[index : index + 2]) for index in range(0, len(verses) - 1, 2)]
        values = [str(get_qafiyah([bayt])[0][0]).strip() for bayt in bayts if bayt.strip()]
        values = [value for value in values if value]
        if values:
            return Counter(values).most_common(1)[0][0]
    except Exception:
        pass

    endings = []
    for index in range(1, len(verses), 2):
        letter = last_arabic_letter(verses[index])
        if letter:
            endings.append(letter)
    if not endings:
        for verse in verses:
            letter = last_arabic_letter(verse)
            if letter:
                endings.append(letter)
    return Counter(endings).most_common(1)[0][0] if endings else ""


def last_arabic_letter(text: str) -> str:
    text = DIACRITICS_RE.sub("", str(text or "").replace("ـ", ""))
    letters = ARABIC_LETTER_RE.findall(text)
    if not letters:
        return ""
    letter = letters[-1]
    return {"ى": "ي", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا"}.get(letter, letter)


def format_native_prefix(lines: list[str]) -> str:
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        return ""
    parts: list[str] = []
    index = 0
    while index < len(lines):
        first = lines[index]
        second = lines[index + 1] if index + 1 < len(lines) else ""
        if second:
            parts.append(f"<|bsep|> {first} <|vsep|> {second} </|bsep|>")
        else:
            parts.append(f"<|bsep|> {first}")
        index += 2
    return " ".join(parts).strip()


def resolve_output_path(output_jsonl: str, prompt_mode: str) -> Path:
    if output_jsonl:
        return Path(output_jsonl)
    timestamp = utc_now_iso().replace(":", "").replace("-", "")
    return OUTPUTS_ROOT / f"ashaar_native_manifest_{prompt_mode}_{timestamp}.jsonl"


if __name__ == "__main__":
    raise SystemExit(main())
