#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from metrics import score_generation_row


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append meter/count metrics to generated rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as out:
        for row in read_jsonl(Path(args.input_jsonl)):
            scored = dict(row)
            scored.update(score_generation_row(row))
            out.write(json.dumps(scored, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= int(args.limit):
                break
    print(f"scored_rows={count} output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
