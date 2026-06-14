#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from metrics import score_generation_row


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append meter/count metrics to baseline generation rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_handle:
        for row in read_jsonl(Path(args.input_jsonl)):
            scored = dict(row)
            scored.update(score_generation_row(row))
            output_handle.write(json.dumps(scored, ensure_ascii=False) + "\n")
            count += 1
            if args.limit and count >= int(args.limit):
                break
    print(f"scored_rows={count} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

