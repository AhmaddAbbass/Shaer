#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import httpx
import pyarrow.ipc as ipc
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError

from prompt_contracts import (
    DESCRIPTION_PROMPT_VERSION,
    DESCRIPTION_SYSTEM_PROMPT,
    build_description_user_prompt,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_CACHE_DIR = ROOT / "cache"
DATASET_CACHE_ROOT = Path("/root/.cache/huggingface/datasets")
DATASET_CACHE_GLOB = (
    "Shaer-AI___ashaar-with-descriptions-baseform-final-trimmed/default/0.0.0/*/"
    "ashaar-with-descriptions-baseform-final-trimmed-train-*.arrow"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_poem_verses(verses: Any) -> list[str]:
    if not isinstance(verses, list):
        return []
    out: list[str] = []
    for item in verses:
        text = str(item).strip()
        if not text:
            return []
        out.append(text)
    return out


def requested_bayts_from_verses(verses: list[str]) -> int | None:
    if not verses or len(verses) % 2 != 0:
        return None
    return len(verses) // 2


def find_dataset_arrow_paths() -> list[Path]:
    paths = sorted(DATASET_CACHE_ROOT.glob(DATASET_CACHE_GLOB))
    if not paths:
        raise FileNotFoundError("could not find cached dataset arrow files for source dataset")
    return paths


def load_filtered_rows(max_bayts: int, drop_meters: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in find_dataset_arrow_paths():
        with ipc.open_stream(path) as reader:
            table = reader.read_all()
        for row in table.to_pylist():
            verses = normalize_poem_verses(row.get("poem verses"))
            requested_bayts = requested_bayts_from_verses(verses)
            if requested_bayts is None or requested_bayts > max_bayts:
                continue
            meter = str(row["base_meter"]).strip()
            if meter in drop_meters:
                continue
            prepared = dict(row)
            prepared["poem verses"] = verses
            prepared["requested_bayts"] = requested_bayts
            rows.append(prepared)
    return rows


def choose_rows(rows: list[dict[str, Any]], limit: int, seed: int, distinct_meters: bool) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    if not distinct_meters:
        return rows[:limit]

    picked: list[dict[str, Any]] = []
    seen_meters: set[str] = set()
    for row in rows:
        meter = str(row["base_meter"]).strip()
        if meter in seen_meters:
            continue
        picked.append(row)
        seen_meters.add(meter)
        if len(picked) >= limit:
            return picked

    if len(picked) < limit:
        for row in rows:
            if row in picked:
                continue
            picked.append(row)
            if len(picked) >= limit:
                break
    return picked[:limit]


def extract_chat_content(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : idx + 1]
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def clean_description(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def validate_description_text(text: str) -> tuple[bool, str | None]:
    if not text:
        return False, "empty_new_description"
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return False, "json_leakage"
    if '"new_description"' in text or "'new_description'" in text:
        return False, "json_key_leakage"
    return True, None


def extract_new_description(content: str) -> tuple[str | None, str | None]:
    parsed = extract_first_json_object(content)
    candidate: str
    if parsed is not None:
        if "new_description" not in parsed:
            return None, "missing_new_description_key"
        candidate = clean_description(parsed.get("new_description", ""))
    else:
        raw = clean_description(content)
        if not raw:
            return None, "empty_content"
        candidate = raw

    valid, reason = validate_description_text(candidate)
    if not valid:
        return None, reason
    return candidate, None


@dataclass
class OpenRouterClient:
    api_key: str
    model: str
    base_url: str
    timeout: float
    max_retries: int
    referrer: str
    app_title: str
    cache_dir: Path

    def headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referrer:
            headers["HTTP-Referer"] = self.referrer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def cache_key(self, payload: dict[str, Any], cache_salt: str = "") -> str:
        raw = json.dumps({"payload": payload, "cache_salt": cache_salt}, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def chat_json(self, system_prompt: str, user_prompt: str, cache_salt: str = "") -> tuple[dict[str, Any], bool]:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False,
            },
            "reasoning": {"effort": "none"},
            "response_format": {"type": "json_object"},
            "max_tokens": 300,
        }
        key = self.cache_key(payload, cache_salt=cache_salt)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), True

        timeout_obj = httpx.Timeout(self.timeout, connect=min(20.0, self.timeout))
        last_error = None
        response_json = None
        t0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout_obj) as client:
                    response = client.post(
                        f"{self.base_url.rstrip('/')}/chat/completions",
                        headers=self.headers(),
                        json=payload,
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                time.sleep(2.0 * (attempt + 1))

        out = {
            "timestamp_utc": utc_now_iso(),
            "model": self.model,
            "cache_key": key,
            "response": response_json,
            "error": last_error,
            "latency_sec": time.time() - t0,
        }
        cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out, False


def replace_description_in_prompt(prompt: str, old_description: str, new_description: str) -> str:
    if old_description in prompt:
        return prompt.replace(old_description, new_description, 1)
    return prompt


def build_client(cache_dir: Path) -> OpenRouterClient:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    return OpenRouterClient(
        api_key=api_key,
        model=os.getenv("JUDGE_MODEL", "qwen/qwen3.5-35b-a3b").strip(),
        base_url=os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        timeout=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120")),
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "3")),
        referrer=os.getenv("OPENROUTER_REFERRER", "").strip(),
        app_title=os.getenv("OPENROUTER_APP_TITLE", "Shaer-Description-Generation").strip(),
        cache_dir=cache_dir,
    )


def generate_description(
    client: OpenRouterClient,
    row: dict[str, Any],
    max_row_attempts: int,
) -> dict[str, Any]:
    last_reason = None
    last_error = None
    total_latency = 0.0
    last_cached = False
    for attempt in range(max_row_attempts):
        response_obj, cached = client.chat_json(
            DESCRIPTION_SYSTEM_PROMPT,
            build_description_user_prompt(row["poem verses"]),
            cache_salt="" if attempt == 0 else f"retry_{attempt}",
        )
        last_cached = cached
        total_latency += float(response_obj.get("latency_sec") or 0.0)
        content = extract_chat_content(response_obj.get("response") or {})
        candidate, reason = extract_new_description(content)
        if candidate:
            return {
                "status": "ok",
                "enhanced_description": candidate,
                "attempt_count": attempt + 1,
                "latency_sec": total_latency,
                "cached": last_cached,
                "validation_error": None,
                "response_error": response_obj.get("error"),
            }
        last_reason = reason
        last_error = response_obj.get("error")
    return {
        "status": "error",
        "enhanced_description": "",
        "attempt_count": max_row_attempts,
        "latency_sec": total_latency,
        "cached": last_cached,
        "validation_error": last_reason,
        "response_error": last_error,
    }


def load_remote_results(
    api: HfApi | None,
    repo_id: str,
    path_in_repo: str,
    local_download_dir: Path,
) -> tuple[list[dict[str, Any]], Path | None]:
    if api is None:
        return [], None
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=api.token,
            local_dir=str(local_download_dir),
            local_dir_use_symlinks=False,
        )
    except (EntryNotFoundError, HfHubHTTPError, FileNotFoundError):
        return [], None
    downloaded_path = Path(downloaded)
    return iter_jsonl(downloaded_path), downloaded_path


def upload_progress_files(
    api: HfApi | None,
    repo_id: str,
    progress_prefix: str,
    worker_id: int,
    results_path: Path,
    summary_path: Path,
) -> None:
    if api is None:
        return
    worker_prefix = f"{progress_prefix}/workers/worker_{worker_id:02d}"
    uploads = [
        (results_path, f"{worker_prefix}/results.jsonl"),
        (summary_path, f"{worker_prefix}/summary.json"),
    ]
    for local_path, remote_path in uploads:
        last_error: Exception | None = None
        for attempt in range(1, 7):
            try:
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                )
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt >= 6:
                    raise
                time.sleep(min(8.0, 0.75 * attempt))
        if last_error is not None:
            raise last_error


def run_sample_mode(args: argparse.Namespace) -> None:
    run_name = args.run_name or f"descgen_{timestamp_slug()}"
    output_dir = ensure_dir(OUTPUT_ROOT / run_name)
    cache_dir = ensure_dir(DEFAULT_CACHE_DIR / run_name)
    log_path = output_dir / "run.log"

    def log(msg: str) -> None:
        line = f"{utc_now_iso()} | {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    dump_json(
        output_dir / "config_snapshot.json",
        {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "style_version": DESCRIPTION_PROMPT_VERSION,
            "num_rows": args.num_rows,
            "seed": args.seed,
            "max_bayts": args.max_bayts,
            "distinct_meters": args.distinct_meters,
            "model": os.getenv("JUDGE_MODEL", "qwen/qwen3.5-35b-a3b"),
            "base_url": os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"),
            "drop_meters": ["المتدارك"],
        },
    )

    rows = load_filtered_rows(max_bayts=args.max_bayts, drop_meters={"المتدارك"})
    selected = choose_rows(rows, limit=args.num_rows, seed=args.seed, distinct_meters=args.distinct_meters)
    log(f"loaded {len(rows)} filtered rows and selected {len(selected)} rows")

    client = build_client(cache_dir)
    selected_rows_path = output_dir / "selected_rows.jsonl"
    results_path = output_dir / "results.jsonl"
    comparison_path = output_dir / "prompt_comparison.txt"
    latencies: list[float] = []

    with selected_rows_path.open("w", encoding="utf-8") as selected_f, results_path.open("w", encoding="utf-8") as results_f, comparison_path.open("w", encoding="utf-8") as compare_f:
        for idx, row in enumerate(selected, start=1):
            selected_payload = {
                "sample_index": idx,
                "id": int(row["id"]),
                "base_meter": str(row["base_meter"]),
                "form": str(row["form"]),
                "requested_bayts": int(row["requested_bayts"]),
                "description": str(row["description"]),
                "sft_prompt": str(row["sft_prompt"]),
                "sft_completion": str(row["sft_completion"]),
            }
            selected_f.write(json.dumps(selected_payload, ensure_ascii=False) + "\n")

            log(
                f"regenerating sample {idx}/{len(selected)} | id={row['id']} meter={row['base_meter']} bayts={row['requested_bayts']}"
            )
            generated = generate_description(client, row, max_row_attempts=args.max_row_attempts)
            new_description = generated["enhanced_description"] or f"[ERROR] {generated['validation_error'] or generated['response_error'] or 'missing_new_description'}"
            old_description = clean_description(str(row["description"]))
            old_prompt = str(row["sft_prompt"])
            new_prompt = replace_description_in_prompt(old_prompt, str(row["description"]), new_description)

            result_payload = {
                "timestamp_utc": utc_now_iso(),
                "sample_index": idx,
                "id": int(row["id"]),
                "base_meter": str(row["base_meter"]),
                "form": str(row["form"]),
                "requested_bayts": int(row["requested_bayts"]),
                **generated,
                "old_description": old_description,
                "new_description": new_description,
                "old_prompt": old_prompt,
                "new_prompt": new_prompt,
                "poem_text": "\n".join(str(v) for v in row["poem verses"]),
            }
            results_f.write(json.dumps(result_payload, ensure_ascii=False) + "\n")
            if generated.get("latency_sec") is not None:
                latencies.append(float(generated["latency_sec"]))

            compare_f.write("=" * 100 + "\n")
            compare_f.write(
                f"Sample {idx} | id={row['id']} | meter={row['base_meter']} | form={row['form']} | bayts={row['requested_bayts']}\n"
            )
            compare_f.write("=" * 100 + "\n\n")
            compare_f.write(f"LATENCY: {float(generated.get('latency_sec') or 0.0):.3f} seconds\n\n")
            compare_f.write("OLD DESCRIPTION:\n")
            compare_f.write(old_description + "\n\n")
            compare_f.write("NEW DESCRIPTION:\n")
            compare_f.write(new_description + "\n\n")
            compare_f.write("OLD PROMPT:\n")
            compare_f.write(old_prompt + "\n\n")
            compare_f.write("NEW PROMPT:\n")
            compare_f.write(new_prompt + "\n\n")
            compare_f.write("REFERENCE POEM:\n")
            compare_f.write("\n".join(str(v) for v in row["poem verses"]) + "\n\n")

    successful = 0
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if json.loads(line).get("status") == "ok":
            successful += 1

    avg_latency = mean(latencies) if latencies else None
    median_latency = median(latencies) if latencies else None
    projected_total_seconds = avg_latency * len(rows) if avg_latency is not None else None

    dump_json(
        output_dir / "summary.json",
        {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "filtered_source_rows": len(rows),
            "selected_rows": len(selected),
            "successful_rows": successful,
            "latency_seconds_avg": avg_latency,
            "latency_seconds_median": median_latency,
            "projected_total_seconds_for_filtered_rows": projected_total_seconds,
            "projected_total_hours_for_filtered_rows": (projected_total_seconds / 3600.0) if projected_total_seconds is not None else None,
            "projected_total_days_for_filtered_rows": (projected_total_seconds / 86400.0) if projected_total_seconds is not None else None,
            "comparison_file": str(comparison_path),
            "results_file": str(results_path),
        },
    )


def run_worker_mode(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.workers_manifest).read_text(encoding="utf-8"))
    manifest_run_name = str(manifest["run_name"])
    if args.run_name and args.run_name != manifest_run_name:
        raise RuntimeError(f"run_name mismatch: {args.run_name} != {manifest_run_name}")
    run_name = manifest_run_name

    shard_info = None
    for shard in manifest["worker_shards"]:
        if int(shard["worker_id"]) == args.worker_id:
            shard_info = shard
            break
    if shard_info is None:
        raise RuntimeError(f"worker_id {args.worker_id} not present in manifest")

    worker_dir = ensure_dir(Path(shard_info["output_dir"]))
    cache_dir = ensure_dir(worker_dir / "cache")
    log_path = worker_dir / "run.log"
    results_path = worker_dir / "results.jsonl"
    summary_path = worker_dir / "summary.json"
    remote_cache_dir = ensure_dir(worker_dir / "remote_resume")

    def log(msg: str) -> None:
        line = f"{utc_now_iso()} | {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    if results_path.exists() and not args.resume:
        raise RuntimeError(f"{results_path} already exists; rerun with --resume to continue safely")
    results_path.touch(exist_ok=True)

    client = build_client(cache_dir)
    hf_token = os.getenv("HF_TOKEN", "").strip()
    api = HfApi(token=hf_token) if hf_token else None
    repo_id = str(manifest["repo_id"])
    progress_prefix = str(manifest["progress_prefix"])

    local_rows = iter_jsonl(results_path)
    existing_results: dict[int, dict[str, Any]] = {
        int(row["source_index"]): row for row in local_rows if str(row.get("status")) == "ok"
    }

    if args.resume:
        remote_rows, remote_results_path = load_remote_results(
            api=api,
            repo_id=repo_id,
            path_in_repo=f"{progress_prefix}/workers/worker_{args.worker_id:02d}/results.jsonl",
            local_download_dir=remote_cache_dir,
        )
        if remote_rows and remote_results_path is not None and results_path.stat().st_size == 0:
            shutil.copyfile(remote_results_path, results_path)
        for row in remote_rows:
            if str(row.get("status")) != "ok":
                continue
            existing_results.setdefault(int(row["source_index"]), row)

    shard_rows = iter_jsonl(Path(shard_info["shard_path"]))
    if args.max_rows > 0:
        shard_rows = shard_rows[: args.max_rows]

    skipped_existing = 0
    processed_now = 0
    errors_now = 0
    synced_count = 0

    dump_json(
        worker_dir / "config_snapshot.json",
        {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "worker_id": args.worker_id,
            "workers_manifest": args.workers_manifest,
            "style_version": DESCRIPTION_PROMPT_VERSION,
            "max_row_attempts": args.max_row_attempts,
            "sync_every": args.sync_every,
            "resume": bool(args.resume),
            "repo_id": repo_id,
            "progress_prefix": progress_prefix,
            "row_limit": args.max_rows if args.max_rows > 0 else None,
        },
    )

    def sync_progress() -> None:
        nonlocal synced_count
        summary = {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "worker_id": args.worker_id,
            "assigned_rows": len(shard_rows),
            "completed_ok": len(existing_results),
            "skipped_existing": skipped_existing,
            "processed_now": processed_now,
            "errors_now": errors_now,
            "sync_count": synced_count + 1,
        }
        dump_json(summary_path, summary)
        upload_progress_files(api, repo_id, progress_prefix, args.worker_id, results_path, summary_path)
        synced_count += 1

    for idx, row in enumerate(shard_rows, start=1):
        source_index = int(row["source_index"])
        if source_index in existing_results:
            skipped_existing += 1
            continue
        log(
            f"worker={args.worker_id:02d} processing {idx}/{len(shard_rows)} | source_index={source_index} | meter={row['base_meter']} | bayts={row['requested_bayts']}"
        )
        generated = generate_description(client, row, max_row_attempts=args.max_row_attempts)
        result_payload = {
            "timestamp_utc": utc_now_iso(),
            "run_name": run_name,
            "worker_id": args.worker_id,
            "source_index": source_index,
            "id": int(row["id"]),
            "base_meter": str(row["base_meter"]),
            "form": str(row["form"]),
            "requested_bayts": int(row["requested_bayts"]),
            "old_description": clean_description(str(row["description"])),
            **generated,
        }
        append_jsonl(results_path, result_payload)
        processed_now += 1
        if generated["status"] == "ok":
            existing_results[source_index] = result_payload
        else:
            errors_now += 1
        if processed_now % max(args.sync_every, 1) == 0:
            sync_progress()

    sync_progress()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced descriptions for Arabic poetry rows")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bayts", type=int, default=4)
    parser.add_argument("--distinct-meters", action="store_true", default=True)
    parser.add_argument("--workers-manifest", default="")
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sync-every", type=int, default=25)
    parser.add_argument("--max-row-attempts", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)
    args = parse_args()
    if args.workers_manifest:
        if args.worker_id < 0:
            raise RuntimeError("--worker-id is required when --workers-manifest is provided")
        run_worker_mode(args)
        return
    run_sample_mode(args)


if __name__ == "__main__":
    main()
