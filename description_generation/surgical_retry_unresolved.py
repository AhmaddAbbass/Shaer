#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from huggingface_hub import HfApi

from regenerate_descriptions import (
    ENV_PATH,
    OUTPUT_ROOT,
    clean_description,
    extract_chat_content,
    extract_new_description,
    upload_progress_files,
    utc_now_iso,
    validate_description_text,
)
from prompt_contracts import build_description_user_prompt


PLAIN_RETRY_VERSION = "plain_text_retry_v1"
PLAIN_SYSTEM_PROMPTS = [
    """أنت تكتب وصفًا عربيًا لقصيدة.

اكتب سطرًا عربيًا واحدًا فقط يبدأ بـ "القصيدة تتحدث عن...".
الجواب يجب أن يكون وصفًا فقط، لا طلبًا ولا أمرًا ولا شرحًا زائدًا.
التزم بالمعنى الظاهر في الأبيات، وكن موجزًا نسبيًا.
لا تكتب JSON.
لا تكتب مفاتيح مثل new_description.
لا تكتب أقواسًا أو أقواسًا معقوفة أو علامات اقتباس حول الجواب.
لا تكتب أي مقدمة أو تعليق خارج الوصف نفسه.
""",
    """المطلوب وصف عربي واحد لقصيدة.

أعد الجواب كسطر واحد فقط يبدأ بـ "القصيدة تتحدث عن...".
ممنوع JSON وممنوع الكود وممنوع الشرح وممنوع التعداد.
صف معنى الأبيات وصورها ونبرتها العامة من غير اختراع معانٍ غير ظاهرة.
""",
    """اكتب وصفًا عربيًا واحدًا فقط للقصيدة.

النتيجة المطلوبة:
- سطر واحد فقط
- يبدأ بـ "القصيدة تتحدث عن..."
- بلا JSON
- بلا مفاتيح
- بلا علامات اقتباس
- بلا أي كلام زائد
""",
]


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


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_ok_results(manifests: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    ok_rows: dict[int, dict[str, Any]] = {}
    for manifest in manifests:
        for shard in manifest["worker_shards"]:
            results_path = Path(shard["output_dir"]) / "results.jsonl"
            for row in iter_jsonl(results_path):
                if str(row.get("status")) != "ok":
                    continue
                ok_rows.setdefault(int(row["source_index"]), row)
    return ok_rows


def unresolved_rows(base_manifest: dict[str, Any], other_manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = collect_ok_results([base_manifest, *other_manifests])
    staged_rows = iter_jsonl(Path(base_manifest["staged_rows_path"]))
    return [row for row in staged_rows if int(row["source_index"]) not in ok_rows]


def strip_wrappers(text: str) -> str:
    out = clean_description(text)
    out = re.sub(r"^```(?:json)?\s*", "", out)
    out = re.sub(r"\s*```$", "", out)
    out = re.sub(r"^(?:new_description|الوصف(?: الجديد)?)\s*[:：-]\s*", "", out, flags=re.IGNORECASE)
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {'"', "'"}:
        out = out[1:-1].strip()
    return clean_description(out)


def extract_keyed_description_via_regex(text: str) -> str | None:
    match = re.search(
        r"""["']new_description["']\s*:\s*["'](.*?)["']\s*(?:\}|,|$)""",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = match.group(1)
    value = value.replace(r"\/", "/").replace(r"\"", '"').replace(r"\n", " ").replace(r"\t", " ")
    return clean_description(value)


def extract_plain_candidate(content: str) -> tuple[str | None, str | None]:
    candidate, reason = extract_new_description(content)
    if candidate:
        return candidate, None
    regex_candidate = extract_keyed_description_via_regex(content)
    if regex_candidate:
        valid, validation_reason = validate_description_text(regex_candidate)
        if valid and regex_candidate.startswith("القصيدة تتحدث عن"):
            return regex_candidate, None
    raw = strip_wrappers(content)
    if not raw:
        return None, reason or "empty_content"
    valid, validation_reason = validate_description_text(raw)
    if not valid:
        return None, validation_reason
    if not raw.startswith("القصيدة تتحدث عن"):
        return None, "missing_prefix"
    return raw, None


class PlainTextOpenRouterClient:
    def __init__(self, cache_dir: Path) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing")
        self.model = os.getenv("JUDGE_MODEL", "qwen/qwen3.5-35b-a3b").strip()
        self.base_url = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").strip()
        self.timeout = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "3"))
        self.referrer = os.getenv("OPENROUTER_REFERRER", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "Shaer-Description-Generation").strip()
        self.cache_dir = ensure_dir(cache_dir)

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

    def chat_plain(self, system_prompt: str, user_prompt: str, cache_salt: str = "") -> tuple[dict[str, Any], bool]:
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
            "max_tokens": 220,
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


def try_row(client: PlainTextOpenRouterClient, row: dict[str, Any], max_row_attempts: int) -> dict[str, Any]:
    total_latency = 0.0
    last_cached = False
    last_reason = None
    last_error = None
    user_prompt = (
        build_description_user_prompt(row["poem verses"])
        + "\n\nأعد الجواب كسطر عربي واحد فقط يبدأ بـ \"القصيدة تتحدث عن...\" ومن دون JSON أو مفاتيح أو شرح زائد."
    )
    for attempt in range(max_row_attempts):
        system_prompt = PLAIN_SYSTEM_PROMPTS[attempt % len(PLAIN_SYSTEM_PROMPTS)]
        response_obj, cached = client.chat_plain(system_prompt, user_prompt, cache_salt=f"plain_retry_{attempt}")
        last_cached = cached
        total_latency += float(response_obj.get("latency_sec") or 0.0)
        content = extract_chat_content(response_obj.get("response") or {})
        candidate, reason = extract_plain_candidate(content)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Surgical plaintext retry for unresolved description rows")
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--extra-manifest", action="append", default=[])
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-row-attempts", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    args = parse_args()
    base_manifest = load_manifest(Path(args.base_manifest))
    extra_manifests = [load_manifest(Path(path)) for path in args.extra_manifest]
    rows = unresolved_rows(base_manifest, extra_manifests)

    run_root = ensure_dir(OUTPUT_ROOT / args.run_name)
    results_path = run_root / "results.jsonl"
    summary_path = run_root / "summary.json"
    cache_dir = ensure_dir(run_root / "cache")
    api = HfApi(token=os.getenv("HF_TOKEN", "").strip()) if os.getenv("HF_TOKEN", "").strip() else None
    progress_prefix = f"progress/{args.run_name}"

    if results_path.exists():
        results_path.unlink()

    client = PlainTextOpenRouterClient(cache_dir=cache_dir)
    ok_rows = 0
    error_rows = 0

    for index, row in enumerate(rows, start=1):
        generated = try_row(client, row, max_row_attempts=args.max_row_attempts)
        payload = {
            "timestamp_utc": utc_now_iso(),
            "run_name": args.run_name,
            "retry_version": PLAIN_RETRY_VERSION,
            "source_index": int(row["source_index"]),
            "id": int(row["id"]),
            "base_meter": str(row["base_meter"]),
            "form": str(row["form"]),
            "requested_bayts": int(row["requested_bayts"]),
            "old_description": clean_description(str(row["description"])),
            **generated,
        }
        append_jsonl(results_path, payload)
        ok_rows += 1 if generated["status"] == "ok" else 0
        error_rows += 1 if generated["status"] == "error" else 0
        print(
            f"{utc_now_iso()} | row={index}/{len(rows)} | source_index={row['source_index']} | status={generated['status']} | attempts={generated['attempt_count']}",
            flush=True,
        )

    summary = {
        "timestamp_utc": utc_now_iso(),
        "run_name": args.run_name,
        "base_manifest": args.base_manifest,
        "extra_manifests": args.extra_manifest,
        "retry_version": PLAIN_RETRY_VERSION,
        "assigned_rows": len(rows),
        "completed_ok": ok_rows,
        "errors_now": error_rows,
    }
    dump_json(summary_path, summary)
    upload_progress_files(api, base_manifest["repo_id"], progress_prefix, 0, results_path, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
