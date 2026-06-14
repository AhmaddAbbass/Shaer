#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from curl_cffi import requests
import yaml

from judge_common import load_env, read_jsonl


PROMPT_FILE = Path("evaluation/judge_prompts.yaml")
DEFAULT_METRICS = [
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


class JudgeClient:
    def __init__(
        self,
        *,
        cache_dir: str,
        cache_namespace: str,
        model: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 128,
    ) -> None:
        self.base_url = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.getenv("JUDGE_MODEL", "qwen/qwen3.5-35b-a3b").strip()
        self.referrer = os.getenv("OPENROUTER_REFERRER", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "Shaer-Evaluation").strip()
        self.timeout = float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "3"))
        self.retry_sleep_seconds = float(os.getenv("JUDGE_RETRY_SLEEP_SECONDS", "2"))
        self.request_sleep_seconds = float(os.getenv("JUDGE_REQUEST_SLEEP_SECONDS", "0.25"))
        self.strict_provider_routing = os.getenv("JUDGE_STRICT_PROVIDER_ROUTING", "").strip().lower() in {"1", "true", "yes"}
        self.max_tokens = int(max_tokens)
        self.json_mode = bool(json_mode)
        self.cache_dir = Path(cache_dir) / cache_namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session(impersonate="chrome124")

    def judge(self, system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], bool]:
        payload = self._build_payload(system_prompt, user_prompt)
        cache_key = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8")), True

        last_error = ""
        response_json = None
        t0 = time.time()
        for attempt in range(self.max_retries + 1):
            try:
                response_json = self._call_once(payload)
                last_error = ""
                break
            except requests.RequestsError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))

        out = {
            "response": response_json,
            "error": last_error,
            "latency_sec": time.time() - t0,
            "model": self.model,
            "cache_key": cache_key,
        }
        cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return out, False

    def _build_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": "none"},
            "max_tokens": self.max_tokens,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_score",
                        "description": "Submit the integer score for the requested metric.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                            },
                            "required": ["score"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "submit_score"}},
        }
        if self.strict_provider_routing:
            payload["provider"] = {
                "require_parameters": True,
                "allow_fallbacks": False,
            }
        return payload

    def _call_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referrer:
            headers["HTTP-Referer"] = self.referrer
        if self.app_title:
            headers["X-Title"] = self.app_title
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if self.request_sleep_seconds > 0:
            time.sleep(self.request_sleep_seconds)
        response.raise_for_status()
        return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Arabic poems with one metric at a time using OpenRouter.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metric", required=True, choices=DEFAULT_METRICS)
    parser.add_argument("--prompt-file", default=str(PROMPT_FILE))
    parser.add_argument("--poem-field", default="generated_text")
    parser.add_argument("--description-field", default="enhanced_description")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--cache-dir", default="evaluation/outputs/judge_cache")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-model", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    prompt_config = load_prompt_config(Path(args.prompt_file))
    metric_cfg = prompt_config["metrics"][args.metric]
    rows = read_jsonl(Path(args.input_jsonl))
    if args.limit_rows:
        rows = rows[: int(args.limit_rows)]

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run_preview(rows, prompt_config, metric_cfg, args)
        return 0

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it before running judge_llm.py without --dry-run.")

    judge_model = str(args.judge_model or os.getenv("JUDGE_MODEL", "")).strip()
    client = JudgeClient(
        cache_dir=str(args.cache_dir),
        cache_namespace=f"judge_{args.metric}",
        model=judge_model or None,
        json_mode=True,
        max_tokens=128,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            poem = str(row.get(args.poem_field) or "").strip()
            description = str(row.get(args.description_field) or row.get("description") or "").strip()
            if not poem:
                continue
            user_prompt = render_user_prompt(metric_cfg, poem=poem, description=description)
            response_obj, cache_hit = client.judge(prompt_config["shared_system_prompt"], user_prompt)
            parsed = extract_judge_json(response_obj, score_key="score")
            usage = extract_usage(response_obj.get("response"))
            record = {
                args.id_field: row.get(args.id_field, ""),
                "metric": args.metric,
                "score": safe_int_score(parsed.get("score")),
                "judge_model": client.model,
                "judge_prompt_version": str(prompt_config.get("version") or ""),
                "judge_cache_hit": bool(cache_hit),
                "judge_error": str(parsed.get("error") or ""),
                "judge_latency_sec": float(response_obj.get("latency_sec") or 0.0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
                "reported_cost_usd": float(usage.get("reported_cost_usd") or 0.0),
                "judge_raw_response": response_obj.get("response"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"scored_rows={len(rows)} metric={args.metric} output={output_path}")
    return 0


def load_prompt_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def render_user_prompt(metric_cfg: dict[str, Any], *, poem: str, description: str) -> str:
    template = str(metric_cfg["user_template"])
    return template.replace("{poem}", poem).replace("{description}", description)


def safe_int_score(value: Any) -> int:
    try:
        score = int(value)
    except Exception:
        return 0
    return max(0, min(5, score))


def extract_content(response: dict[str, Any] | None) -> str:
    response = response or {}
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def extract_tool_score(response: dict[str, Any] | None) -> int | None:
    response = response or {}
    try:
        tool_calls = response["choices"][0]["message"]["tool_calls"]
    except Exception:
        return None
    for tool_call in tool_calls or []:
        try:
            arguments = tool_call["function"]["arguments"]
            parsed = json.loads(arguments)
        except Exception:
            continue
        if isinstance(parsed, dict) and "score" in parsed:
            return safe_int_score(parsed.get("score"))
    return None


def extract_judge_json(response_obj: dict[str, Any], *, score_key: str = "score") -> dict[str, Any]:
    tool_score = extract_tool_score(response_obj.get("response"))
    if tool_score is not None:
        return {
            score_key: tool_score,
            "error": response_obj.get("error") or "",
        }
    content = extract_content(response_obj.get("response"))
    parsed = parse_json_object(content)
    if not isinstance(parsed, dict):
        return {score_key: 0, "error": response_obj.get("error") or "json_not_found"}
    return {
        score_key: parsed.get(score_key, 0),
        "error": response_obj.get("error") or "",
    }


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def extract_usage(response: dict[str, Any] | None) -> dict[str, Any]:
    response = response or {}
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "reasoning_tokens": completion_details.get("reasoning_tokens", 0),
        "reported_cost_usd": usage.get("cost", 0.0),
    }


def dry_run_preview(rows: list[dict[str, Any]], prompt_config: dict[str, Any], metric_cfg: dict[str, Any], args: argparse.Namespace) -> None:
    preview_count = min(2, len(rows))
    payload = {
        "version": prompt_config.get("version"),
        "metric": args.metric,
        "shared_system_prompt": prompt_config["shared_system_prompt"],
        "examples": [],
    }
    for row in rows[:preview_count]:
        poem = str(row.get(args.poem_field) or "").strip()
        description = str(row.get(args.description_field) or row.get("description") or "").strip()
        payload["examples"].append(
            {
                "id": row.get(args.id_field, ""),
                "user_prompt": render_user_prompt(metric_cfg, poem=poem, description=description),
            }
        )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
