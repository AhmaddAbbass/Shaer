#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download

try:
    from curl_cffi import requests as http_requests
except Exception:  # pragma: no cover - fallback for machines without curl_cffi
    import requests as http_requests  # type: ignore


METRICS = [
    "description_adherence",
    "meaning",
    "fluency",
    "coherence",
    "poeticness",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    dataset_id: str
    repo_path: str
    file_format: str
    poem_field: str
    description_field: str
    id_candidates: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_rows: int = 3481


DATASETS: dict[str, DatasetSpec] = {
    "shaer": DatasetSpec(
        key="shaer",
        dataset_id="Shaer-AI/shaer-sft-test",
        repo_path="data/test-00000-of-00001.parquet",
        file_format="parquet",
        poem_field="generated_text",
        description_field="enhanced_description",
        id_candidates=("id",),
        metrics=tuple(METRICS),
    ),
    "ashaar": DatasetSpec(
        key="ashaar",
        dataset_id="Shaer-AI/shaer-eval-ashaar-native-controls",
        repo_path="data/test.jsonl",
        file_format="jsonl",
        poem_field="generated_text",
        description_field="enhanced_description",
        id_candidates=("generation_id", "id", "input_row_id"),
        metrics=("meaning", "fluency", "coherence", "poeticness"),
    ),
    "yehia": DatasetSpec(
        key="yehia",
        dataset_id="Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template",
        repo_path="data/test.jsonl",
        file_format="jsonl",
        poem_field="generated_text",
        description_field="enhanced_description",
        id_candidates=("generation_id", "id", "input_row_id"),
        metrics=tuple(METRICS),
    ),
    "fanar": DatasetSpec(
        key="fanar",
        dataset_id="Shaer-AI/fanar-eval-native-prompt",
        repo_path="data/train-00000-of-00001.parquet",
        file_format="parquet",
        poem_field="generated_text",
        description_field="enhanced_description",
        id_candidates=("generation_id", "id", "input_row_id"),
        metrics=("meaning", "fluency", "coherence", "poeticness"),
    ),
}


TARGET_MODELS: dict[str, str] = {
    "qwen/qwen3.7-max": "qwen3_7_max",
    "openai/gpt-5.6-terra": "gpt5_6_terra",
}


class OpenRouterJudge:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        cache_dir: Path,
        base_url: str,
        timeout: float,
        retries: int,
        retry_sleep: float,
        request_sleep: float,
        strict_provider_routing: bool,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.request_sleep = request_sleep
        self.strict_provider_routing = strict_provider_routing
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.session = http_requests.Session(impersonate="chrome124")
        except TypeError:
            self.session = http_requests.Session()

    def score(self, *, system_prompt: str, user_prompt: str, cache_namespace: str) -> dict[str, Any]:
        payload = self._payload(system_prompt, user_prompt)
        cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        cache_path = self.cache_dir / cache_namespace / f"{cache_key}.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cached = read_valid_cache(cache_path)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        last_error = ""
        started = time.time()
        for attempt in range(self.retries + 1):
            try:
                response_json = self._call(payload)
                score = parse_score(response_json)
                out = {
                    "score": score,
                    "response": response_json,
                    "model": self.model,
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "latency_sec": time.time() - started,
                    "error": "",
                }
                atomic_write_json(cache_path, out)
                return out
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_sleep * (attempt + 1))

        out = {
            "score": None,
            "response": None,
            "model": self.model,
            "cache_key": cache_key,
            "cache_hit": False,
            "latency_sec": time.time() - started,
            "error": last_error,
        }
        atomic_write_json(cache_path, out)
        raise RuntimeError(last_error)

    def _payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "reasoning": {"effort": "none"},
            "max_tokens": 128,
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
            payload["provider"] = {"require_parameters": True, "allow_fallbacks": False}
        return payload

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Shaer-Additional-Judges"),
        }
        referrer = os.getenv("OPENROUTER_REFERRER", "").strip()
        if referrer:
            headers["HTTP-Referer"] = referrer

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if self.request_sleep > 0:
            time.sleep(self.request_sleep)
        if getattr(response, "status_code", 200) >= 400:
            body = getattr(response, "text", "")
            raise RuntimeError(f"HTTP {response.status_code}: {body[:1000]}")
        return response.json()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone additional Shaer judge evaluations.")
    parser.add_argument("--models", nargs="+", default=list(TARGET_MODELS), help="OpenRouter model IDs.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--prompt-file", type=Path, default=Path(__file__).with_name("judge_prompts_v2_strict.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("judge_runs") / "additional_judges")
    parser.add_argument("--env-file", action="append", type=Path, default=[], help="Extra .env file(s) to load.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-rows", type=int, default=0, help="Pilot mode. Never use with --push-to-hf.")
    parser.add_argument("--ignore-existing-columns", action="store_true")
    parser.add_argument("--push-to-hf", action="store_true")
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "120")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("JUDGE_MAX_RETRIES", "3")))
    parser.add_argument("--retry-sleep", type=float, default=float(os.getenv("JUDGE_RETRY_SLEEP_SECONDS", "2")))
    parser.add_argument("--request-sleep", type=float, default=float(os.getenv("JUDGE_REQUEST_SLEEP_SECONDS", "0.25")))
    parser.add_argument("--strict-provider-routing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(args.env_file)

    if args.push_to_hf and args.limit_rows:
        raise RuntimeError("--push-to-hf is blocked when --limit-rows is set")
    if args.workers < 1:
        raise RuntimeError("--workers must be at least 1")

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    hf_token = os.getenv("HF_TOKEN", "").strip() or None

    prompt_config = load_prompt_config(args.prompt_file)
    args.output_root.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    id_fields: dict[str, str] = {}
    for dataset_key in args.datasets:
        spec = DATASETS[dataset_key]
        rows = download_dataset(spec, args.output_root / "source_files", hf_token)
        if args.limit_rows:
            rows = rows[: args.limit_rows]
        id_field = detect_id_field(rows, spec.id_candidates)
        ensure_unique_ids(rows, id_field, spec.key)
        all_rows[dataset_key] = rows
        id_fields[dataset_key] = id_field
        log(f"loaded dataset={dataset_key} rows={len(rows)} id_field={id_field}")

    run_summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": args.models,
        "datasets": args.datasets,
        "limit_rows": args.limit_rows,
        "workers": args.workers,
        "prompt_file": str(args.prompt_file),
        "model_dataset_stats": [],
    }

    for model in args.models:
        prefix = model_prefix(model)
        client = OpenRouterJudge(
            api_key=openrouter_key,
            model=model,
            cache_dir=args.output_root / "cache",
            base_url=args.base_url,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
            request_sleep=args.request_sleep,
            strict_provider_routing=args.strict_provider_routing,
        )
        for dataset_key in args.datasets:
            stats = evaluate_dataset_model(
                client=client,
                model=model,
                prefix=prefix,
                spec=DATASETS[dataset_key],
                rows=all_rows[dataset_key],
                id_field=id_fields[dataset_key],
                prompt_config=prompt_config,
                workers=args.workers,
                output_root=args.output_root,
                reuse_existing=not args.ignore_existing_columns,
            )
            run_summary["model_dataset_stats"].append(stats)

    validation = validate_all(all_rows, id_fields, args.models, args.datasets, args.limit_rows)
    run_summary["validation"] = validation

    updated_files = write_updated_dataset_files(all_rows, args.datasets, args.output_root)
    run_summary["updated_files"] = {key: str(path) for key, path in updated_files.items()}

    if args.push_to_hf:
        upload_updated_files(updated_files, hf_token)
        run_summary["pushed_to_hf"] = True
    else:
        run_summary["pushed_to_hf"] = False

    atomic_write_json(args.output_root / "validation_summary.json", run_summary)
    write_markdown_summary(args.output_root / "additional_judge_results.md", run_summary)
    log(f"done summary={args.output_root / 'additional_judge_results.md'}")
    return 0


def load_env(extra_paths: list[Path]) -> None:
    script_path = Path(__file__).resolve()
    candidates: list[Path] = []
    candidates.extend(extra_paths)
    candidates.append(Path.cwd() / ".env")
    for parent in [script_path.parent, *script_path.parents]:
        candidates.append(parent / ".env")

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        for key, value in parse_env_file(resolved).items():
            os.environ.setdefault(key, value)


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_prompt_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "shared_system_prompt" not in cfg or "metrics" not in cfg:
        raise RuntimeError(f"bad prompt file: {path}")
    for metric in METRICS:
        if metric not in cfg["metrics"]:
            raise RuntimeError(f"prompt file missing metric: {metric}")
    return cfg


def download_dataset(spec: DatasetSpec, source_root: Path, hf_token: str | None) -> list[dict[str, Any]]:
    local_dir = source_root / spec.key
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = Path(
        hf_hub_download(
            repo_id=spec.dataset_id,
            filename=spec.repo_path,
            repo_type="dataset",
            token=hf_token,
            cache_dir=local_dir,
        )
    )
    if spec.file_format == "parquet":
        df = pd.read_parquet(local_path)
        df = df.astype(object).where(pd.notna(df), None)
        return df.to_dict(orient="records")
    if spec.file_format == "jsonl":
        rows: list[dict[str, Any]] = []
        with local_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise RuntimeError(f"unsupported dataset format: {spec.file_format}")


def evaluate_dataset_model(
    *,
    client: OpenRouterJudge,
    model: str,
    prefix: str,
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
    id_field: str,
    prompt_config: dict[str, Any],
    workers: int,
    output_root: Path,
    reuse_existing: bool,
) -> dict[str, Any]:
    log(f"start model={model} dataset={spec.key} rows={len(rows)} metrics={','.join(spec.metrics)}")
    system_prompt = str(prompt_config["shared_system_prompt"])
    started = time.time()
    stats = {
        "model": model,
        "prefix": prefix,
        "dataset": spec.key,
        "rows": len(rows),
        "metrics": list(spec.metrics),
        "expected_calls": len(rows) * len(spec.metrics),
        "existing_scores": 0,
        "cache_hits": 0,
        "api_calls": 0,
        "errors": 0,
        "latency_sec": 0.0,
    }

    pending: list[tuple[dict[str, Any], str, str, str, str]] = []
    for row in rows:
        row_id = str(row[id_field])
        poem = require_text(row, spec.poem_field, spec.key, row_id)
        description = str(row.get(spec.description_field) or "")
        for metric in spec.metrics:
            column = f"{prefix}_{metric}"
            if reuse_existing and is_valid_score(row.get(column)):
                row[column] = int(row[column])
                stats["existing_scores"] += 1
                continue
            user_prompt = build_user_prompt(prompt_config, metric, poem, description)
            cache_namespace = f"{prefix}/{spec.key}/{metric}"
            pending.append((row, row_id, metric, column, user_prompt, cache_namespace))  # type: ignore[arg-type]

    def run_one(task: tuple[dict[str, Any], str, str, str, str, str]) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
        row, row_id, metric, column, user_prompt, cache_namespace = task
        result = client.score(system_prompt=system_prompt, user_prompt=user_prompt, cache_namespace=cache_namespace)
        return row, metric, column, result

    completed = 0
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(run_one, task): task for task in pending}
        for future in futures.as_completed(future_map):
            task = future_map[future]
            try:
                row, metric, column, result = future.result()
                score = result.get("score")
                if not is_valid_score(score):
                    raise RuntimeError(f"invalid score for metric={metric}: {score}")
                row[column] = int(score)
                if result.get("cache_hit"):
                    stats["cache_hits"] += 1
                else:
                    stats["api_calls"] += 1
                stats["latency_sec"] += float(result.get("latency_sec") or 0)
            except Exception as exc:
                stats["errors"] += 1
                _, row_id, metric, column, _, _ = task
                raise RuntimeError(f"failed model={model} dataset={spec.key} row_id={row_id} metric={metric} column={column}: {exc}") from exc
            completed += 1
            if completed == len(pending) or completed % 100 == 0:
                log(f"progress model={model} dataset={spec.key} completed={completed}/{len(pending)}")

    stats["wall_sec"] = time.time() - started
    write_scored_rows(output_root / "scored_rows" / prefix / spec.key / "scores.jsonl", rows, id_field, prefix, spec.metrics)
    atomic_write_json(output_root / "scored_rows" / prefix / spec.key / "summary.json", stats)
    log(f"finish model={model} dataset={spec.key} api_calls={stats['api_calls']} cache_hits={stats['cache_hits']} wall_sec={stats['wall_sec']:.1f}")
    return stats


def build_user_prompt(prompt_config: dict[str, Any], metric: str, poem: str, description: str) -> str:
    metric_cfg = prompt_config["metrics"][metric]
    template = str(metric_cfg["user_template"])
    if metric_cfg.get("requires_description") and not description.strip():
        raise RuntimeError(f"metric requires description but description is empty: {metric}")
    return template.format(poem=poem, description=description)


def require_text(row: dict[str, Any], field: str, dataset_key: str, row_id: str) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise RuntimeError(f"missing {field} for dataset={dataset_key} row_id={row_id}")
    return str(value)


def parse_score(response_json: dict[str, Any]) -> int:
    choices = response_json.get("choices")
    if not choices:
        raise RuntimeError("response has no choices")
    message = choices[0].get("message") or {}

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        args = function.get("arguments")
        score = parse_score_payload(args)
        if score is not None:
            return score

    score = parse_score_payload(message.get("content"))
    if score is not None:
        return score
    raise RuntimeError(f"could not parse score from response: {json.dumps(response_json, ensure_ascii=False)[:1000]}")


def parse_score_payload(payload: Any) -> int | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return normalize_score(payload.get("score"))
    if isinstance(payload, list):
        payload = "".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in payload)
    text = str(payload).strip()
    if not text:
        return None
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return normalize_score(parsed.get("score"))
    except Exception:
        pass
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return normalize_score(parsed.get("score"))
        except Exception:
            pass
    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))
    return None


def normalize_score(value: Any) -> int | None:
    try:
        score = int(value)
    except Exception:
        return None
    if 1 <= score <= 5:
        return score
    return None


def is_valid_score(value: Any) -> bool:
    return normalize_score(value) is not None


def read_valid_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if is_valid_score(data.get("score")):
        data["score"] = int(data["score"])
        return data
    return None


def detect_id_field(rows: list[dict[str, Any]], candidates: tuple[str, ...]) -> str:
    if not rows:
        raise RuntimeError("empty dataset")
    for field in candidates:
        if field in rows[0]:
            return field
    raise RuntimeError(f"could not find ID field from candidates: {candidates}")


def ensure_unique_ids(rows: list[dict[str, Any]], id_field: str, dataset_key: str) -> None:
    seen: set[Any] = set()
    for row in rows:
        row_id = row.get(id_field)
        if row_id in seen:
            raise RuntimeError(f"duplicate row ID in {dataset_key}: {row_id}")
        seen.add(row_id)


def validate_all(
    all_rows: dict[str, list[dict[str, Any]]],
    id_fields: dict[str, str],
    models: list[str],
    datasets: list[str],
    limit_rows: int,
) -> dict[str, Any]:
    validation: dict[str, Any] = {"datasets": {}, "aggregates": {}}
    for dataset_key in datasets:
        spec = DATASETS[dataset_key]
        rows = all_rows[dataset_key]
        expected_rows = limit_rows or spec.expected_rows
        if len(rows) != expected_rows:
            raise RuntimeError(f"{dataset_key} row count mismatch: got {len(rows)}, expected {expected_rows}")
        ensure_unique_ids(rows, id_fields[dataset_key], dataset_key)
        validation["datasets"][dataset_key] = {"rows": len(rows), "models": {}}
        validation["aggregates"][dataset_key] = {}
        for model in models:
            prefix = model_prefix(model)
            validation["datasets"][dataset_key]["models"][prefix] = {}
            validation["aggregates"][dataset_key][prefix] = {}
            for metric in spec.metrics:
                column = f"{prefix}_{metric}"
                scores = [row.get(column) for row in rows]
                bad = [score for score in scores if not is_valid_score(score)]
                if bad:
                    raise RuntimeError(f"{dataset_key}/{prefix}/{metric} has {len(bad)} invalid scores")
                numeric = [int(score) for score in scores]
                validation["datasets"][dataset_key]["models"][prefix][metric] = {
                    "column": column,
                    "missing": 0,
                    "min": min(numeric),
                    "max": max(numeric),
                    "mean": sum(numeric) / len(numeric),
                }
                validation["aggregates"][dataset_key][prefix][metric] = round(sum(numeric) / len(numeric), 4)
            for metric in set(METRICS) - set(spec.metrics):
                column = f"{prefix}_{metric}"
                if column in rows[0] and any(row.get(column) is not None for row in rows):
                    raise RuntimeError(f"{dataset_key} should not contain non-null non-applicable column: {column}")
    return validation


def write_updated_dataset_files(
    all_rows: dict[str, list[dict[str, Any]]],
    datasets: list[str],
    output_root: Path,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for dataset_key in datasets:
        spec = DATASETS[dataset_key]
        filename = Path(spec.repo_path).name
        path = output_root / "updated_datasets" / dataset_key / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = all_rows[dataset_key]
        if spec.file_format == "jsonl":
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        elif spec.file_format == "parquet":
            pd.DataFrame(rows).to_parquet(path, index=False)
        else:
            raise RuntimeError(f"unsupported format: {spec.file_format}")
        out[dataset_key] = path
    return out


def upload_updated_files(updated_files: dict[str, Path], hf_token: str | None) -> None:
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required for --push-to-hf")
    api = HfApi(token=hf_token)
    for dataset_key, path in updated_files.items():
        spec = DATASETS[dataset_key]
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=spec.repo_path,
            repo_id=spec.dataset_id,
            repo_type="dataset",
            commit_message="Add additional judge metric columns",
        )


def write_scored_rows(path: Path, rows: list[dict[str, Any]], id_field: str, prefix: str, metrics: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            out = {id_field: row.get(id_field)}
            for metric in metrics:
                out[f"{prefix}_{metric}"] = row.get(f"{prefix}_{metric}")
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Additional Judge Results",
        "",
        f"Created at: `{summary['created_at']}`",
        f"Limit rows: `{summary['limit_rows'] or 'none'}`",
        f"Workers: `{summary['workers']}`",
        "",
        "## Aggregates",
        "",
    ]
    aggregates = summary["validation"]["aggregates"]
    for dataset_key, model_scores in aggregates.items():
        lines.append(f"### {dataset_key}")
        lines.append("")
        lines.append("| Judge | Description | Meaning | Fluency | Coherence | Poeticness |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for prefix, metric_scores in model_scores.items():
            row = [
                prefix,
                fmt_metric(metric_scores.get("description_adherence")),
                fmt_metric(metric_scores.get("meaning")),
                fmt_metric(metric_scores.get("fluency")),
                fmt_metric(metric_scores.get("coherence")),
                fmt_metric(metric_scores.get("poeticness")),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    lines.extend(["## Run Stats", ""])
    lines.append("| Model | Dataset | API Calls | Cache Hits | Existing | Wall Sec |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for stat in summary["model_dataset_stats"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{stat['model']}`",
                    f"`{stat['dataset']}`",
                    str(stat["api_calls"]),
                    str(stat["cache_hits"]),
                    str(stat["existing_scores"]),
                    f"{float(stat['wall_sec']):.1f}",
                ]
            )
            + " |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def model_prefix(model: str) -> str:
    if model in TARGET_MODELS:
        return TARGET_MODELS[model]
    prefix = model.lower().replace("/", "_").replace("-", "_").replace(".", "_")
    return re.sub(r"[^a-z0-9_]+", "_", prefix).strip("_")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(tmp_name).replace(path)
    except Exception:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
