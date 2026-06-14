from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import time
import math
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable

DEFAULT_DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
DEFAULT_OUTPUT_DATASET_ID = "Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1"
DEFAULT_BASE_MODEL_ID = "Navid-AI/Yehia-7B-preview"
DEFAULT_SFT_ADAPTER_REPO = "Shaer-AI/Shaer-adapters"
DEFAULT_SFT_ADAPTER_MODE = "fresh_sft/train"
DEFAULT_RUN_ROOT = "grpo/outputs/preprocess_meter_count"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def summarize_scores(values: Iterable[float]) -> dict[str, float]:
    vals = [safe_float(value) for value in values]
    if not vals:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "std": float(pstdev(vals) if len(vals) > 1 else 0.0),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def rate(values: Iterable[bool]) -> float:
    vals = [bool(value) for value in values]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    repo_env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(repo_env, override=False)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_json_or_none(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        f.flush()
        os.fsync(f.fileno())
    return count


def safe_filename(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "row"


def row_uid(source_split: str, source_index: int) -> str:
    return f"{safe_filename(source_split)}_{int(source_index):08d}"


def run_config_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run_config.json"


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "manifest" / "train_rows.jsonl"


def shard_manifest_path(run_dir: str | Path, shard_id: int) -> Path:
    return Path(run_dir) / "manifest" / f"shard_{int(shard_id):02d}.jsonl"


def generated_path(run_dir: str | Path, shard_id: int, uid: str) -> Path:
    return Path(run_dir) / "generated" / f"shard_{int(shard_id):02d}" / f"{safe_filename(uid)}.json"


def scored_path(run_dir: str | Path, uid: str) -> Path:
    return Path(run_dir) / "scored" / f"{safe_filename(uid)}.json"


def score_claim_path(run_dir: str | Path, uid: str) -> Path:
    return Path(run_dir) / "claims" / "score" / f"{safe_filename(uid)}.claim"


def generator_done_path(run_dir: str | Path, shard_id: int) -> Path:
    return Path(run_dir) / "status" / f"generator_shard_{int(shard_id):02d}_done.json"


def status_path(run_dir: str | Path, name: str) -> Path:
    return Path(run_dir) / "status" / f"{safe_filename(name)}.json"


def event_log_path(run_dir: str | Path, name: str) -> Path:
    return Path(run_dir) / "events" / f"{safe_filename(name)}.jsonl"


def failure_log_path(run_dir: str | Path, name: str) -> Path:
    return Path(run_dir) / "failures" / f"{safe_filename(name)}.jsonl"


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    return read_json(run_config_path(run_dir))


def load_manifest(run_dir: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(manifest_path(run_dir))


def load_shard_manifest(run_dir: str | Path, shard_id: int) -> list[dict[str, Any]]:
    path = shard_manifest_path(run_dir, shard_id)
    if path.exists():
        return read_jsonl(path)
    return [row for row in load_manifest(run_dir) if int(row["generator_shard_id"]) == int(shard_id)]


def expected_shard_ids(run_dir: str | Path) -> list[int]:
    cfg = load_run_config(run_dir)
    n = int(cfg.get("num_generator_shards", 2))
    return list(range(n))


def source_prompt(row: dict[str, Any]) -> str:
    for key in ("sft_prompt", "prompt", "messages_prompt"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def source_completion_text(row: dict[str, Any]) -> str:
    value = row.get("sft_completion")
    if isinstance(value, str) and value.strip():
        return value.strip()
    verses = row.get("poem verses") or row.get("poem_verses")
    if isinstance(verses, list):
        return "\n".join(str(line).strip() for line in verses if str(line).strip()).strip()
    return ""


def requested_lines_from_row(row: dict[str, Any]) -> int:
    if row.get("requested_lines") is not None:
        return safe_int(row.get("requested_lines"), 0)
    if row.get("sft_num_lines") is not None:
        return safe_int(row.get("sft_num_lines"), 0)
    if row.get("requested_bayts") is not None:
        return max(0, safe_int(row.get("requested_bayts"), 0) * 2)
    verses = row.get("poem verses") or row.get("poem_verses")
    if isinstance(verses, list):
        return len([line for line in verses if str(line).strip()])
    prompt = source_prompt(row)
    m = re.search(r"اكتب\s+(\d+)\s+شطراً", prompt)
    return safe_int(m.group(1), 0) if m else 0


def requested_bayts_from_row(row: dict[str, Any]) -> int:
    if row.get("requested_bayts") is not None:
        return max(1, safe_int(row.get("requested_bayts"), 1))
    lines = requested_lines_from_row(row)
    return max(1, lines // 2) if lines > 0 else 1


def meter_label_from_row(row: dict[str, Any]) -> str:
    value = row.get("meter_label")
    if value is not None and str(value).strip():
        return str(value).strip()
    base = str(row.get("base_meter") or "").strip()
    form = str(row.get("form") or "").strip()
    if not form or form == "تام":
        return base
    return f"{form} {base}".strip()


def manifest_row_from_source(
    row: dict[str, Any],
    *,
    train_row_index: int,
    source_split: str,
    manifest_order: int,
    generator_shard_id: int,
) -> dict[str, Any]:
    source_index = safe_int(row.get("source_index"), train_row_index)
    requested_bayts = requested_bayts_from_row(row)
    requested_lines = requested_lines_from_row(row) or requested_bayts * 2
    uid = row_uid(source_split, source_index)
    source_id = row.get("id")
    return {
        "row_uid": uid,
        "source_dataset_id": "",
        "source_split": source_split,
        "source_index": int(source_index),
        "source_row_index_in_split": int(train_row_index),
        "source_id": str(source_id) if source_id is not None else "",
        "manifest_order": int(manifest_order),
        "generator_shard_id": int(generator_shard_id),
        "sft_prompt": source_prompt(row),
        "sft_completion": source_completion_text(row),
        "description": str(row.get("description") or ""),
        "enhanced_description": str(row.get("enhanced_description") or ""),
        "base_meter": str(row.get("base_meter") or ""),
        "form": str(row.get("form") or ""),
        "meter_label": meter_label_from_row(row),
        "requested_bayts": int(requested_bayts),
        "requested_lines": int(requested_lines),
        "length_bucket": str(row.get("length_bucket") or ""),
        "sampler_group": str(row.get("sampler_group") or ""),
        "split_group": str(row.get("split_group") or ""),
        "poem_url": str(row.get("poem url") or row.get("poem_url") or ""),
    }


def meter_round_robin_indices(rows: list[dict[str, Any]]) -> list[int]:
    by_meter: dict[str, deque[int]] = defaultdict(deque)
    for idx, row in enumerate(rows):
        by_meter[str(row.get("base_meter") or "__unknown__")].append(idx)
    active = deque(sorted(by_meter))
    ordered: list[int] = []
    while active:
        meter = active.popleft()
        bucket = by_meter[meter]
        ordered.append(bucket.popleft())
        if bucket:
            active.append(meter)
    return ordered


def balanced_subset_indices(rows: list[dict[str, Any]], rows_per_base_meter: int) -> list[int]:
    if rows_per_base_meter <= 0:
        return list(range(len(rows)))
    by_meter: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_meter[str(row.get("base_meter") or "__unknown__")].append(idx)
    selected: list[int] = []
    for meter in sorted(by_meter):
        selected.extend(by_meter[meter][: int(rows_per_base_meter)])
    return sorted(selected)


def try_claim_score(run_dir: str | Path, uid: str, worker_id: str, stale_after_seconds: int) -> Path | None:
    claim = score_claim_path(run_dir, uid)
    ensure_dir(claim.parent)
    payload = {
        "row_uid": uid,
        "worker_id": worker_id,
        "pid": os.getpid(),
        "claimed_at": utc_now_iso(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(claim, flags)
    except FileExistsError:
        try:
            age = time.time() - claim.stat().st_mtime
        except FileNotFoundError:
            return try_claim_score(run_dir, uid, worker_id, stale_after_seconds)
        if age < max(1, int(stale_after_seconds)):
            return None
        try:
            claim.unlink()
        except OSError:
            return None
        try:
            fd = os.open(claim, flags)
        except FileExistsError:
            return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return claim


def release_claim(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def all_generators_done(run_dir: str | Path) -> bool:
    return all(generator_done_path(run_dir, shard_id).exists() for shard_id in expected_shard_ids(run_dir))


def difficulty_rule_spec(num_candidates: int) -> dict[str, Any]:
    num_candidates = max(1, int(num_candidates))
    return {
        "version": "meter_count_k8_relabel_v1",
        "strong_meter_threshold": 0.70,
        "easy_min_strong_exact_count": int(max(1, math.ceil(num_candidates / 2))),
        "easy_min_mean_meter": 0.50,
        "hard_zero_strong_meter_is_hard": True,
        "hard_max_strong_meter_count": int(max(1, math.floor(num_candidates / 8))),
        "hard_max_mean_meter": 0.30,
        "notes": [
            "strong_meter means candidate meter_score >= 0.70",
            "strong_exact means candidate meter_score >= 0.70 and exact bayt count",
            "easy requires reliable strong exact hits",
            "hard includes rows with zero strong meter hits or a single fluke hit with very low mean meter",
            "medium is everything else",
        ],
    }


def classify_difficulty(
    *,
    num_candidates: int,
    mean_meter: float,
    num_strong_meter: int,
    num_strong_exact: int,
) -> str:
    rule = difficulty_rule_spec(num_candidates)
    if num_strong_exact >= int(rule["easy_min_strong_exact_count"]) and mean_meter > float(rule["easy_min_mean_meter"]):
        return "easy"
    if num_strong_meter == 0:
        return "hard"
    if num_strong_meter <= int(rule["hard_max_strong_meter_count"]) and mean_meter < float(rule["hard_max_mean_meter"]):
        return "hard"
    return "medium"


def row_score_summary(candidate_scores: list[dict[str, Any]]) -> dict[str, Any]:
    meter_scores = [safe_float(c["meter_score"]) for c in candidate_scores]
    count_scores = [safe_float(c["count_adherence_score"]) for c in candidate_scores]
    combined_scores = [0.8 * meter + 0.2 * count for meter, count in zip(meter_scores, count_scores)]
    meter_summary = summarize_scores(meter_scores)
    count_summary = summarize_scores(count_scores)
    combined_summary = summarize_scores(combined_scores)
    success_flags = [meter >= 0.70 and count >= 0.999999 for meter, count in zip(meter_scores, count_scores)]
    count_exact_flags = [count >= 0.999999 for count in count_scores]
    strong_meter_flags = [meter >= 0.70 for meter in meter_scores]
    bad_meter_flags = [meter < 0.30 for meter in meter_scores]
    success_rate = rate(success_flags)
    count_exact_rate = rate(count_exact_flags)
    num_candidates = len(candidate_scores)
    num_strong_meter = int(sum(strong_meter_flags))
    num_strong_exact = int(sum(success_flags))
    num_bad_meter = int(sum(bad_meter_flags))
    difficulty_rule = difficulty_rule_spec(num_candidates)
    return {
        "meter_summary": meter_summary,
        "count_adherence_summary": count_summary,
        "meter_count_combined_summary": combined_summary,
        "meter_count_success_rate": success_rate,
        "count_exact_rate": count_exact_rate,
        "num_candidates": int(num_candidates),
        "num_strong_meter_candidates": num_strong_meter,
        "num_strong_exact_candidates": num_strong_exact,
        "num_bad_meter_candidates": num_bad_meter,
        "strong_meter_rate": float(num_strong_meter / num_candidates) if num_candidates else 0.0,
        "strong_exact_rate": float(num_strong_exact / num_candidates) if num_candidates else 0.0,
        "bad_meter_rate": float(num_bad_meter / num_candidates) if num_candidates else 0.0,
        "mean_meter_score": float(meter_summary["mean"]),
        "mean_count_adherence_score": float(count_summary["mean"]),
        "difficulty_rule": difficulty_rule,
        "difficulty": classify_difficulty(
            num_candidates=num_candidates,
            mean_meter=float(meter_summary["mean"]),
            num_strong_meter=num_strong_meter,
            num_strong_exact=num_strong_exact,
        ),
    }


def count_existing_json_files(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.json"))


def run_counts(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest_rows = load_manifest(run_dir) if manifest_path(run_dir).exists() else []
    generated = 0
    for shard_id in expected_shard_ids(run_dir) if run_config_path(run_dir).exists() else [0, 1]:
        generated += count_existing_json_files(run_dir / "generated" / f"shard_{shard_id:02d}")
    scored = count_existing_json_files(run_dir / "scored")
    return {
        "manifest_rows": len(manifest_rows),
        "generated_rows": generated,
        "scored_rows": scored,
        "all_generators_done": all_generators_done(run_dir) if run_config_path(run_dir).exists() else False,
    }


def event_rows(run_dir: str | Path, name: str) -> list[dict[str, Any]]:
    return read_jsonl(event_log_path(run_dir, name))


def parse_args_run_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Preprocess run directory.")
