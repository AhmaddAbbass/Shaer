#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = PROJECT_ROOT / "evaluation" / "outputs"

ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
PROMPT_LEAK_RE = re.compile(r"(البحر الأساسي|الصيغة:|الموضوع:|اكتب\s+\d+\s+شطر)")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv("/root/.env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def setup_logger(run_dir: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def atomic_write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    iso_method = getattr(value, "isoformat", None)
    if callable(iso_method):
        try:
            return iso_method()
        except Exception:
            pass
    return str(value)


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        return out.strip()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def env_snapshot(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "created_at_utc": utc_now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi": run_cmd(["nvidia-smi"]),
        "hf_token_present": bool(os.getenv("HF_TOKEN")),
        "gh_token_present": bool(os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")),
    }
    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
        snap["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.is_available():
            snap["gpu_model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            snap["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            snap["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        snap["torch_error"] = repr(exc)
    for package_name in ("transformers", "datasets", "vllm", "llama_cpp"):
        try:
            module = __import__(package_name)
            snap[f"{package_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            snap[f"{package_name}_error"] = repr(exc)
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    snap["disk_total_gb"] = round(usage.total / 1e9, 2)
    snap["disk_free_gb"] = round(usage.free / 1e9, 2)
    if extra:
        snap.update(extra)
    return snap


def build_meter_label(base_meter: str, form: str, meter_label: str = "") -> str:
    meter_label = str(meter_label or "").strip()
    if meter_label:
        return meter_label
    base_meter = str(base_meter or "").strip()
    form = str(form or "").strip()
    if not form or form == "تام":
        return base_meter
    return f"{form} {base_meter}".strip()


def parse_poem_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").replace("\r", "\n").split("\n") if line.strip()]


def requested_lines_from_row(row: dict[str, Any]) -> int:
    for key in ("requested_num_lines", "sft_num_lines", "requested_lines", "continuation_requested_num_lines"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return int(value)
    bayts = row.get("requested_bayts")
    if bayts is not None and str(bayts).strip():
        return int(bayts) * 2
    return len(parse_poem_lines(str(row.get("reference_completion") or "")))


def build_plain_arabic_instruction_prompt(row: dict[str, Any]) -> str:
    meter = str(row.get("meter_label") or row.get("base_meter") or "").strip()
    form = str(row.get("form") or "").strip()
    requested_num_lines = requested_lines_from_row(row)
    description = str(row.get("enhanced_description") or row.get("description") or "").strip()
    lines = [
        "اكتب قصيدة عربية عمودية كلاسيكية فقط دون مقدمة أو شرح.",
    ]
    if meter:
        lines.append(f"البحر: {meter}")
    if form:
        lines.append(f"الصيغة: {form}")
    if requested_num_lines:
        lines.append(f"عدد الأشطر المطلوب: {requested_num_lines}")
    if description:
        lines.append(f"الموضوع: {description}")
    if requested_num_lines:
        lines.append(f"أخرج {requested_num_lines} أشطارًا فقط ملتزمة بالمطلوب.")
    else:
        lines.append("أخرج الأبيات فقط ملتزمة بالمطلوب.")
    return "\n".join(lines).strip()


def build_fanar_metadata_prompt(row: dict[str, Any]) -> tuple[str, str]:
    lines = parse_poem_lines(str(row.get("reference_completion") or ""))
    if len(lines) < 2:
        return "", "missing first bayt for Fanar prompt"
    hemistich_1 = lines[0]
    hemistich_2 = lines[1]
    meter = str(row.get("fanar_meter") or row.get("base_meter") or "").strip()
    topic = str(row.get("fanar_topic") or "").strip()
    era = str(row.get("fanar_era") or "").strip()
    poet = str(row.get("fanar_poet_id") or row.get("fanar_poet") or "").strip()
    rhyme = str(row.get("fanar_rhyme_letter") or "").strip()
    missing = []
    if not meter:
        missing.append("meter")
    if not topic:
        missing.append("topic")
    if not era:
        missing.append("era")
    if not poet:
        missing.append("poet")
    if not rhyme:
        missing.append("rhyme_letter")
    if missing:
        return "", f"missing Fanar metadata: {', '.join(missing)}"
    prompt = f"{hemistich_1}[{meter}][{topic}][{era}][{poet}]{hemistich_2}[{rhyme}]"
    return prompt, ""


def health_check(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "empty", "empty output"
    lines = parse_poem_lines(raw)
    arabic_chars = len(ARABIC_RE.findall(raw))
    if arabic_chars < 12:
        return "unhealthy", "too few Arabic characters"
    if LATIN_RE.search(raw):
        return "unhealthy", "latin script present"
    if PROMPT_LEAK_RE.search(raw):
        return "unhealthy", "prompt leakage"
    if raw.lstrip().startswith(("{", "[", "```", "#")):
        return "unhealthy", "structured or markdown output"
    if not lines:
        return "unhealthy", "no parseable poem lines"
    if len(set(lines)) <= 1 and len(lines) > 2:
        return "unhealthy", "repeated garbage lines"
    return "ok", ""


def strip_prefix_if_present(prefix: str, text: str) -> tuple[str, bool]:
    prefix = str(prefix or "").strip()
    text = str(text or "").strip()
    if not prefix or not text:
        return text, False
    if text.startswith(prefix):
        return text[len(prefix) :].lstrip("\r\n "), True
    prefix_lines = parse_poem_lines(prefix)
    text_lines = parse_poem_lines(text)
    if prefix_lines and text_lines[: len(prefix_lines)] == prefix_lines:
        stripped = "\n".join(text_lines[len(prefix_lines) :]).strip()
        return stripped, True
    return text, False


def load_completed_keys(path: Path, key_fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
    completed: set[tuple[Any, ...]] = set()
    for row in iter_jsonl(path):
        if row.get("generation_status") != "ok" or row.get("health_status") != "ok":
            continue
        completed.add(tuple(row.get(field) for field in key_fields))
    return completed


def baseline_input_row_id(row: dict[str, Any]) -> str:
    for key in ("input_row_id", "manifest_row_id", "selected_shaer_generation_id", "generation_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return f"source_row_{int(row.get('source_row_index') or 0):05d}"
