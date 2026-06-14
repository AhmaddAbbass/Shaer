import csv
from collections import Counter
from difflib import SequenceMatcher
import fcntl
import hashlib
import json
import logging
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
DIGIT_CHAR_RE = re.compile(r"\d")
FORBIDDEN_ARTIFACT_RE = re.compile(r"[`~@#$%^&*_+=<>{}\[\]|\\/]")
ARABIC_TOKEN_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
DEFAULT_SOURCE_DATASET_ID = "Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits"
_POEM_LEXICON_CACHE: dict[str, set[str]] = {}


def load_env():
    load_dotenv(ENV_PATH, override=False)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name, log_path=None, also_stdout=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if log_path:
        ensure_dir(Path(log_path).parent)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if also_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, row):
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_csv_rows(rows: list[dict[str, Any]], path):
    ensure_dir(Path(path).parent)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_env(env_dict):
    out = {}
    for k, v in env_dict.items():
        if any(x in k.upper() for x in ["TOKEN", "KEY", "PASSWORD", "SECRET"]):
            out[k] = "***MASKED***" if v else ""
        else:
            out[k] = v
    return out


def clamp01(x):
    try:
        x = float(x)
    except Exception:
        x = 0.0
    return max(0.0, min(1.0, x))


def mean(xs):
    vals = []
    for x in xs:
        try:
            vals.append(float(x))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def logmean_prob(xs, eps=1e-8):
    vals = []
    for x in xs:
        try:
            x = float(x)
            x = max(eps, min(1.0, x))
            vals.append(x)
        except Exception:
            pass
    if not vals:
        return 0.0
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_arabic_token(token: str) -> str:
    token = str(token or "").strip()
    token = token.replace("ـ", "")
    token = ARABIC_DIACRITICS_RE.sub("", token)
    token = (
        token.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    return token


def tokenize_arabic_words(text: str) -> List[str]:
    tokens = []
    for token in ARABIC_TOKEN_RE.findall(str(text or "")):
        normalized = normalize_arabic_token(token)
        if normalized:
            tokens.append(normalized)
    return tokens


def distinct_n(tokens: List[str], n: int) -> float:
    if n <= 0:
        return 0.0
    if len(tokens) < n:
        return 1.0 if tokens else 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def opening_signature(tokens: List[str], size: int = 2) -> str:
    if not tokens:
        return ""
    return " ".join(tokens[: min(size, len(tokens))])


def jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sequence_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, str(text_a or ""), str(text_b or "")).ratio()


def is_unicode_punctuation(char: str) -> bool:
    return bool(char) and unicodedata.category(char).startswith("P")


def is_arabic_or_punctuation(char: str) -> bool:
    return bool(char) and (
        bool(ARABIC_CHAR_RE.fullmatch(char))
        or is_unicode_punctuation(char)
        or char.isspace()
    )


def contamination_stats(text: str) -> Dict[str, Any]:
    text = str(text or "")
    non_arabic_non_punct_count = sum(1 for char in text if not is_arabic_or_punctuation(char))
    contamination_ratio = non_arabic_non_punct_count / max(len(text), 1)
    return {
        "text_length": len(text),
        "stripped_text_length": len(text.strip()),
        "non_arabic_non_punct_count": int(non_arabic_non_punct_count),
        "contamination_ratio": float(contamination_ratio),
    }


def get_dataset_source_id_for_phase():
    load_env()
    for key in ["GRPO_SOURCE_DATASET_ID", "PHASE1_DATASET_SOURCE_ID", "SOURCE_DATASET_ID"]:
        source = os.getenv(key, "").strip()
        if source:
            return source
    source = os.getenv("PHASE1_DATASET_ID", "").strip()
    if source:
        return source
    return DEFAULT_SOURCE_DATASET_ID


def poem_lexicon_cache_path(dataset_id: str) -> Path:
    digest = hashlib.sha256(str(dataset_id).encode("utf-8")).hexdigest()[:16]
    return PROJECT_ROOT / "grpo" / "cache" / f"poem_lexicon_{digest}.json"


def build_poem_lexicon(dataset_id: str, hf_token: str = None) -> set[str]:
    counter = Counter()
    ds = load_dataset(dataset_id, split="train", token=hf_token)
    for row in ds:
        poem_text = row_to_poem_text(row)
        for token in tokenize_arabic_words(poem_text):
            counter[token] += 1
    return set(counter.keys())


def get_poem_lexicon(dataset_id: str = "", hf_token: str = None) -> set[str]:
    dataset_id = str(dataset_id or get_dataset_source_id_for_phase()).strip() or DEFAULT_SOURCE_DATASET_ID
    cached = _POEM_LEXICON_CACHE.get(dataset_id)
    if cached is not None:
        return cached

    cache_path = poem_lexicon_cache_path(dataset_id)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        tokens = set(payload.get("tokens", []))
        _POEM_LEXICON_CACHE[dataset_id] = tokens
        return tokens

    hf_token = hf_token or os.getenv("HF_TOKEN", "").strip() or None
    tokens = build_poem_lexicon(dataset_id=dataset_id, hf_token=hf_token)
    ensure_dir(cache_path.parent)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_id": dataset_id,
                "token_count": len(tokens),
                "tokens": sorted(tokens),
            },
            f,
            ensure_ascii=False,
        )
    _POEM_LEXICON_CACHE[dataset_id] = tokens
    return tokens


def extract_text(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, str):
        return normalize_whitespace(obj)

    if isinstance(obj, dict):
        if "content" in obj:
            return extract_text(obj["content"])
        if "text" in obj:
            return extract_text(obj["text"])
        if "generated_text" in obj:
            return extract_text(obj["generated_text"])
        if "message" in obj:
            return extract_text(obj["message"])

    if isinstance(obj, list):
        parts = [extract_text(x) for x in obj]
        parts = [p for p in parts if p]
        return "\n".join(parts).strip()

    return normalize_whitespace(str(obj))


def split_poem_lines(text: str) -> List[str]:
    text = normalize_whitespace(text)
    lines = [x.strip() for x in text.split("\n")]
    return [x for x in lines if x]


def pair_lines_to_bayts(lines: List[str]) -> List[str]:
    bayts = []
    i = 0
    while i + 1 < len(lines):
        bayts.append(f"{lines[i]} // {lines[i+1]}")
        i += 2
    return bayts


def poem_structure(text: str) -> Dict[str, Any]:
    lines = split_poem_lines(text)
    return {
        "lines": lines,
        "num_lines": len(lines),
        "complete_bayts": len(lines) // 2,
        "has_odd_tail": len(lines) % 2 == 1,
        "odd_tail_line": lines[-1] if lines and len(lines) % 2 == 1 else "",
    }


def normalize_meter_label(base_meter, form, meter_label=None):
    if meter_label and str(meter_label).strip():
        return str(meter_label).strip()
    base_meter = str(base_meter or "").strip()
    form = str(form or "").strip()
    if not base_meter:
        return ""
    if not form or form == "تام":
        return base_meter
    return f"{form} {base_meter}".strip()


def row_to_poem_text(row):
    candidate_cols = [
        "poem_text",
        "text",
        "completion",
        "sft_completion",
        "poem verses",
        "poem_verses",
        "lines",
        "poem_lines",
    ]
    for col in candidate_cols:
        if col not in row:
            continue
        value = row[col]
        if isinstance(value, str) and value.strip():
            return normalize_whitespace(value)
        if isinstance(value, list) and value:
            return "\n".join(str(x).strip() for x in value if str(x).strip()).strip()
    return ""


def row_to_description(row):
    for col in ["enhanced_description", "description", "poem_description", "desc", "summary"]:
        if col in row and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def row_to_prompt(row):
    for col in ["prompt", "sft_prompt", "messages_prompt"]:
        if col in row and str(row[col]).strip():
            return str(row[col]).strip()
    return ""


def row_to_meter_fields(row):
    base_meter = row.get("base_meter", "")
    form = row.get("form", "")
    meter_label = normalize_meter_label(base_meter, form, row.get("meter_label", ""))
    return base_meter, form, meter_label


def extract_requested_lines_from_prompt(prompt: str):
    prompt = str(prompt or "")
    m = re.search(r"اكتب\s+(\d+)\s+شطراً", prompt)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def row_to_requested_lines(row):
    for col in ["requested_lines", "num_lines"]:
        try:
            if col in row and row[col] is not None and str(row[col]).strip():
                return int(row[col])
        except Exception:
            pass

    if "requested_bayts" in row:
        try:
            raw_bayts = row["requested_bayts"]
            if raw_bayts is not None and str(raw_bayts).strip():
                return int(raw_bayts) * 2
        except Exception:
            pass

    raw = row.get("sft_num_lines")
    try:
        if raw is not None and str(raw).strip():
            return int(raw)
    except Exception:
        pass

    prompt = row_to_prompt(row)
    requested_lines = extract_requested_lines_from_prompt(prompt)
    if requested_lines is not None:
        return requested_lines

    poem_text = row_to_poem_text(row)
    s = poem_structure(poem_text)
    return s["num_lines"]


def length_bucket(requested_bayts: int) -> str:
    requested_bayts = int(requested_bayts or 0)
    if requested_bayts <= 3:
        return "1-3"
    if requested_bayts <= 6:
        return "4-6"
    if requested_bayts <= 10:
        return "7-10"
    return "11-20"


def load_and_prepare_dataset(
    dataset_id,
    split="train",
    max_bayts=None,
    allowed_meters=None,
    sample_size=None,
    hf_token=None,
):
    ds = load_dataset(dataset_id, split=split, token=hf_token)

    def _map_row(row, idx):
        poem_text = row_to_poem_text(row)
        desc = row_to_description(row)
        prompt = row_to_prompt(row)
        base_meter, form, meter_label = row_to_meter_fields(row)
        struct = poem_structure(poem_text)
        requested_lines = row_to_requested_lines(row)
        requested_bayts = int(requested_lines) // 2 if int(requested_lines) > 0 else struct["complete_bayts"]
        source_index = row.get("source_index", idx)
        try:
            source_index = int(source_index)
        except Exception:
            source_index = idx
        return {
            "row_uid": str(row.get("row_uid", "")).strip(),
            "source_id": str(row.get("source_id", row.get("id", ""))).strip(),
            "source_split": str(row.get("source_split", split)).strip(),
            "source_index": source_index,
            "prompt": prompt,
            "description": desc,
            "poem_text": poem_text,
            "base_meter": base_meter,
            "form": form,
            "meter_label": meter_label,
            "requested_bayts": requested_bayts,
            "requested_lines": requested_lines,
            "has_odd_tail": struct["has_odd_tail"],
            "odd_tail_line": struct["odd_tail_line"],
            "length_bucket": str(row.get("length_bucket", "")).strip() or length_bucket(requested_bayts),
        }

    ds = ds.map(_map_row, with_indices=True)

    def _keep(row):
        if not row["prompt"]:
            return False
        if not row["description"]:
            return False
        if not row["meter_label"]:
            return False
        if row["requested_bayts"] <= 0:
            return False
        if max_bayts is not None and row["requested_bayts"] > int(max_bayts):
            return False
        if allowed_meters:
            return row["meter_label"] in allowed_meters or row["base_meter"] in allowed_meters
        return True

    ds = ds.filter(_keep)

    if sample_size is not None:
        sample_size = min(int(sample_size), len(ds))
        ds = ds.select(range(sample_size))

    return ds


def parse_allowed_meters_env():
    load_env()
    raw = os.getenv("PHASE1_ALLOWED_METERS", "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_dataset_id_for_phase():
    load_env()
    phase1 = os.getenv("PHASE1_DATASET_ID", "").strip()
    if phase1:
        return phase1
    return get_dataset_source_id_for_phase()


def resolve_sft_adapter_path(adapter_repo: str = "", hf_token: str = None, mode: str = ""):
    adapter_repo = str(adapter_repo or "").strip()
    hf_token = hf_token or os.getenv("HF_TOKEN", "").strip() or None
    mode = str(mode or os.getenv("SFT_ADAPTER_MODE", "full")).strip() or "full"

    if not adapter_repo:
        raise ValueError("SFT_ADAPTER_REPO is missing")

    adapter_root = Path(adapter_repo)
    if adapter_root.exists():
        if (adapter_root / "adapter_config.json").exists():
            return str(adapter_root), {"source": "local_root", "mode": mode}
        for candidate in [
            adapter_root / "adapters" / mode / "latest",
            adapter_root / "adapter",
        ]:
            if (candidate / "adapter_config.json").exists():
                return str(candidate), {"source": "local_nested", "mode": mode}

    manifest = None
    manifest_path = f"manifests/{mode}/latest.json"
    try:
        manifest_file = hf_hub_download(
            repo_id=adapter_repo,
            filename=manifest_path,
            token=hf_token,
        )
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = None

    snapshot_path = snapshot_download(repo_id=adapter_repo, token=hf_token)
    snapshot_root = Path(snapshot_path)

    candidates = []
    if manifest and manifest.get("adapter_latest_path_in_repo"):
        candidates.append(snapshot_root / manifest["adapter_latest_path_in_repo"])
    candidates.extend([
        snapshot_root,
        snapshot_root / "adapters" / mode / "latest",
        snapshot_root / "adapter",
    ])

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if (candidate / "adapter_config.json").exists():
            return str(candidate), {
                "source": "snapshot",
                "mode": mode,
                "snapshot_root": str(snapshot_root),
                "manifest_path": manifest_path if manifest else "",
                "adapter_repo": adapter_repo,
            }

    raise FileNotFoundError(
        f"Could not resolve adapter_config.json for adapter repo '{adapter_repo}' in mode '{mode}'"
    )


def score_count_adherence(requested_bayts: int, generated_poem: str) -> Dict[str, Any]:
    struct = poem_structure(generated_poem)
    gen_bayts = struct["complete_bayts"]
    requested = int(requested_bayts or 0)
    if requested <= 0:
        return {"score": 0.0, "generated_bayts": gen_bayts, "requested_bayts": requested}

    diff = abs(gen_bayts - requested)
    if diff == 0:
        score = 1.0
    else:
        score = max(0.0, 1.0 - diff / max(1, requested))
    return {
        "score": clamp01(score),
        "generated_bayts": gen_bayts,
        "requested_bayts": requested,
        "has_odd_tail": struct["has_odd_tail"],
        "num_lines": struct["num_lines"],
    }


def score_arabic_cleanliness(generated_poem: str) -> Dict[str, Any]:
    text = extract_text(generated_poem)
    contamination = contamination_stats(text)
    arabic_char_count = len(ARABIC_CHAR_RE.findall(text))
    latin_char_count = len(LATIN_CHAR_RE.findall(text))
    digit_char_count = len(DIGIT_CHAR_RE.findall(text))
    forbidden_artifact_count = len(FORBIDDEN_ARTIFACT_RE.findall(text))
    has_arabic = arabic_char_count > 0
    contains_latin = latin_char_count > 0
    contains_digits = digit_char_count > 0
    contains_forbidden_artifacts = forbidden_artifact_count > 0

    tokens = tokenize_arabic_words(text)
    lexicon = get_poem_lexicon()
    known_token_count = sum(1 for token in tokens if token in lexicon)
    arabic_token_count = len(tokens)
    lexical_known_ratio = known_token_count / arabic_token_count if arabic_token_count else 0.0
    content_tokens = sorted({token for token in tokens if len(token) >= 4})
    known_content_count = sum(1 for token in content_tokens if token in lexicon)
    content_known_ratio = known_content_count / len(content_tokens) if content_tokens else lexical_known_ratio
    lexical_plausibility_score = (
        clamp01(lexical_known_ratio / 0.85) * clamp01(content_known_ratio / 0.80)
        if arabic_token_count
        else 0.0
    )

    artifact_free_score = 1.0 if has_arabic and not contains_latin and not contains_digits and not contains_forbidden_artifacts else 0.0
    # Keep lexical plausibility as a diagnostic, but only use the artifact-free gate in the reward.
    score = artifact_free_score
    return {
        "score": score,
        **contamination,
        "has_arabic": has_arabic,
        "contains_latin": contains_latin,
        "contains_digits": contains_digits,
        "contains_forbidden_artifacts": contains_forbidden_artifacts,
        "arabic_char_count": arabic_char_count,
        "latin_char_count": latin_char_count,
        "digit_char_count": digit_char_count,
        "forbidden_artifact_count": forbidden_artifact_count,
        "arabic_token_count": arabic_token_count,
        "known_token_count": known_token_count,
        "lexical_known_ratio": lexical_known_ratio,
        "content_token_count": len(content_tokens),
        "known_content_count": known_content_count,
        "content_known_ratio": content_known_ratio,
        "lexical_plausibility_score": lexical_plausibility_score,
        "artifact_free_score": artifact_free_score,
    }


def score_friend_reward_floors(
    generated_poem: str,
    generated_bayts: int,
    count_adherence_score: float,
    near_duplicate_score: float,
    distinct_2_score: float,
    opening_diversity_score: float,
) -> Dict[str, Any]:
    text = extract_text(generated_poem)
    contamination = contamination_stats(text)
    contamination_ratio = float(contamination["contamination_ratio"])
    stripped_text_length = int(contamination["stripped_text_length"])
    generated_bayts = int(generated_bayts or 0)

    hard_gate_reasons = []
    if contamination_ratio > 0.40:
        hard_gate_reasons.append("contamination_gt_0.40")
    if generated_bayts <= 0:
        hard_gate_reasons.append("no_complete_bayt")
    if stripped_text_length < 10:
        hard_gate_reasons.append("too_short")
    hard_gate_score = 0.0 if hard_gate_reasons else 1.0

    if contamination_ratio > 0.35:
        arabic_floor_score = 0.50
    elif contamination_ratio > 0.10:
        arabic_floor_score = 0.65
    elif contamination_ratio > 0.02:
        arabic_floor_score = 0.85
    else:
        arabic_floor_score = 1.0

    count_floor_score = 0.70 + 0.30 * clamp01(count_adherence_score)
    repeat_soft_score = clamp01(
        0.50 * clamp01(near_duplicate_score)
        + 0.30 * clamp01(distinct_2_score)
        + 0.20 * clamp01(opening_diversity_score)
    )
    repeat_floor_score = 0.70 + 0.30 * repeat_soft_score

    return {
        **contamination,
        "hard_gate_score": float(hard_gate_score),
        "hard_gate_blocked": bool(hard_gate_score == 0.0),
        "hard_gate_reason": ",".join(hard_gate_reasons),
        "arabic_floor_score": float(clamp01(arabic_floor_score)),
        "count_floor_score": float(clamp01(count_floor_score)),
        "repeat_soft_score": float(clamp01(repeat_soft_score)),
        "repeat_floor_score": float(clamp01(repeat_floor_score)),
    }


def score_minimal_hard_gate(
    generated_poem: str,
    *,
    generated_bayts: int | None = None,
    clean_out: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    text = extract_text(generated_poem)
    clean = clean_out or score_arabic_cleanliness(text)
    contamination = contamination_stats(text)
    struct = poem_structure(text)
    complete_bayts = int(generated_bayts if generated_bayts is not None else struct["complete_bayts"] or 0)

    reasons = []
    if not text.strip():
        reasons.append("empty_output")
    if not bool(clean.get("has_arabic", False)):
        reasons.append("no_arabic")
    if bool(clean.get("contains_latin", False)):
        reasons.append("latin_leakage")
    if bool(clean.get("contains_digits", False)):
        reasons.append("digit_leakage")
    if bool(clean.get("contains_forbidden_artifacts", False)):
        reasons.append("artifact_leakage")
    if complete_bayts <= 0:
        reasons.append("no_complete_bayt")

    score = 0.0 if reasons else 1.0
    return {
        **contamination,
        "hard_gate_score": float(score),
        "hard_gate_blocked": bool(score == 0.0),
        "hard_gate_reason": ",".join(reasons),
        "complete_bayts": complete_bayts,
        "num_lines": int(struct["num_lines"]),
        "has_odd_tail": bool(struct["has_odd_tail"]),
        "text_is_empty": not text.strip(),
        "contains_latin": bool(clean.get("contains_latin", False)),
        "contains_digits": bool(clean.get("contains_digits", False)),
        "contains_forbidden_artifacts": bool(clean.get("contains_forbidden_artifacts", False)),
        "has_arabic": bool(clean.get("has_arabic", False)),
    }


def score_weighted_reward_train(
    *,
    meter_score: float,
    count_adherence_score: float,
    judge_quality_score: float | None = None,
    judge_meaning_fit_score: float | None = None,
    judge_naturalness_score: float | None = None,
    repeat_soft_score: float,
    hard_gate_score: float,
) -> Dict[str, float]:
    meter_clamped = clamp01(meter_score)
    count_clamped = clamp01(count_adherence_score)
    repeat_clamped = clamp01(repeat_soft_score)
    if judge_meaning_fit_score is not None and judge_naturalness_score is not None:
        meaning_clamped = clamp01(judge_meaning_fit_score)
        naturalness_clamped = clamp01(judge_naturalness_score)
        meter_meaning_core = meter_clamped * meaning_clamped
        naturalness_bonus = 0.20 * naturalness_clamped
        count_bonus = 0.20 * count_clamped * meaning_clamped
        repeat_bonus = 0.15 * repeat_clamped * naturalness_clamped
        weighted_sum = (
            0.45 * meter_meaning_core
            + naturalness_bonus
            + count_bonus
            + repeat_bonus
        )
        total_score = clamp01(hard_gate_score) * weighted_sum
        return {
            "meter_meaning_core": float(meter_meaning_core),
            "naturalness_bonus": float(naturalness_bonus),
            "count_bonus": float(count_bonus),
            "repeat_bonus": float(repeat_bonus),
            "weighted_sum": float(weighted_sum),
            "total_score": float(total_score),
        }

    judge_clamped = clamp01(judge_quality_score or 0.0)
    meter_judge_core = meter_clamped * judge_clamped
    judge_gated_aux = judge_clamped * (
        0.20 * count_clamped
        + 0.15 * repeat_clamped
    )
    weighted_sum = (
        0.65 * meter_judge_core
        + judge_gated_aux
    )
    total_score = clamp01(hard_gate_score) * weighted_sum
    return {
        "meter_judge_core": float(meter_judge_core),
        "judge_gated_aux": float(judge_gated_aux),
        "weighted_sum": float(weighted_sum),
        "total_score": float(total_score),
    }


def score_repeat_soft_signal(
    *,
    exact_repeat_score: float,
    near_duplicate_score: float,
    opening_diversity_score: float,
    distinct_2_score: float,
) -> float:
    return float(
        clamp01(
            0.35 * clamp01(exact_repeat_score)
            + 0.35 * clamp01(near_duplicate_score)
            + 0.15 * clamp01(opening_diversity_score)
            + 0.15 * clamp01(distinct_2_score)
        )
    )


def score_repetition_penalty(generated_poem: str) -> Dict[str, Any]:
    lines = [normalize_whitespace(line) for line in split_poem_lines(extract_text(generated_poem))]
    lines = [line for line in lines if line]
    line_count = len(lines)
    if line_count <= 1:
        return {
            "score": 1.0,
            "line_count": line_count,
            "has_repeated_line": False,
            "max_line_repeat_count": 1 if line_count == 1 else 0,
            "longest_consecutive_repeat_run": 1 if line_count == 1 else 0,
            "dominant_line_fraction": 1.0 if line_count == 1 else 0.0,
            "duplicate_extra_lines": 0,
            "dominant_repeated_line": lines[0] if lines else "",
            "exact_repeat_score": 1.0,
            "near_duplicate_score": 1.0,
            "opening_diversity_score": 1.0,
            "distinct_2_score": 1.0,
            "distinct_1": 1.0 if lines else 0.0,
            "distinct_2": 1.0 if lines else 0.0,
            "distinct_3": 1.0 if lines else 0.0,
            "opening_dominance_fraction": 1.0 if line_count == 1 else 0.0,
            "line_pair_similarity_mean": 0.0,
            "line_pair_similarity_max": 0.0,
            "sequence_similarity_mean": 0.0,
            "sequence_similarity_max": 0.0,
            "near_duplicate_pair_fraction": 0.0,
        }

    counts = Counter(lines)
    dominant_repeated_line, max_line_repeat_count = counts.most_common(1)[0]
    duplicate_extra_lines = sum(count - 1 for count in counts.values() if count >= 2)

    longest_consecutive_repeat_run = 1
    current_run = 1
    for previous, current in zip(lines, lines[1:]):
        if current == previous:
            current_run += 1
        else:
            current_run = 1
        longest_consecutive_repeat_run = max(longest_consecutive_repeat_run, current_run)

    dominant_line_fraction = max_line_repeat_count / line_count
    exact_repeat_score = 1.0 - (max_line_repeat_count - 1) / max(1, line_count - 1)

    line_tokens = [tokenize_arabic_words(line) for line in lines]
    openings = [opening_signature(tokens, size=2) for tokens in line_tokens if tokens]
    opening_counts = Counter(openings)
    opening_dominance_fraction = (max(opening_counts.values()) / len(openings)) if openings else 0.0
    opening_diversity_score = clamp01(1.0 - max(0.0, opening_dominance_fraction - (1.0 / 3.0)) / (2.0 / 3.0))

    pair_similarities = []
    sequence_similarities = []
    for i in range(len(line_tokens)):
        for j in range(i + 1, len(line_tokens)):
            pair_similarities.append(jaccard_similarity(line_tokens[i], line_tokens[j]))
            sequence_similarities.append(sequence_similarity(" ".join(line_tokens[i]), " ".join(line_tokens[j])))
    line_pair_similarity_mean = mean(pair_similarities)
    line_pair_similarity_max = max(pair_similarities) if pair_similarities else 0.0
    combined_pair_similarities = [max(jaccard, seq) for jaccard, seq in zip(pair_similarities, sequence_similarities)]
    near_duplicate_pair_fraction = (
        sum(1.0 for sim in combined_pair_similarities if sim >= 0.70) / len(combined_pair_similarities) if combined_pair_similarities else 0.0
    )
    similarity_penalties = [max(0.0, (sim - 0.45) / 0.55) for sim in combined_pair_similarities]
    near_duplicate_score = clamp01(1.0 - mean(similarity_penalties))

    all_tokens = [token for tokens in line_tokens for token in tokens]
    distinct_1 = distinct_n(all_tokens, 1)
    distinct_2 = distinct_n(all_tokens, 2)
    distinct_3 = distinct_n(all_tokens, 3)
    distinct_2_score = clamp01((distinct_2 - 0.35) / 0.65)

    repeat_components = [
        clamp01(exact_repeat_score),
        near_duplicate_score,
        opening_diversity_score,
        distinct_2_score,
    ]
    score = 1.0
    for component in repeat_components:
        score *= clamp01(component)

    return {
        "score": clamp01(score),
        "line_count": line_count,
        "has_repeated_line": max_line_repeat_count >= 2,
        "max_line_repeat_count": int(max_line_repeat_count),
        "longest_consecutive_repeat_run": int(longest_consecutive_repeat_run),
        "dominant_line_fraction": float(dominant_line_fraction),
        "duplicate_extra_lines": int(duplicate_extra_lines),
        "dominant_repeated_line": dominant_repeated_line,
        "exact_repeat_score": clamp01(exact_repeat_score),
        "near_duplicate_score": near_duplicate_score,
        "opening_diversity_score": opening_diversity_score,
        "distinct_2_score": distinct_2_score,
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "distinct_3": float(distinct_3),
        "opening_dominance_fraction": float(opening_dominance_fraction),
        "line_pair_similarity_mean": float(line_pair_similarity_mean),
        "line_pair_similarity_max": float(line_pair_similarity_max),
        "sequence_similarity_mean": float(mean(sequence_similarities)),
        "sequence_similarity_max": float(max(sequence_similarities) if sequence_similarities else 0.0),
        "near_duplicate_pair_fraction": float(near_duplicate_pair_fraction),
    }


def extract_content_from_chat_response(data):
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content)


def extract_first_json_any(text):
    text = str(text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def extract_first_json_object(text):
    parsed = extract_first_json_any(text)
    if isinstance(parsed, dict):
        return parsed
    return None


class JudgeClient:
    def __init__(
        self,
        cache_dir: str,
        cache_namespace: str = "judge",
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ):
        load_env()
        self.base_url = os.getenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model = model or os.getenv("JUDGE_MODEL", "qwen/qwen3.5-35b-a3b").strip()
        self.timeout = float(timeout or os.getenv("JUDGE_TIMEOUT_SECONDS", "120"))
        self.max_retries = int(max_retries or os.getenv("JUDGE_MAX_RETRIES", "3"))
        self.max_tokens = int(max_tokens or os.getenv("JUDGE_MAX_TOKENS", "256"))
        self.referrer = os.getenv("OPENROUTER_REFERRER", "").strip()
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "Shaer-GRPO").strip()
        self.json_mode = json_mode
        self.cache_dir = ensure_dir(Path(cache_dir) / cache_namespace)

    def _cache_key(self, payload):
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _headers(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referrer:
            headers["HTTP-Referer"] = self.referrer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def _build_payload(self, system_prompt, user_prompt):
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
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _call_once(self, payload):
        timeout_obj = httpx.Timeout(self.timeout, connect=min(20.0, self.timeout))
        with httpx.Client(timeout=timeout_obj) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    def judge(self, system_prompt, user_prompt):
        payload = self._build_payload(system_prompt, user_prompt)
        key = self._cache_key(payload)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["cache_key"] = key
            return cached, True

        last_error = None
        t0 = time.time()
        raw_response = None
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self._call_once(payload)
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                time.sleep(2.0 * (attempt + 1))

        latency_sec = time.time() - t0
        out = {
            "response": raw_response,
            "latency_sec": latency_sec,
            "error": last_error,
            "cache_key": key,
            "model": self.model,
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return out, False


def extract_judge_json(response_obj, score_key="score"):
    response = response_obj.get("response") or {}
    content = extract_content_from_chat_response(response)
    parsed = extract_first_json_object(content)
    usage = response.get("usage", {}) or {}
    ctd = usage.get("completion_tokens_details", {}) or {}
    msg = ((response.get("choices") or [{}])[0]).get("message", {}) or {}

    if parsed is None:
        return {
            score_key: 0.0,
            "notes": "json_not_found",
            "raw": content,
            "error": response_obj.get("error") or "json_not_found",
            "mode_respected": (
                ctd.get("reasoning_tokens", None) == 0
                and not bool(msg.get("reasoning"))
                and not bool(msg.get("reasoning_details"))
            ) if response else None,
            "reasoning_tokens": ctd.get("reasoning_tokens", None),
        }

    out = dict(parsed)
    out["raw"] = content
    out["error"] = response_obj.get("error")
    out["mode_respected"] = (
        ctd.get("reasoning_tokens", None) == 0
        and not bool(msg.get("reasoning"))
        and not bool(msg.get("reasoning_details"))
    ) if response else None
    out["reasoning_tokens"] = ctd.get("reasoning_tokens", None)
    return out


def iter_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out
