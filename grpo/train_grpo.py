import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from rewards.common import (
    append_jsonl,
    ensure_dir,
    extract_text,
    length_bucket,
    load_and_prepare_dataset,
    mask_env,
    poem_structure,
    parse_allowed_meters_env,
    row_to_description,
    row_to_meter_fields,
    row_to_prompt,
    row_to_requested_lines,
    resolve_sft_adapter_path,
    save_json,
    score_arabic_cleanliness,
    score_count_adherence,
    score_minimal_hard_gate,
    score_repeat_soft_signal,
    score_repetition_penalty,
    score_weighted_reward_train,
    setup_logger,
    sha256_text,
    write_csv_rows,
)
from rewards.judge_quality import batch_score_judge_quality
from rewards.meter import score_meter_poem


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT.parent / ".env"
CHECKPOINT_MANIFEST_NAME = "grpo_resume_manifest.json"
ACTIVE_REWARD_NAMES = [
    "meter",
    "count_adherence",
    "arabic_clean",
    "hard_gate",
    "repeat_soft",
    "arabic_floor",
    "count_floor",
    "lexical_plausibility",
    "repeat_penalty",
    "repeat_floor",
    "near_duplicate_penalty",
    "opening_diversity",
    "distinct_2",
    "judge_quality",
    "judge_meaning_fit",
    "judge_naturalness",
    "total_composite",
    "exact_count_bonus",
    "meter_count_clean",
]


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_cfg():
    with open(ROOT / "grpo_config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(x) for x in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_utc_timestamp(value: str) -> float:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").timestamp()
    except Exception:
        return 0.0


def resolve_generations_repo(output_repo: str) -> str:
    explicit = os.getenv("GRPO_GENERATIONS_REPO", "").strip()
    if explicit:
        return explicit
    if not output_repo or "/" not in output_repo:
        return ""
    owner, name = output_repo.split("/", 1)
    return f"{owner}/{name}-generations"


def row_sort_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)


def stable_hash(seed: int, *parts: Any) -> int:
    blob = "|".join(str(part) for part in (seed, *parts))
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16], 16)


def resolve_base_model_id(cfg) -> str:
    configured = str(cfg["run"].get("base_model_id", "")).strip()
    return os.getenv("BASE_MODEL_ID", "").strip() or configured


def resolve_start_adapter_repo(cfg) -> str:
    configured = str(cfg["run"].get("start_adapter_repo", "")).strip()
    return os.getenv("SFT_ADAPTER_REPO", "").strip() or configured


def resolve_start_adapter_mode(cfg) -> str:
    configured = str(cfg["run"].get("start_adapter_mode", "fresh_sft/train")).strip() or "fresh_sft/train"
    return os.getenv("SFT_ADAPTER_MODE", "").strip() or configured


def read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported manifest format: {path}")


def build_manifest_index(rows: list[dict[str, Any]], key: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    ordered_keys: list[str] = []
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        ordered_keys.append(value)
        index[value] = dict(row)
    return ordered_keys, index


def ordered_rows_from_manifest(prepared_ds, ordered_keys: list[str], manifest_index: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    prepared_index: dict[str, dict[str, Any]] = {}
    for row in prepared_ds:
        value = str(row.get(key_name, "")).strip()
        if value:
            prepared_index[value] = dict(row)

    missing = [value for value in ordered_keys if value not in prepared_index]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Manifest rows missing from dataset for key={key_name}: {preview}")

    selected_rows: list[dict[str, Any]] = []
    for value in ordered_keys:
        row = dict(prepared_index[value])
        row.update({f"manifest_{k}": v for k, v in manifest_index[value].items()})
        selected_rows.append(row)
    return selected_rows


def sort_key_for_bucket(value: str) -> tuple[int, str]:
    order = ["1-3", "4-6", "7-10", "11-20", "short_le_8", "full_gt_8"]
    if value in order:
        return (order.index(value), value)
    return (len(order), value)


def choose_balanced_rows(rows: list[dict[str, Any]], *, per_meter_per_bucket: int, seed: int) -> list[dict[str, Any]]:
    if per_meter_per_bucket <= 0:
        return list(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("base_meter", "")), str(row.get("length_bucket", "")))
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for (meter, bucket), bucket_rows in sorted(grouped.items(), key=lambda item: (item[0][0], sort_key_for_bucket(item[0][1]))):
        ordered = sorted(
            bucket_rows,
            key=lambda row: (
                stable_hash(seed, meter, bucket, row.get("row_uid", row.get("source_id", row.get("source_index", "")))),
                int(row.get("source_index", 0) or 0),
            ),
        )
        selected.extend(ordered[: min(per_meter_per_bucket, len(ordered))])
    selected.sort(
        key=lambda row: (
            str(row.get("base_meter", "")),
            sort_key_for_bucket(str(row.get("length_bucket", ""))),
            int(row.get("source_index", 0) or 0),
        )
    )
    return selected


def choose_sanity_subset(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    ordered = sorted(
        rows,
        key=lambda row: (
            stable_hash(seed, row.get("base_meter", ""), row.get("length_bucket", ""), row.get("row_uid", row.get("source_id", row.get("source_index", "")))),
            str(row.get("base_meter", "")),
            int(row.get("source_index", 0) or 0),
        ),
    )
    return ordered[:limit]


def prepare_runtime_row_from_raw(row: dict[str, Any], idx: int, dataset_split: str) -> dict[str, Any]:
    prompt = row_to_prompt(row)
    description = row_to_description(row)
    base_meter, form, meter_label = row_to_meter_fields(row)
    requested_lines = row_to_requested_lines(row)
    requested_bayts_value = row.get("requested_bayts", "")
    try:
        requested_bayts = int(requested_bayts_value)
    except Exception:
        requested_bayts = int(requested_lines) // 2 if int(requested_lines or 0) > 0 else 0
    source_index = row.get("source_index", idx)
    try:
        source_index = int(source_index)
    except Exception:
        source_index = idx
    return {
        "row_uid": str(row.get("row_uid", "")).strip(),
        "source_id": str(row.get("source_id", row.get("id", ""))).strip(),
        "source_split": str(row.get("source_split", dataset_split)).strip(),
        "source_index": source_index,
        "prompt": prompt,
        "description": description,
        "base_meter": base_meter,
        "form": form,
        "meter_label": meter_label,
        "requested_bayts": requested_bayts,
        "requested_lines": requested_lines,
        "length_bucket": str(row.get("length_bucket", "")).strip() or length_bucket(requested_bayts),
        "difficulty": str(row.get("difficulty", "")).strip(),
    }


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        ensure_dir(path.parent)

    def write(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.path, payload)


class GenerationRecorder:
    def __init__(self, path: Path, reward_order: list[str], reward_weights: dict[str, float], run_meta: dict[str, Any] | None = None):
        self.path = path
        self.reward_order = reward_order
        self.reward_weights = reward_weights
        self.run_meta = run_meta or {}
        self.first_reward = reward_order[0]
        self.last_reward = reward_order[-1]
        self.batch_index = 0
        self.current_batch: dict[str, Any] | None = None
        ensure_dir(path.parent)

    def _as_list(self, value, target_len: int):
        if isinstance(value, list):
            return value
        return [value] * target_len

    def _candidate_rows(self, prompts, completions, kwargs):
        trainer_state = kwargs.get("trainer_state")
        size = len(completions)
        source_index = self._as_list(kwargs.get("source_index", []), size)
        description = self._as_list(kwargs.get("description", []), size)
        meter_label = self._as_list(kwargs.get("meter_label", []), size)
        base_meter = self._as_list(kwargs.get("base_meter", []), size)
        requested_bayts = self._as_list(kwargs.get("requested_bayts", []), size)
        requested_lines = self._as_list(kwargs.get("requested_lines", []), size)
        dataset_split = self._as_list(kwargs.get("dataset_split", []), size)
        length_bucket = self._as_list(kwargs.get("length_bucket", []), size)
        completion_ids = self._as_list(kwargs.get("completion_ids", []), size)

        prompt_counters: dict[str, int] = {}
        rows = []
        for i, completion in enumerate(completions):
            prompt_text = extract_text(prompts[i] if prompts and i < len(prompts) else "")
            completion_text = extract_text(completion)
            prompt_hash = sha256_text(prompt_text)
            completion_hash = sha256_text(completion_text)
            prompt_candidate_index = prompt_counters.get(prompt_hash, 0)
            prompt_counters[prompt_hash] = prompt_candidate_index + 1
            mode = "train" if str(dataset_split[i] if i < len(dataset_split) else "") == "train" else "eval"
            trainer_global_step = int(getattr(trainer_state, "global_step", 0) or 0)
            logical_step = trainer_global_step + (1 if mode == "train" else 0)
            source_value = json_safe(source_index[i] if i < len(source_index) else None)
            prompt_group_id = sha256_text(
                json.dumps(
                    {
                        "chain_id": self.run_meta.get("chain_id", ""),
                        "run_id": self.run_meta.get("run_id", ""),
                        "mode": mode,
                        "logical_step": logical_step,
                        "source_index": source_value,
                        "prompt_hash": prompt_hash,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            rows.append(
                {
                    "timestamp_utc": utc_now_iso(),
                    "run_id": str(self.run_meta.get("run_id", "")),
                    "run_dir": str(self.run_meta.get("run_dir", "")),
                    "chain_id": str(self.run_meta.get("chain_id", "")),
                    "root_run_id": str(self.run_meta.get("root_run_id", "")),
                    "parent_run_id": str(self.run_meta.get("parent_run_id", "")),
                    "run_sequence_index": int(self.run_meta.get("run_sequence_index", 0) or 0),
                    "batch_index": self.batch_index,
                    "trainer_global_step_raw": trainer_global_step,
                    "global_step": trainer_global_step,
                    "logical_step": logical_step,
                    "epoch": float(getattr(trainer_state, "epoch", 0.0) or 0.0),
                    "dataset_split": str(dataset_split[i] if i < len(dataset_split) else ""),
                    "mode": mode,
                    "mode_order": 0 if mode == "train" else 1,
                    "source_index": source_value,
                    "requested_bayts": json_safe(requested_bayts[i] if i < len(requested_bayts) else None),
                    "requested_lines": json_safe(requested_lines[i] if i < len(requested_lines) else None),
                    "length_bucket": str(length_bucket[i] if i < len(length_bucket) else ""),
                    "meter_label": str(meter_label[i] if i < len(meter_label) else ""),
                    "base_meter": str(base_meter[i] if i < len(base_meter) else ""),
                    "prompt_hash": prompt_hash,
                    "prompt_group_id": prompt_group_id,
                    "prompt_preview": prompt_text[:400],
                    "prompt_candidate_index": prompt_candidate_index,
                    "candidate_index": prompt_candidate_index,
                    "completion_hash": completion_hash,
                    "completion_ids_hash": sha256_text(json.dumps(json_safe(completion_ids[i] if i < len(completion_ids) else []), ensure_ascii=False)),
                    "completion_text": completion_text,
                    "description_preview": str(description[i] if i < len(description) else "")[:400],
                }
            )
        return rows

    def _finalize_current_batch(self, incomplete: bool = False):
        if self.current_batch is None:
            return
        seen = set(self.current_batch["seen_rewards"])
        for row in self.current_batch["rows"]:
            total = 0.0
            for reward_name, weight in self.reward_weights.items():
                value = row.get(f"reward_{reward_name}")
                if value is None:
                    continue
                total += float(weight) * float(value)
            row["reward_total"] = total
            row["reward_total_mean"] = total
            row["incomplete_reward_batch"] = incomplete or (seen != set(self.reward_order))
            append_jsonl(self.path, row)
        self.current_batch = None

    def record_reward_batch(self, reward_name: str, completions, prompts, scores, extras, kwargs):
        reward_name = str(reward_name)
        prompts = prompts or []
        completions = completions or []
        scores = scores or []
        extras = extras or [{} for _ in completions]

        completion_hashes = [sha256_text(extract_text(c)) for c in completions]
        prompt_hashes = [sha256_text(extract_text(prompts[i] if prompts and i < len(prompts) else "")) for i in range(len(completions))]

        if reward_name == self.first_reward or self.current_batch is None:
            if self.current_batch is not None:
                self._finalize_current_batch(incomplete=True)
            self.batch_index += 1
            self.current_batch = {
                "completion_hashes": completion_hashes,
                "prompt_hashes": prompt_hashes,
                "rows": self._candidate_rows(prompts, completions, kwargs),
                "seen_rewards": [],
            }
        else:
            if (
                self.current_batch["completion_hashes"] != completion_hashes
                or self.current_batch["prompt_hashes"] != prompt_hashes
            ):
                self._finalize_current_batch(incomplete=True)
                self.batch_index += 1
                self.current_batch = {
                    "completion_hashes": completion_hashes,
                    "prompt_hashes": prompt_hashes,
                    "rows": self._candidate_rows(prompts, completions, kwargs),
                    "seen_rewards": [],
                }

        for i, row in enumerate(self.current_batch["rows"]):
            row[f"reward_{reward_name}"] = float(scores[i]) if i < len(scores) else None
            row[f"{reward_name}_extra"] = json_safe(extras[i] if i < len(extras) else {})
        self.current_batch["seen_rewards"].append(reward_name)

        if reward_name == self.last_reward:
            self._finalize_current_batch(incomplete=False)

    def close(self):
        self._finalize_current_batch(incomplete=True)


class StructuredMetricsCallback(TrainerCallback):
    def __init__(self, metrics_path: Path, metrics_csv_path: Path, logger):
        self.metrics_writer = JsonlWriter(metrics_path)
        self.metrics_csv_path = metrics_csv_path
        self.logger = logger
        self.rows: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logs = logs or {}
        mode = "eval" if any(str(k).startswith("eval_") for k in logs) else "train"
        row = {
            "timestamp_utc": utc_now_iso(),
            "mode": mode,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        for key, value in logs.items():
            row[str(key)] = json_safe(value)
        self.rows.append(row)
        self.metrics_writer.write(row)
        write_csv_rows(self.rows, self.metrics_csv_path)
        self.logger.info("metrics_logged mode=%s step=%s", mode, row["global_step"])


class CheckpointEventCallback(TrainerCallback):
    def __init__(self, run_dir: Path, events_path: Path, fingerprint: str, runtime_summary: dict[str, Any], output_repo: str):
        self.run_dir = run_dir
        self.events_writer = JsonlWriter(events_path)
        self.fingerprint = fingerprint
        self.runtime_summary = runtime_summary
        self.output_repo = output_repo

    def _manifest_payload(self, step: int):
        payload = dict(self.runtime_summary)
        payload.update(
            {
                "timestamp_utc": utc_now_iso(),
                "config_fingerprint": self.fingerprint,
                "global_step": int(step),
                "output_repo": self.output_repo,
            }
        )
        return payload

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        step = int(state.global_step)
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step}"
        if checkpoint_dir.exists():
            save_json(self._manifest_payload(step), checkpoint_dir / CHECKPOINT_MANIFEST_NAME)
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "checkpoint_saved",
                "global_step": step,
                "local_checkpoint_dir": str(checkpoint_dir),
                "hub_model_id": self.output_repo or "",
                "expected_hub_prefix": "last-checkpoint" if self.output_repo else "",
            }
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "evaluation_completed",
                "global_step": int(state.global_step),
                "metrics": json_safe(metrics or {}),
            }
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "train_end",
                "global_step": int(state.global_step),
                "best_model_checkpoint": json_safe(getattr(state, "best_model_checkpoint", None)),
            }
        )


def add_metric_aliases(logs: dict[str, Any]) -> dict[str, Any]:
    out = dict(logs)
    if "reward" in out and "reward_total_mean" not in out:
        out["reward_total_mean"] = out["reward"]
    if "eval_reward" in out and "eval_reward_total_mean" not in out:
        out["eval_reward_total_mean"] = out["eval_reward"]

    for reward_name in ACTIVE_REWARD_NAMES:
        train_mean = f"rewards/{reward_name}/mean"
        train_std = f"rewards/{reward_name}/std"
        eval_mean = f"eval_rewards/{reward_name}/mean"
        eval_std = f"eval_rewards/{reward_name}/std"
        if train_mean in out:
            out.setdefault(f"reward_{reward_name}_mean", out[train_mean])
        if train_std in out:
            out.setdefault(f"reward_{reward_name}_std", out[train_std])
        if eval_mean in out:
            out.setdefault(f"eval_reward_{reward_name}_mean", out[eval_mean])
        if eval_std in out:
            out.setdefault(f"eval_reward_{reward_name}_std", out[eval_std])
    return out


class InstrumentedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, eval_num_generations: int | None = None, **kwargs):
        self.eval_num_generations = eval_num_generations
        self.last_logged_metrics: dict[str, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        merged = add_metric_aliases({**logs, **metrics})
        self.last_logged_metrics[mode] = dict(merged)
        Trainer.log(self, merged, start_time)
        self._metrics[mode].clear()

    def evaluate(self, *args, **kwargs):
        original_num_generations = self.num_generations
        original_args_num_generations = self.args.num_generations
        if self.eval_num_generations is not None:
            self.num_generations = int(self.eval_num_generations)
            self.args.num_generations = int(self.eval_num_generations)
        try:
            metrics = super().evaluate(*args, **kwargs)
        finally:
            self.num_generations = original_num_generations
            self.args.num_generations = original_args_num_generations
        metrics = dict(metrics or {})
        metrics.update(self.last_logged_metrics.get("eval", {}))
        return add_metric_aliases(metrics)


def create_config_fingerprint(cfg, dataset_id: str) -> str:
    payload = {
        "dataset_id": dataset_id,
        "base_model_id": resolve_base_model_id(cfg),
        "sft_adapter_repo": resolve_start_adapter_repo(cfg),
        "sft_adapter_mode": resolve_start_adapter_mode(cfg),
        "active_rewards": cfg["phase1"]["active_rewards"],
        "reward_weights": cfg["phase1"]["reward_weights"],
        "curated_manifest_path": str(cfg["dataset"].get("train_manifest_path", "")),
        "hard_diagnostic_manifest_path": str(cfg["dataset"].get("hard_diagnostic_manifest_path", "")),
        "generation": cfg["generation"],
        "trainer": {
            "learning_rate": cfg["trainer"]["learning_rate"],
            "gradient_accumulation_steps": cfg["trainer"]["gradient_accumulation_steps"],
            "max_completion_length": cfg["generation"]["max_completion_length"],
            "use_vllm": cfg["generation"]["use_vllm"],
            "vllm_mode": cfg["generation"]["vllm_mode"],
        },
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_run_dir(cfg, mode: str) -> Path:
    env_override = os.getenv("GRPO_RUN_DIR", "").strip()
    if env_override:
        run_dir = Path(env_override).expanduser().resolve()
        ensure_dir(run_dir)
        return run_dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    root = cfg["sanity_check"]["output_root"] if mode == "sanity" else cfg["run"]["output_root"]
    prefix = "sanity" if mode == "sanity" else cfg["run"]["run_name_prefix"]
    run_dir = Path(root) / f"{prefix}_{ts}"
    ensure_dir(run_dir)
    return run_dir


def build_model_and_tokenizer(cfg):
    load_dotenv(ENV_PATH, override=False)
    base_model_id = resolve_base_model_id(cfg)
    adapter_repo = resolve_start_adapter_repo(cfg)
    adapter_mode = resolve_start_adapter_mode(cfg)
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    adapter_path, adapter_meta = resolve_sft_adapter_path(adapter_repo=adapter_repo, hf_token=hf_token, mode=adapter_mode)
    use_vllm = bool(cfg["generation"]["use_vllm"])
    load_in_4bit_requested = bool(cfg["model"]["load_in_4bit"])
    load_in_4bit_effective = load_in_4bit_requested and not use_vllm

    bnb_config = None
    if load_in_4bit_effective:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["model"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["model"]["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False

    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    model = PeftModel.from_pretrained(model, adapter_path, token=hf_token, is_trainable=True)
    total_params = sum(int(param.numel()) for param in model.parameters())
    trainable_params = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return model, tokenizer, {
        "adapter_path": adapter_path,
        "load_in_4bit_requested": load_in_4bit_requested,
        "load_in_4bit_effective": load_in_4bit_effective,
        "vllm_colocate_runtime": use_vllm,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": float(trainable_params / total_params) if total_params else 0.0,
        **adapter_meta,
    }


def build_dataset_rows(prepared_rows: list[dict[str, Any]], dataset_split: str):
    rows = []
    for row in prepared_rows:
        rows.append(
            {
                "prompt": row["prompt"],
                "description": row["description"],
                "meter_label": row["meter_label"],
                "base_meter": row["base_meter"],
                "requested_bayts": row["requested_bayts"],
                "requested_lines": row["requested_lines"],
                "length_bucket": row["length_bucket"],
                "source_index": row["source_index"],
                "row_uid": row.get("row_uid", ""),
                "source_id": row.get("source_id", ""),
                "source_split": row.get("source_split", dataset_split),
                "difficulty": row.get("difficulty", ""),
                "selection_pool": row.get("manifest_selection_pool", ""),
                "dataset_split": dataset_split,
            }
        )
    return rows


def build_datasets(cfg, mode: str):
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    max_bayts = os.getenv("PHASE1_MAX_BAYTS", "").strip() or None
    allowed_meters = parse_allowed_meters_env() or None

    dataset_cfg = cfg["dataset"]
    train_dataset_id = str(os.getenv("GRPO_TRAIN_DATASET_ID", dataset_cfg["train_dataset_id"])).strip()
    source_dataset_id = str(os.getenv("GRPO_SOURCE_DATASET_ID", dataset_cfg["source_dataset_id"])).strip()
    train_manifest_path = Path(os.getenv("GRPO_CURATED_MANIFEST_PATH", dataset_cfg["train_manifest_path"])).expanduser().resolve()
    hard_manifest_path = Path(os.getenv("GRPO_HARD_DIAGNOSTIC_MANIFEST_PATH", dataset_cfg["hard_diagnostic_manifest_path"])).expanduser().resolve()
    eval_bucket_quota = int(dataset_cfg["eval_bank_per_meter_per_bucket"])
    test_bucket_quota = int(dataset_cfg["test_bank_per_meter_per_bucket"])
    eval_allowed_length_buckets = {
        str(value).strip()
        for value in (dataset_cfg.get("eval_allowed_length_buckets") or [])
        if str(value).strip()
    }
    eval_drop_meters = {
        str(value).strip()
        for value in (dataset_cfg.get("eval_drop_meters") or [])
        if str(value).strip()
    }

    def filter_eval_rows(rows):
        filtered = []
        for row in rows:
            base = str(row.get("base_meter", "")).strip()
            bucket = str(row.get("length_bucket", "")).strip()
            if eval_drop_meters and base in eval_drop_meters:
                continue
            if eval_allowed_length_buckets and bucket not in eval_allowed_length_buckets:
                continue
            filtered.append(row)
        return filtered

    train_manifest_rows = read_manifest_rows(train_manifest_path)
    train_manifest_ordered_uids, train_manifest_index = build_manifest_index(train_manifest_rows, "row_uid")
    hard_manifest_rows = read_manifest_rows(hard_manifest_path)
    hard_manifest_ordered_uids, hard_manifest_index = build_manifest_index(hard_manifest_rows, "row_uid")

    needed_train_uids = set(train_manifest_ordered_uids) | set(hard_manifest_ordered_uids)
    raw_train_dataset = load_dataset(train_dataset_id, split=cfg["dataset"]["train_split"], token=hf_token)
    raw_selected_train = raw_train_dataset.filter(lambda row: str(row.get("row_uid", "")).strip() in needed_train_uids)
    prepared_train_index: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(raw_selected_train):
        prepared = prepare_runtime_row_from_raw(dict(row), idx, cfg["dataset"]["train_split"])
        row_uid = str(prepared.get("row_uid", "")).strip()
        if not row_uid:
            continue
        if max_bayts is not None and int(prepared["requested_bayts"] or 0) > int(max_bayts):
            continue
        if allowed_meters and prepared["meter_label"] not in allowed_meters and prepared["base_meter"] not in allowed_meters:
            continue
        prepared_train_index[row_uid] = prepared

    missing_train = [row_uid for row_uid in train_manifest_ordered_uids if row_uid not in prepared_train_index]
    if missing_train:
        raise RuntimeError(f"Curated manifest rows missing from derived train dataset: {missing_train[:5]}")
    missing_hard = [row_uid for row_uid in hard_manifest_ordered_uids if row_uid not in prepared_train_index]
    if missing_hard:
        raise RuntimeError(f"Hard diagnostic rows missing from derived train dataset: {missing_hard[:5]}")

    train_selected_rows = []
    for row_uid in train_manifest_ordered_uids:
        row = dict(prepared_train_index[row_uid])
        row.update({f"manifest_{k}": v for k, v in train_manifest_index[row_uid].items()})
        train_selected_rows.append(row)

    hard_selected_rows = []
    for row_uid in hard_manifest_ordered_uids:
        row = dict(prepared_train_index[row_uid])
        row.update({f"manifest_{k}": v for k, v in hard_manifest_index[row_uid].items()})
        hard_selected_rows.append(row)

    eval_prepared = load_and_prepare_dataset(
        dataset_id=source_dataset_id,
        split=cfg["dataset"]["eval_split"],
        max_bayts=max_bayts,
        allowed_meters=allowed_meters,
        hf_token=hf_token,
    )
    test_prepared = load_and_prepare_dataset(
        dataset_id=source_dataset_id,
        split=cfg["dataset"]["test_split"],
        max_bayts=max_bayts,
        allowed_meters=allowed_meters,
        hf_token=hf_token,
    )
    eval_selected_rows = choose_balanced_rows(filter_eval_rows(list(eval_prepared)), per_meter_per_bucket=eval_bucket_quota, seed=int(cfg["run"]["seed"]))
    test_selected_rows = choose_balanced_rows(filter_eval_rows(list(test_prepared)), per_meter_per_bucket=test_bucket_quota, seed=int(cfg["run"]["seed"]) + 1)

    if mode == "sanity":
        sanity_limit = int(cfg["sanity_check"]["subset_size"])
        train_selected_rows = choose_sanity_subset(train_selected_rows, limit=sanity_limit, seed=int(cfg["run"]["seed"]))
        eval_selected_rows = choose_sanity_subset(eval_selected_rows, limit=sanity_limit, seed=int(cfg["run"]["seed"]) + 1)
        hard_selected_rows = choose_sanity_subset(hard_selected_rows, limit=sanity_limit, seed=int(cfg["run"]["seed"]) + 2)

    train_rows = build_dataset_rows(train_selected_rows, cfg["dataset"]["train_split"])
    eval_rows = build_dataset_rows(eval_selected_rows, cfg["dataset"]["eval_split"])
    hard_rows = build_dataset_rows(hard_selected_rows, "hard_diagnostic")
    dataset_summary = {
        "dataset_id": train_dataset_id,
        "dataset_source_id": source_dataset_id,
        "train_dataset_id": train_dataset_id,
        "source_dataset_id": source_dataset_id,
        "train_split": cfg["dataset"]["train_split"],
        "eval_split": cfg["dataset"]["eval_split"],
        "test_split": cfg["dataset"]["test_split"],
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "test_size": len(test_selected_rows),
        "hard_diagnostic_size": len(hard_rows),
        "phase1_max_bayts": max_bayts,
        "allowed_meters": allowed_meters or [],
        "train_manifest_path": str(train_manifest_path),
        "hard_diagnostic_manifest_path": str(hard_manifest_path),
        "eval_bank_per_meter_per_bucket": eval_bucket_quota,
        "test_bank_per_meter_per_bucket": test_bucket_quota,
        "eval_allowed_length_buckets": sorted(eval_allowed_length_buckets),
        "eval_drop_meters": sorted(eval_drop_meters),
        "train_length_bucket_counts": {},
        "eval_length_bucket_counts": {},
        "hard_diagnostic_length_bucket_counts": {},
        "train_base_meter_counts": {},
        "eval_base_meter_counts": {},
        "hard_diagnostic_base_meter_counts": {},
    }
    for row in train_rows:
        bucket = row["length_bucket"]
        dataset_summary["train_length_bucket_counts"][bucket] = dataset_summary["train_length_bucket_counts"].get(bucket, 0) + 1
        meter = row["base_meter"]
        dataset_summary["train_base_meter_counts"][meter] = dataset_summary["train_base_meter_counts"].get(meter, 0) + 1
    for row in eval_rows:
        bucket = row["length_bucket"]
        dataset_summary["eval_length_bucket_counts"][bucket] = dataset_summary["eval_length_bucket_counts"].get(bucket, 0) + 1
        meter = row["base_meter"]
        dataset_summary["eval_base_meter_counts"][meter] = dataset_summary["eval_base_meter_counts"].get(meter, 0) + 1
    for row in hard_rows:
        bucket = row["length_bucket"]
        dataset_summary["hard_diagnostic_length_bucket_counts"][bucket] = dataset_summary["hard_diagnostic_length_bucket_counts"].get(bucket, 0) + 1
        meter = row["base_meter"]
        dataset_summary["hard_diagnostic_base_meter_counts"][meter] = dataset_summary["hard_diagnostic_base_meter_counts"].get(meter, 0) + 1

    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows), dataset_summary


def build_reward_fns(cfg, run_dir: Path, recorder: GenerationRecorder):
    active = cfg["phase1"]["active_rewards"]
    weights_cfg = cfg["phase1"]["reward_weights"]
    reward_fns = []
    reward_weights = []
    meter_debug_path = run_dir / "reward_meter_debug.jsonl"
    count_adherence_debug_path = run_dir / "reward_count_adherence_debug.jsonl"
    arabic_clean_debug_path = run_dir / "reward_arabic_clean_debug.jsonl"
    hard_gate_debug_path = run_dir / "reward_hard_gate_debug.jsonl"
    repeat_soft_debug_path = run_dir / "reward_repeat_soft_debug.jsonl"
    arabic_floor_debug_path = run_dir / "reward_arabic_floor_debug.jsonl"
    count_floor_debug_path = run_dir / "reward_count_floor_debug.jsonl"
    lexical_plausibility_debug_path = run_dir / "reward_lexical_plausibility_debug.jsonl"
    repeat_penalty_debug_path = run_dir / "reward_repeat_penalty_debug.jsonl"
    repeat_floor_debug_path = run_dir / "reward_repeat_floor_debug.jsonl"
    near_duplicate_debug_path = run_dir / "reward_near_duplicate_penalty_debug.jsonl"
    opening_diversity_debug_path = run_dir / "reward_opening_diversity_debug.jsonl"
    distinct2_debug_path = run_dir / "reward_distinct_2_debug.jsonl"
    judge_quality_debug_path = run_dir / "reward_judge_quality_debug.jsonl"
    judge_meaning_fit_debug_path = run_dir / "reward_judge_meaning_fit_debug.jsonl"
    judge_naturalness_debug_path = run_dir / "reward_judge_naturalness_debug.jsonl"
    total_composite_debug_path = run_dir / "reward_total_composite_debug.jsonl"
    exact_count_debug_path = run_dir / "reward_exact_count_bonus_debug.jsonl"
    meter_count_clean_debug_path = run_dir / "reward_meter_count_clean_debug.jsonl"
    judge_meaning_prompt_file_cfg = str(
        cfg["phase1"].get(
            "judge_meaning_fit_prompt_file",
            cfg["phase1"].get("judge_quality_prompt_file", "prompts/judge_quality_selected.yaml"),
        )
    ).strip()
    judge_meaning_prompt_file = Path(judge_meaning_prompt_file_cfg)
    if not judge_meaning_prompt_file.is_absolute():
        judge_meaning_prompt_file = ROOT / judge_meaning_prompt_file
    judge_naturalness_prompt_file_cfg = str(
        cfg["phase1"].get("judge_naturalness_prompt_file", "prompts/judge_naturalness_v1.yaml")
    ).strip()
    judge_naturalness_prompt_file = Path(judge_naturalness_prompt_file_cfg)
    if not judge_naturalness_prompt_file.is_absolute():
        judge_naturalness_prompt_file = ROOT / judge_naturalness_prompt_file
    judge_cache_dir_cfg = str(
        cfg["phase1"].get(
            "judge_cache_dir",
            cfg["phase1"].get("judge_quality_cache_dir", run_dir / "judge_cache"),
        )
    ).strip()
    judge_cache_dir = Path(judge_cache_dir_cfg)
    if not judge_cache_dir.is_absolute():
        judge_cache_dir = ROOT / judge_cache_dir
    judge_max_workers = int(cfg["phase1"].get("judge_max_workers", cfg["phase1"].get("judge_quality_max_workers", 8)))

    component_cache: dict[str, Any] = {"key": None, "rows": None}

    def normalize_list(value, size: int):
        if isinstance(value, list):
            return value
        return [value] * size

    def prepare_component_rows(completions, meter_label, base_meter, requested_bayts, dataset_split, description=None):
        completions = completions or []
        size = len(completions)
        meter_label_local = normalize_list(meter_label, size)
        base_meter_local = normalize_list(base_meter, size)
        requested_bayts_local = normalize_list(requested_bayts, size)
        dataset_split_local = normalize_list(dataset_split, size)
        description_local = normalize_list(description, size)
        cache_key = (
            tuple(sha256_text(extract_text(completion)) for completion in completions),
            tuple(str(value or "") for value in meter_label_local),
            tuple(str(value or "") for value in base_meter_local),
            tuple(int(value or 0) for value in requested_bayts_local),
            tuple(str(value or "") for value in dataset_split_local),
            tuple(sha256_text(str(value or "")) for value in description_local),
        )
        if component_cache["key"] == cache_key and component_cache["rows"] is not None:
            return meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, description_local, component_cache["rows"]

        texts = [extract_text(completion) for completion in completions]
        judge_meaning_rows = batch_score_judge_quality(
            descriptions=[str(value or "") for value in description_local],
            poems=texts,
            prompt_file=str(judge_meaning_prompt_file),
            cache_dir=str(judge_cache_dir),
            cache_namespace="judge_meaning_fit",
            max_workers=judge_max_workers,
        )
        judge_naturalness_rows = batch_score_judge_quality(
            descriptions=[str(value or "") for value in description_local],
            poems=texts,
            prompt_file=str(judge_naturalness_prompt_file),
            cache_dir=str(judge_cache_dir),
            cache_namespace="judge_naturalness",
            max_workers=judge_max_workers,
        )

        rows = []
        for i, completion in enumerate(completions):
            text = texts[i]
            target = meter_label_local[i] if i < len(meter_label_local) else ""
            base = base_meter_local[i] if i < len(base_meter_local) else ""
            requested = int(requested_bayts_local[i] or 0) if i < len(requested_bayts_local) else 0
            split_name = str(dataset_split_local[i] if i < len(dataset_split_local) else "")
            desc = str(description_local[i] if i < len(description_local) else "")

            meter_out = score_meter_poem(text, target, base_meter=base, aggregator="logmean")
            count_out = score_count_adherence(requested, text)
            clean_out = score_arabic_cleanliness(text)
            repeat_out = score_repetition_penalty(text)
            hard_gate_out = score_minimal_hard_gate(
                generated_poem=text,
                generated_bayts=int(count_out.get("generated_bayts", 0) or 0),
                clean_out=clean_out,
            )
            judge_meaning_out = judge_meaning_rows[i] if i < len(judge_meaning_rows) and judge_meaning_rows[i] is not None else {
                "score": 0.0,
                "failure_mode": "missing",
                "notes": "missing",
                "cache_hit": False,
                "latency_sec": None,
                "error": "missing",
                "mode_respected": None,
                "reasoning_tokens": None,
                "prompt_name": judge_meaning_prompt_file.stem,
                "prompt_file": str(judge_meaning_prompt_file),
            }
            judge_naturalness_out = judge_naturalness_rows[i] if i < len(judge_naturalness_rows) and judge_naturalness_rows[i] is not None else {
                "score": 0.0,
                "failure_mode": "missing",
                "notes": "missing",
                "cache_hit": False,
                "latency_sec": None,
                "error": "missing",
                "mode_respected": None,
                "reasoning_tokens": None,
                "prompt_name": judge_naturalness_prompt_file.stem,
                "prompt_file": str(judge_naturalness_prompt_file),
            }
            meter_score = float(meter_out["score"])
            count_score = float(count_out["score"])
            arabic_score = float(clean_out["score"])
            repeat_score = float(repeat_out["score"])
            judge_meaning_score = float(judge_meaning_out["score"])
            judge_naturalness_score = float(judge_naturalness_out["score"])
            judge_quality_score = 0.5 * (judge_meaning_score + judge_naturalness_score)
            hard_gate_score = float(hard_gate_out["hard_gate_score"])
            repeat_soft_score = score_repeat_soft_signal(
                exact_repeat_score=float(repeat_out["exact_repeat_score"]),
                near_duplicate_score=float(repeat_out["near_duplicate_score"]),
                opening_diversity_score=float(repeat_out["opening_diversity_score"]),
                distinct_2_score=float(repeat_out["distinct_2_score"]),
            )
            weighted_reward_out = score_weighted_reward_train(
                meter_score=meter_score,
                count_adherence_score=count_score,
                judge_meaning_fit_score=judge_meaning_score,
                judge_naturalness_score=judge_naturalness_score,
                repeat_soft_score=repeat_soft_score,
                hard_gate_score=hard_gate_score,
            )
            total_score = float(weighted_reward_out["total_score"])
            exact_flag = 1.0 if int(count_out.get("generated_bayts", -1) or -1) == int(requested or 0) else 0.0
            judge_failure_parts = []
            for label, out in [("meaning", judge_meaning_out), ("naturalness", judge_naturalness_out)]:
                failure_mode = str(out.get("failure_mode", "") or "")
                if failure_mode:
                    judge_failure_parts.append(f"{label}:{failure_mode}")
            judge_notes_parts = []
            for label, out in [("meaning", judge_meaning_out), ("naturalness", judge_naturalness_out)]:
                notes = str(out.get("notes", "") or "").strip()
                if notes:
                    judge_notes_parts.append(f"{label}:{notes}")

            rows.append(
                {
                    "requested_bayts": requested,
                    "generated_bayts": int(count_out.get("generated_bayts", 0) or 0),
                    "count_adherence_score": count_score,
                    "exact_count": bool(exact_flag),
                    "has_odd_tail": count_out.get("has_odd_tail"),
                    "num_lines": count_out.get("num_lines"),
                    "arabic_clean_ok": bool(arabic_score),
                    "arabic_clean_score": arabic_score,
                    "hard_gate_score": hard_gate_score,
                    "hard_gate_blocked": bool(hard_gate_out["hard_gate_blocked"]),
                    "hard_gate_reason": str(hard_gate_out["hard_gate_reason"]),
                    "reward_weighted_sum": float(weighted_reward_out["weighted_sum"]),
                    "meter_meaning_core": float(weighted_reward_out.get("meter_meaning_core", 0.0)),
                    "naturalness_bonus": float(weighted_reward_out.get("naturalness_bonus", 0.0)),
                    "count_bonus": float(weighted_reward_out.get("count_bonus", 0.0)),
                    "repeat_bonus": float(weighted_reward_out.get("repeat_bonus", 0.0)),
                    "arabic_floor_score": 1.0,
                    "count_floor_score": 1.0,
                    "repeat_soft_score": repeat_soft_score,
                    "repeat_floor_score": 1.0,
                    "contamination_ratio": float(hard_gate_out["contamination_ratio"]),
                    "non_arabic_non_punct_count": int(hard_gate_out["non_arabic_non_punct_count"]),
                    "text_length": int(hard_gate_out["text_length"]),
                    "stripped_text_length": int(hard_gate_out["stripped_text_length"]),
                    "has_arabic": bool(clean_out["has_arabic"]),
                    "contains_latin": bool(clean_out["contains_latin"]),
                    "contains_digits": bool(clean_out["contains_digits"]),
                    "contains_forbidden_artifacts": bool(clean_out["contains_forbidden_artifacts"]),
                    "arabic_char_count": int(clean_out["arabic_char_count"]),
                    "latin_char_count": int(clean_out["latin_char_count"]),
                    "digit_char_count": int(clean_out["digit_char_count"]),
                    "forbidden_artifact_count": int(clean_out["forbidden_artifact_count"]),
                    "arabic_token_count": int(clean_out["arabic_token_count"]),
                    "known_token_count": int(clean_out["known_token_count"]),
                    "lexical_known_ratio": float(clean_out["lexical_known_ratio"]),
                    "content_token_count": int(clean_out["content_token_count"]),
                    "known_content_count": int(clean_out["known_content_count"]),
                    "content_known_ratio": float(clean_out["content_known_ratio"]),
                    "lexical_plausibility_score": float(clean_out["lexical_plausibility_score"]),
                    "artifact_free_score": float(clean_out["artifact_free_score"]),
                    "repeat_penalty_score": repeat_score,
                    "has_repeated_line": bool(repeat_out["has_repeated_line"]),
                    "max_line_repeat_count": int(repeat_out["max_line_repeat_count"]),
                    "longest_consecutive_repeat_run": int(repeat_out["longest_consecutive_repeat_run"]),
                    "dominant_line_fraction": float(repeat_out["dominant_line_fraction"]),
                    "duplicate_extra_lines": int(repeat_out["duplicate_extra_lines"]),
                    "dominant_repeated_line": str(repeat_out["dominant_repeated_line"]),
                    "exact_repeat_score": float(repeat_out["exact_repeat_score"]),
                    "near_duplicate_score": float(repeat_out["near_duplicate_score"]),
                    "opening_diversity_score": float(repeat_out["opening_diversity_score"]),
                    "distinct_2_score": float(repeat_out["distinct_2_score"]),
                    "distinct_1": float(repeat_out["distinct_1"]),
                    "distinct_2": float(repeat_out["distinct_2"]),
                    "distinct_3": float(repeat_out["distinct_3"]),
                    "opening_dominance_fraction": float(repeat_out["opening_dominance_fraction"]),
                    "line_pair_similarity_mean": float(repeat_out["line_pair_similarity_mean"]),
                    "line_pair_similarity_max": float(repeat_out["line_pair_similarity_max"]),
                    "sequence_similarity_mean": float(repeat_out["sequence_similarity_mean"]),
                    "sequence_similarity_max": float(repeat_out["sequence_similarity_max"]),
                    "near_duplicate_pair_fraction": float(repeat_out["near_duplicate_pair_fraction"]),
                    "judge_quality_score": float(judge_quality_score),
                    "judge_meaning_fit_score": float(judge_meaning_score),
                    "judge_naturalness_score": float(judge_naturalness_score),
                    "judge_failure_mode": " | ".join(judge_failure_parts),
                    "judge_notes": " || ".join(judge_notes_parts),
                    "judge_cache_hit": bool(judge_meaning_out.get("cache_hit", False) and judge_naturalness_out.get("cache_hit", False)),
                    "judge_latency_sec": (
                        (float(judge_meaning_out.get("latency_sec", 0.0) or 0.0) + float(judge_naturalness_out.get("latency_sec", 0.0) or 0.0))
                    ),
                    "judge_error": " | ".join(
                        str(value)
                        for value in [judge_meaning_out.get("error"), judge_naturalness_out.get("error")]
                        if value not in (None, "", "None")
                    ),
                    "judge_mode_respected": bool(
                        judge_meaning_out.get("mode_respected", True) and judge_naturalness_out.get("mode_respected", True)
                    ),
                    "judge_reasoning_tokens": int(
                        float(judge_meaning_out.get("reasoning_tokens", 0) or 0)
                        + float(judge_naturalness_out.get("reasoning_tokens", 0) or 0)
                    ),
                    "judge_prompt_name": f"{judge_meaning_out.get('prompt_name', judge_meaning_prompt_file.stem)}+{judge_naturalness_out.get('prompt_name', judge_naturalness_prompt_file.stem)}",
                    "judge_prompt_file": f"{judge_meaning_out.get('prompt_file', judge_meaning_prompt_file)}|{judge_naturalness_out.get('prompt_file', judge_naturalness_prompt_file)}",
                    "judge_meaning_fit_failure_mode": str(judge_meaning_out.get("failure_mode", "") or ""),
                    "judge_meaning_fit_notes": str(judge_meaning_out.get("notes", "") or ""),
                    "judge_meaning_fit_cache_hit": bool(judge_meaning_out.get("cache_hit", False)),
                    "judge_meaning_fit_latency_sec": judge_meaning_out.get("latency_sec"),
                    "judge_meaning_fit_error": judge_meaning_out.get("error"),
                    "judge_meaning_fit_mode_respected": judge_meaning_out.get("mode_respected"),
                    "judge_meaning_fit_reasoning_tokens": judge_meaning_out.get("reasoning_tokens"),
                    "judge_meaning_fit_prompt_name": str(judge_meaning_out.get("prompt_name", judge_meaning_prompt_file.stem)),
                    "judge_meaning_fit_prompt_file": str(judge_meaning_out.get("prompt_file", judge_meaning_prompt_file)),
                    "judge_naturalness_failure_mode": str(judge_naturalness_out.get("failure_mode", "") or ""),
                    "judge_naturalness_notes": str(judge_naturalness_out.get("notes", "") or ""),
                    "judge_naturalness_cache_hit": bool(judge_naturalness_out.get("cache_hit", False)),
                    "judge_naturalness_latency_sec": judge_naturalness_out.get("latency_sec"),
                    "judge_naturalness_error": judge_naturalness_out.get("error"),
                    "judge_naturalness_mode_respected": judge_naturalness_out.get("mode_respected"),
                    "judge_naturalness_reasoning_tokens": judge_naturalness_out.get("reasoning_tokens"),
                    "judge_naturalness_prompt_name": str(judge_naturalness_out.get("prompt_name", judge_naturalness_prompt_file.stem)),
                    "judge_naturalness_prompt_file": str(judge_naturalness_out.get("prompt_file", judge_naturalness_prompt_file)),
                    "meter_score": meter_score,
                    "total_composite_score": total_score,
                    "target_meter_requested": target,
                    "base_meter": base,
                    "target_meter_used": meter_out["target_meter_used"],
                    "target_resolution": meter_out["target_resolution"],
                    "candidate_labels": meter_out["candidate_labels"],
                    "num_valid_bayts": meter_out["num_valid_bayts"],
                    "num_skipped_bayts": meter_out["num_skipped_bayts"],
                    "complete_bayts": meter_out["complete_bayts"],
                    "odd_tail_line": meter_out["odd_tail_line"],
                    "mean_meter_score": meter_out["mean_score"],
                    "logmean_meter_score": meter_out["logmean_score"],
                    "std_meter_score": meter_out["std_score"],
                    "min_meter_score": meter_out["min_score"],
                    "max_meter_score": meter_out["max_score"],
                    "per_bayt_details": meter_out["per_bayt_details"],
                    "dataset_split": split_name,
                    "description_preview": desc[:400],
                    "poem_preview": text[:400],
                }
            )

        component_cache["key"] = cache_key
        component_cache["rows"] = rows
        return meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, description_local, rows

    if "meter" in active:

        def meter_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["meter_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(meter_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="meter",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        meter_reward_fn.__name__ = "meter"
        reward_fns.append(meter_reward_fn)
        reward_weights.append(float(weights_cfg.get("meter", 1.0)))

    if "count_adherence" in active:

        def count_adherence_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["count_adherence_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(count_adherence_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="count_adherence",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        count_adherence_reward_fn.__name__ = "count_adherence"
        reward_fns.append(count_adherence_reward_fn)
        reward_weights.append(float(weights_cfg.get("count_adherence", 1.0)))

    if "arabic_clean" in active:

        def arabic_clean_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["arabic_clean_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(arabic_clean_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="arabic_clean",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        arabic_clean_reward_fn.__name__ = "arabic_clean"
        reward_fns.append(arabic_clean_reward_fn)
        reward_weights.append(float(weights_cfg.get("arabic_clean", 1.0)))

    if "hard_gate" in active:

        def hard_gate_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["hard_gate_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(hard_gate_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="hard_gate",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        hard_gate_reward_fn.__name__ = "hard_gate"
        reward_fns.append(hard_gate_reward_fn)
        reward_weights.append(float(weights_cfg.get("hard_gate", 1.0)))

    if "repeat_soft" in active:

        def repeat_soft_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["repeat_soft_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(repeat_soft_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="repeat_soft",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        repeat_soft_reward_fn.__name__ = "repeat_soft"
        reward_fns.append(repeat_soft_reward_fn)
        reward_weights.append(float(weights_cfg.get("repeat_soft", 1.0)))

    if "arabic_floor" in active:

        def arabic_floor_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["arabic_floor_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(arabic_floor_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="arabic_floor",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        arabic_floor_reward_fn.__name__ = "arabic_floor"
        reward_fns.append(arabic_floor_reward_fn)
        reward_weights.append(float(weights_cfg.get("arabic_floor", 1.0)))

    if "count_floor" in active:

        def count_floor_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["count_floor_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(count_floor_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="count_floor",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        count_floor_reward_fn.__name__ = "count_floor"
        reward_fns.append(count_floor_reward_fn)
        reward_weights.append(float(weights_cfg.get("count_floor", 1.0)))

    if "lexical_plausibility" in active:

        def lexical_plausibility_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["lexical_plausibility_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(lexical_plausibility_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="lexical_plausibility",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        lexical_plausibility_reward_fn.__name__ = "lexical_plausibility"
        reward_fns.append(lexical_plausibility_reward_fn)
        reward_weights.append(float(weights_cfg.get("lexical_plausibility", 1.0)))

    if "judge_quality" in active:

        def judge_quality_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["judge_quality_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(judge_quality_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="judge_quality",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        judge_quality_reward_fn.__name__ = "judge_quality"
        reward_fns.append(judge_quality_reward_fn)
        reward_weights.append(float(weights_cfg.get("judge_quality", 1.0)))

    if "judge_meaning_fit" in active:

        def judge_meaning_fit_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["judge_meaning_fit_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(judge_meaning_fit_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="judge_meaning_fit",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        judge_meaning_fit_reward_fn.__name__ = "judge_meaning_fit"
        reward_fns.append(judge_meaning_fit_reward_fn)
        reward_weights.append(float(weights_cfg.get("judge_meaning_fit", 1.0)))

    if "judge_naturalness" in active:

        def judge_naturalness_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["judge_naturalness_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(judge_naturalness_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="judge_naturalness",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        judge_naturalness_reward_fn.__name__ = "judge_naturalness"
        reward_fns.append(judge_naturalness_reward_fn)
        reward_weights.append(float(weights_cfg.get("judge_naturalness", 1.0)))

    if "repeat_penalty" in active:

        def repeat_penalty_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["repeat_penalty_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(repeat_penalty_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="repeat_penalty",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        repeat_penalty_reward_fn.__name__ = "repeat_penalty"
        reward_fns.append(repeat_penalty_reward_fn)
        reward_weights.append(float(weights_cfg.get("repeat_penalty", 1.0)))

    if "repeat_floor" in active:

        def repeat_floor_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["repeat_floor_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(repeat_floor_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="repeat_floor",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        repeat_floor_reward_fn.__name__ = "repeat_floor"
        reward_fns.append(repeat_floor_reward_fn)
        reward_weights.append(float(weights_cfg.get("repeat_floor", 1.0)))

    if "near_duplicate_penalty" in active:

        def near_duplicate_penalty_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["near_duplicate_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(near_duplicate_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="near_duplicate_penalty",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        near_duplicate_penalty_reward_fn.__name__ = "near_duplicate_penalty"
        reward_fns.append(near_duplicate_penalty_reward_fn)
        reward_weights.append(float(weights_cfg.get("near_duplicate_penalty", 1.0)))

    if "opening_diversity" in active:

        def opening_diversity_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["opening_diversity_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(opening_diversity_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="opening_diversity",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        opening_diversity_reward_fn.__name__ = "opening_diversity"
        reward_fns.append(opening_diversity_reward_fn)
        reward_weights.append(float(weights_cfg.get("opening_diversity", 1.0)))

    if "distinct_2" in active:

        def distinct_2_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["distinct_2_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(distinct2_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="distinct_2",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        distinct_2_reward_fn.__name__ = "distinct_2"
        reward_fns.append(distinct_2_reward_fn)
        reward_weights.append(float(weights_cfg.get("distinct_2", 1.0)))

    if "total_composite" in active:

        def total_composite_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["total_composite_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(total_composite_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="total_composite",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        total_composite_reward_fn.__name__ = "total_composite"
        reward_fns.append(total_composite_reward_fn)
        reward_weights.append(float(weights_cfg.get("total_composite", 1.0)))

    if "exact_count_bonus" in active:

        def exact_count_bonus_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [1.0 if bool(row["exact_count"]) else 0.0 for row in component_rows]
            extras = []
            for i, row in enumerate(component_rows):
                extra = dict(row)
                extra["bonus_value"] = rewards[i] * float(weights_cfg.get("exact_count_bonus", 1.0))
                extras.append(extra)
                append_jsonl(exact_count_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="exact_count_bonus",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        exact_count_bonus_reward_fn.__name__ = "exact_count_bonus"
        reward_fns.append(exact_count_bonus_reward_fn)
        reward_weights.append(float(weights_cfg.get("exact_count_bonus", 1.0)))

    if "meter_count_clean" in active:

        def meter_count_clean_reward_fn(prompts=None, completions=None, meter_label=None, base_meter=None, requested_bayts=None, dataset_split=None, **kwargs):
            completions = completions or []
            meter_label_local, base_meter_local, requested_bayts_local, dataset_split_local, _description_local, component_rows = prepare_component_rows(
                completions,
                meter_label,
                base_meter,
                requested_bayts,
                dataset_split,
                kwargs.get("description", None),
            )
            rewards = [float(row["total_composite_score"]) for row in component_rows]
            extras = [dict(row) for row in component_rows]
            for i, extra in enumerate(extras):
                append_jsonl(meter_count_clean_debug_path, {"i": i, "score": rewards[i], **extra})

            recorder.record_reward_batch(
                reward_name="meter_count_clean",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "meter_label": meter_label_local,
                    "base_meter": base_meter_local,
                    "requested_bayts": requested_bayts_local,
                    "dataset_split": dataset_split_local,
                },
            )
            return rewards

        meter_count_clean_reward_fn.__name__ = "meter_count_clean"
        reward_fns.append(meter_count_clean_reward_fn)
        reward_weights.append(float(weights_cfg.get("meter_count_clean", 1.0)))

    reward_weight_map = {name: float(weights_cfg.get(name, 1.0)) for name in active}
    return reward_fns, reward_weights, reward_weight_map


def build_grpo_args(cfg, run_dir: Path, mode: str, reward_weights: list[float], output_repo: str):
    trainer_cfg = cfg["trainer"]
    gen_cfg = cfg["generation"]
    sanity_cfg = cfg["sanity_check"]
    seed = int(cfg["run"]["seed"])

    max_steps = int(sanity_cfg["max_steps"]) if mode == "sanity" else int(trainer_cfg["max_steps"])
    num_generations = int(sanity_cfg["num_generations"]) if mode == "sanity" else int(gen_cfg["num_generations"])
    eval_num_generations = int(sanity_cfg["num_generations"]) if mode == "sanity" else int(gen_cfg["num_generations_eval"])
    save_steps = int(sanity_cfg["save_steps"]) if mode == "sanity" else int(trainer_cfg["save_steps"])
    eval_steps = int(sanity_cfg["eval_steps"]) if mode == "sanity" else int(trainer_cfg["eval_steps"])
    logging_steps = int(sanity_cfg["logging_steps"]) if mode == "sanity" else int(trainer_cfg["logging_steps"])
    eval_batch_size = int(trainer_cfg["per_device_eval_batch_size"])
    if eval_num_generations > 0 and eval_batch_size % eval_num_generations != 0:
        eval_batch_size = ((eval_batch_size + eval_num_generations - 1) // eval_num_generations) * eval_num_generations

    args = GRPOConfig(
        output_dir=str(run_dir),
        run_name=run_dir.name,
        seed=seed,
        learning_rate=float(trainer_cfg["learning_rate"]),
        per_device_train_batch_size=int(trainer_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=int(trainer_cfg["gradient_accumulation_steps"]),
        max_steps=max_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(trainer_cfg["save_total_limit"]),
        beta=float(trainer_cfg["beta"]),
        bf16=bool(trainer_cfg["bf16"]),
        remove_unused_columns=bool(trainer_cfg["remove_unused_columns"]),
        report_to=trainer_cfg["report_to"],
        max_prompt_length=int(gen_cfg["max_prompt_length"]),
        max_completion_length=int(gen_cfg["max_completion_length"]),
        num_generations=num_generations,
        use_vllm=bool(gen_cfg["use_vllm"]),
        vllm_mode=str(gen_cfg["vllm_mode"]),
        vllm_gpu_memory_utilization=float(gen_cfg["vllm_gpu_memory_utilization"]),
        temperature=float(gen_cfg["temperature"]),
        top_p=float(gen_cfg["top_p"]),
        reward_weights=reward_weights,
        scale_rewards=str(trainer_cfg["scale_rewards"]),
        loss_type=str(trainer_cfg["loss_type"]),
        mask_truncated_completions=bool(trainer_cfg["mask_truncated_completions"]),
        load_best_model_at_end=bool(trainer_cfg["load_best_model_at_end"]),
        metric_for_best_model=str(trainer_cfg["metric_for_best_model"]),
        greater_is_better=bool(trainer_cfg["greater_is_better"]),
        push_to_hub=bool(output_repo),
        hub_model_id=output_repo or None,
        hub_token=os.getenv("HF_TOKEN", "").strip() or None,
        hub_strategy=str(trainer_cfg["hub_strategy"]),
        log_completions=False,
        gradient_checkpointing=bool(cfg["model"]["gradient_checkpointing"]),
    )
    return args


def save_snapshots(cfg, run_dir: Path, dataset_summary: dict[str, Any], model_meta: dict[str, Any], fingerprint: str, lineage: dict[str, Any]):
    env_snapshot = dict(os.environ)
    save_json(mask_env(env_snapshot), run_dir / "env_snapshot_masked.json")
    save_json(cfg, run_dir / "config_snapshot.json")
    save_json(dataset_summary, run_dir / "dataset_summary.json")
    save_json(json_safe(lineage), run_dir / "lineage.json")
    save_json(
        {
            "config_fingerprint": fingerprint,
            "lineage": json_safe(lineage),
            "dataset_id": dataset_summary["dataset_id"],
            "dataset_source_id": dataset_summary["dataset_source_id"],
            "phase1_max_bayts": os.getenv("PHASE1_MAX_BAYTS", "").strip(),
            "phase1_allowed_meters": parse_allowed_meters_env(),
            "base_model_id": resolve_base_model_id(cfg),
            "sft_adapter_repo": resolve_start_adapter_repo(cfg),
            "sft_adapter_mode": resolve_start_adapter_mode(cfg),
            "grpo_output_repo": resolve_output_repo(cfg),
            "model_meta": model_meta,
        },
        run_dir / "runtime_snapshot.json",
    )


def download_checkpoint_prefix(repo_id: str, prefix: str, hf_token: str, cache_dir: Path) -> str:
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
        allow_patterns=[f"{prefix}/*"],
        cache_dir=str(cache_dir),
    )
    local_path = Path(snapshot_dir) / prefix
    if not local_path.exists():
        raise FileNotFoundError(f"Downloaded checkpoint prefix but local path is missing: {local_path}")
    return str(local_path)


def read_resume_manifest(path: Path) -> dict[str, Any] | None:
    manifest_path = path / CHECKPOINT_MANIFEST_NAME
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_resume_checkpoint(run_dir: Path, dataset_summary: dict[str, Any], fingerprint: str, output_repo: str):
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    resume_mode = os.getenv("GRPO_RESUME_MODE", "auto").strip() or "auto"
    resume_spec = os.getenv("GRPO_RESUME_PATH", "").strip()
    cache_dir = ensure_dir(run_dir / "_resume_cache")

    decision = {
        "timestamp_utc": utc_now_iso(),
        "resume_mode": resume_mode,
        "requested_spec": resume_spec,
        "result": "fresh",
        "reason": "",
        "local_resume_path": None,
        "remote_repo": None,
        "remote_prefix": None,
        "compatibility": "unchecked",
        "config_fingerprint": fingerprint,
    }

    def validate_manifest(local_path: Path, allow_incompatible: bool):
        manifest = read_resume_manifest(local_path)
        if manifest is None:
            decision["compatibility"] = "manifest_missing"
            return True, None
        if manifest.get("config_fingerprint") == fingerprint:
            decision["compatibility"] = "compatible"
            return True, manifest
        decision["compatibility"] = "fingerprint_mismatch"
        if allow_incompatible:
            return True, manifest
        return False, manifest

    if resume_mode == "fresh":
        decision["reason"] = "explicit_fresh"
        return None, decision, None

    if resume_mode not in {"auto", "from_path"}:
        raise ValueError(f"Unsupported GRPO_RESUME_MODE: {resume_mode}")

    if resume_mode == "from_path" and not resume_spec:
        raise ValueError("GRPO_RESUME_MODE=from_path requires GRPO_RESUME_PATH")

    if resume_spec and Path(resume_spec).is_dir():
        local_path = Path(resume_spec).resolve()
        valid, manifest = validate_manifest(local_path, allow_incompatible=(resume_mode == "from_path"))
        if not valid:
            raise RuntimeError(f"Refusing to resume from incompatible checkpoint: {local_path}")
        decision["result"] = "resume"
        decision["reason"] = "explicit_local_path" if resume_mode == "from_path" else "auto_local_path"
        decision["local_resume_path"] = str(local_path)
        return str(local_path), decision, manifest

    def resolve_remote(spec: str):
        if not output_repo:
            return None
        if spec and "@" in spec:
            repo_id, prefix = spec.split("@", 1)
            return repo_id.strip(), prefix.strip()
        if spec:
            return output_repo, spec
        return output_repo, "last-checkpoint"

    remote = resolve_remote(resume_spec)
    if remote is None:
        decision["reason"] = "no_output_repo_for_remote_resume"
        return None, decision, None

    repo_id, prefix = remote
    try:
        local_path_str = download_checkpoint_prefix(repo_id, prefix, hf_token, cache_dir)
        local_path = Path(local_path_str)
        valid, manifest = validate_manifest(local_path, allow_incompatible=(resume_mode == "from_path"))
        if not valid:
            if resume_mode == "from_path":
                raise RuntimeError(f"Refusing to resume from incompatible remote checkpoint: {repo_id}@{prefix}")
            decision["reason"] = "remote_manifest_incompatible"
            return None, decision, manifest
        decision["result"] = "resume"
        decision["reason"] = "explicit_remote_prefix" if resume_mode == "from_path" else "auto_last_checkpoint"
        decision["local_resume_path"] = str(local_path)
        decision["remote_repo"] = repo_id
        decision["remote_prefix"] = prefix
        return str(local_path), decision, manifest
    except Exception as exc:
        decision["reason"] = f"resume_not_found: {type(exc).__name__}"
        if resume_mode == "from_path":
            raise
        return None, decision, None

def resolve_output_repo(cfg) -> str:
    configured = str(cfg["run"].get("output_model_repo", "")).strip()
    return os.getenv("GRPO_OUTPUT_REPO", "").strip() or configured


def build_lineage(run_dir: Path, resume_decision: dict[str, Any], resume_manifest: dict[str, Any] | None) -> dict[str, Any]:
    run_id = run_dir.name
    if resume_decision.get("result") == "resume":
        parent_run_id = str((resume_manifest or {}).get("run_id", ""))
        chain_id = str((resume_manifest or {}).get("chain_id") or (resume_manifest or {}).get("root_run_id") or parent_run_id or run_id)
        root_run_id = str((resume_manifest or {}).get("root_run_id") or chain_id or run_id)
        parent_run_dir = str((resume_manifest or {}).get("run_dir", ""))
        run_sequence_index = int((resume_manifest or {}).get("run_sequence_index", 0) or 0) + 1
    else:
        parent_run_id = ""
        chain_id = run_id
        root_run_id = run_id
        parent_run_dir = ""
        run_sequence_index = 0
    return {
        "created_at_utc": utc_now_iso(),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "chain_id": chain_id,
        "root_run_id": root_run_id,
        "parent_run_id": parent_run_id,
        "parent_run_dir": parent_run_dir,
        "run_sequence_index": run_sequence_index,
    }


def read_lineage(run_dir: Path) -> dict[str, Any] | None:
    for name in ["lineage.json", "run_summary.json"]:
        data = read_json(run_dir / name)
        if data:
            return data
    return None


def discover_chain_runs(run_dir: Path, chain_id: str) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for child in sorted(run_dir.parent.iterdir()):
        if not child.is_dir():
            continue
        lineage = read_lineage(child)
        if not lineage:
            continue
        if str(lineage.get("chain_id", "")) != str(chain_id):
            continue
        out.append((child, lineage))
    out.sort(
        key=lambda item: (
            int(item[1].get("run_sequence_index", 0) or 0),
            parse_utc_timestamp(str(item[1].get("created_at_utc", ""))),
            item[0].name,
        )
    )
    return out


def enrich_rows_for_chain(rows: list[dict[str, Any]], lineage: dict[str, Any], default_mode: str | None = None) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row.setdefault("run_id", str(lineage.get("run_id", "")))
        new_row.setdefault("run_dir", str(lineage.get("run_dir", "")))
        new_row.setdefault("chain_id", str(lineage.get("chain_id", "")))
        new_row.setdefault("root_run_id", str(lineage.get("root_run_id", "")))
        new_row.setdefault("parent_run_id", str(lineage.get("parent_run_id", "")))
        new_row.setdefault("run_sequence_index", int(lineage.get("run_sequence_index", 0) or 0))
        if default_mode is not None:
            new_row.setdefault("mode", default_mode)
        enriched.append(new_row)
    return enriched


def sort_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("run_sequence_index", 0) or 0),
            int(row.get("logical_step", row.get("global_step", 0)) or 0),
            int(row.get("mode_order", 0 if row.get("mode") == "train" else 1)),
            int(row.get("batch_index", 0) or 0),
            row_sort_value(row.get("source_index")),
            str(row.get("prompt_hash", "")),
            int(row.get("candidate_index", row.get("prompt_candidate_index", 0)) or 0),
            str(row.get("completion_hash", "")),
        ),
    )


def sort_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("run_sequence_index", 0) or 0),
            int(row.get("global_step", 0) or 0),
            0 if str(row.get("mode", "train")) == "train" else 1,
            parse_utc_timestamp(str(row.get("timestamp_utc", ""))),
        ),
    )


def write_sorted_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    if not rows:
        pd.DataFrame([]).to_parquet(path, index=False)
        return
    pd.DataFrame([json_safe(row) for row in rows]).to_parquet(path, index=False)


def build_generations_dataset_readme(output_repo: str, generations_repo: str) -> str:
    return "\n".join(
        [
            "---",
            "pretty_name: Shaer GRPO Generations",
            "license: other",
            "task_categories:",
            "- text-generation",
            "---",
            "",
            "# Shaer GRPO Generations",
            "",
            f"- Source model/checkpoint repo: `{output_repo}`",
            f"- This dataset stores run-level generation artifacts exported from GRPO runs.",
            "",
            "## Layout",
            "",
            "- `runs/<run_id>/generations.parquet`: sorted candidate generations for one run",
            "- `runs/<run_id>/metrics.parquet`: run metrics rows for one run",
            "- `runs/<run_id>/run_summary.json`: run summary and lineage metadata",
            "- `chains/<chain_id>/chain_metrics.parquet`: merged metrics across resumed runs in one chain",
            "- `chains/<chain_id>/chain_reward_curves.png`: reward plot across the chain",
            "",
            "## Notes",
            "",
            "- Raw local JSONL logs remain the source of truth during training.",
            "- This Hub dataset is the analysis/archive view for plotting, paper tables, and offline judging.",
            "",
            f"Generated by the Shaer GRPO runtime into `{generations_repo}`.",
        ]
    )


def export_generations_to_hub(run_dir: Path, output_repo: str, lineage: dict[str, Any]) -> dict[str, Any]:
    generations_repo = resolve_generations_repo(output_repo)
    hf_token = os.getenv("HF_TOKEN", "").strip() or None
    if not generations_repo or not hf_token:
        return {"enabled": False, "repo_id": generations_repo, "reason": "missing_repo_or_token"}

    run_generations = read_jsonl(run_dir / "all_generations.jsonl")
    run_metrics = read_jsonl(run_dir / "metrics.jsonl")
    run_events = read_jsonl(run_dir / "checkpoint_events.jsonl")

    run_generations = sort_generation_rows(enrich_rows_for_chain(run_generations, lineage))
    run_metrics = sort_metric_rows(enrich_rows_for_chain(run_metrics, lineage))
    run_events = sort_metric_rows(enrich_rows_for_chain(run_events, lineage))

    chain_runs = discover_chain_runs(run_dir, str(lineage.get("chain_id", "")))
    chain_metrics: list[dict[str, Any]] = []
    chain_run_index: list[dict[str, Any]] = []
    for child, child_lineage in chain_runs:
        child_metrics = sort_metric_rows(enrich_rows_for_chain(read_jsonl(child / "metrics.jsonl"), child_lineage))
        chain_metrics.extend(child_metrics)
        chain_run_index.append(
            {
                "run_id": str(child_lineage.get("run_id", child.name)),
                "run_dir": str(child),
                "chain_id": str(child_lineage.get("chain_id", "")),
                "run_sequence_index": int(child_lineage.get("run_sequence_index", 0) or 0),
                "created_at_utc": str(child_lineage.get("created_at_utc", "")),
            }
        )
    chain_metrics = sort_metric_rows(chain_metrics)

    export_root = run_dir.parent.parent / "hf_generations_exports" / str(lineage.get("run_id", run_dir.name))
    run_export_dir = export_root / "runs" / str(lineage.get("run_id", run_dir.name))
    chain_export_dir = export_root / "chains" / str(lineage.get("chain_id", run_dir.name))
    ensure_dir(run_export_dir)
    ensure_dir(chain_export_dir)

    write_sorted_jsonl(run_generations, run_export_dir / "generations.jsonl")
    write_parquet(run_generations, run_export_dir / "generations.parquet")
    write_sorted_jsonl(run_metrics, run_export_dir / "metrics.jsonl")
    write_parquet(run_metrics, run_export_dir / "metrics.parquet")
    write_sorted_jsonl(run_events, run_export_dir / "checkpoint_events.jsonl")
    save_json(json_safe(lineage), run_export_dir / "lineage.json")
    for name in ["run_summary.json", "resume_decision.json", "effective_runtime_config.json", "dataset_summary.json"]:
        source = run_dir / name
        if source.exists():
            save_json(read_json(source), run_export_dir / name)

    write_sorted_jsonl(chain_metrics, chain_export_dir / "chain_metrics.jsonl")
    write_parquet(chain_metrics, chain_export_dir / "chain_metrics.parquet")
    save_json(json_safe(chain_run_index), chain_export_dir / "chain_runs.json")
    try:
        from plot_live_rewards import render_plots

        render_plots(run_metrics, run_export_dir, "reward_curves_run.png", "Run")
        render_plots(chain_metrics, chain_export_dir, "chain_reward_curves.png", "Chain")
    except Exception:
        pass

    api = HfApi(token=hf_token)
    create_repo(repo_id=generations_repo, repo_type="dataset", token=hf_token, exist_ok=True)

    readme_path = export_root / "README.md"
    readme_path.write_text(build_generations_dataset_readme(output_repo, generations_repo), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=generations_repo,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Update README for {lineage.get('run_id', run_dir.name)}",
    )
    api.upload_folder(
        folder_path=str(run_export_dir),
        path_in_repo=f"runs/{lineage.get('run_id', run_dir.name)}",
        repo_id=generations_repo,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Upload run artifacts for {lineage.get('run_id', run_dir.name)}",
    )
    api.upload_folder(
        folder_path=str(chain_export_dir),
        path_in_repo=f"chains/{lineage.get('chain_id', run_dir.name)}",
        repo_id=generations_repo,
        repo_type="dataset",
        token=hf_token,
        commit_message=f"Update chain artifacts for {lineage.get('chain_id', run_dir.name)}",
    )
    return {
        "enabled": True,
        "repo_id": generations_repo,
        "run_export_dir": str(run_export_dir),
        "chain_export_dir": str(chain_export_dir),
        "chain_run_count": len(chain_run_index),
        "run_generation_rows": len(run_generations),
        "run_metric_rows": len(run_metrics),
        "chain_metric_rows": len(chain_metrics),
    }


def build_runtime_summary(mode: str, dataset_summary: dict[str, Any], model_meta: dict[str, Any], cfg, run_dir: Path, lineage: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        **json_safe(lineage),
        "dataset_id": dataset_summary["dataset_id"],
        "dataset_source_id": dataset_summary["dataset_source_id"],
        "phase1_max_bayts": os.getenv("PHASE1_MAX_BAYTS", "").strip(),
        "allowed_meters": parse_allowed_meters_env(),
        "base_model_id": resolve_base_model_id(cfg),
        "sft_adapter_repo": resolve_start_adapter_repo(cfg),
        "sft_adapter_mode": resolve_start_adapter_mode(cfg),
        "reward_weights": cfg["phase1"]["reward_weights"],
        "model_meta": model_meta,
    }


def build_effective_runtime_config(mode: str, cfg, dataset_summary: dict[str, Any], args: GRPOConfig, reward_weight_map: dict[str, float], output_repo: str, lineage: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": mode,
        "lineage": json_safe(lineage),
        "dataset": {
            "dataset_id": dataset_summary["dataset_id"],
            "dataset_source_id": dataset_summary["dataset_source_id"],
            "train_split": dataset_summary["train_split"],
            "eval_split": dataset_summary["eval_split"],
            "test_split": dataset_summary["test_split"],
            "train_size": dataset_summary["train_size"],
            "eval_size": dataset_summary["eval_size"],
            "test_size": dataset_summary["test_size"],
            "hard_diagnostic_size": dataset_summary["hard_diagnostic_size"],
            "phase1_max_bayts": dataset_summary["phase1_max_bayts"],
            "allowed_meters": dataset_summary["allowed_meters"],
            "train_manifest_path": dataset_summary["train_manifest_path"],
            "hard_diagnostic_manifest_path": dataset_summary["hard_diagnostic_manifest_path"],
        },
        "rewards": {
            "active_rewards": list(cfg["phase1"]["active_rewards"]),
            "weights": reward_weight_map,
        },
        "model": {
            "base_model_id": resolve_base_model_id(cfg),
            "sft_adapter_repo": resolve_start_adapter_repo(cfg),
            "sft_adapter_mode": resolve_start_adapter_mode(cfg),
            "load_in_4bit_requested": bool(cfg["model"]["load_in_4bit"]),
            "gradient_checkpointing": bool(cfg["model"]["gradient_checkpointing"]),
        },
        "effective_generation": {
            "use_vllm": bool(args.use_vllm),
            "vllm_mode": str(args.vllm_mode),
            "vllm_gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
            "max_prompt_length": int(args.max_prompt_length),
            "max_completion_length": int(args.max_completion_length),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "num_generations": int(args.num_generations),
            "num_generations_eval": int(cfg["generation"]["num_generations_eval"]),
        },
        "effective_trainer": {
            "learning_rate": float(args.learning_rate),
            "per_device_train_batch_size": int(args.per_device_train_batch_size),
            "per_device_eval_batch_size_configured": int(cfg["trainer"]["per_device_eval_batch_size"]),
            "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "max_steps": int(args.max_steps),
            "logging_steps": int(args.logging_steps),
            "eval_steps": int(args.eval_steps),
            "save_steps": int(args.save_steps),
            "save_total_limit": int(args.save_total_limit),
            "beta": float(args.beta),
            "scale_rewards": str(args.scale_rewards),
            "loss_type": str(args.loss_type),
            "mask_truncated_completions": bool(args.mask_truncated_completions),
            "bf16": bool(args.bf16),
            "load_best_model_at_end": bool(args.load_best_model_at_end),
            "metric_for_best_model": str(args.metric_for_best_model),
            "greater_is_better": bool(args.greater_is_better),
            "hub_strategy": str(args.hub_strategy),
            "push_to_hub": bool(args.push_to_hub),
            "hub_model_id": output_repo,
            "generations_repo_id": resolve_generations_repo(output_repo),
        },
        "sanity_mode_note": (
            "When mode='sanity', these effective values override the base trainer/generation blocks "
            "from config_snapshot.json."
        ),
    }


def run_training(mode="train"):
    load_dotenv(ENV_PATH, override=False)
    cfg = load_cfg()
    run_dir = build_run_dir(cfg, mode)
    logger = setup_logger(f"train_grpo_{mode}", run_dir / cfg["logging"]["train_log_name"])

    logger.info("mode=%s", mode)
    output_repo = resolve_output_repo(cfg)

    train_dataset, eval_dataset, dataset_summary = build_datasets(cfg, mode)
    model, tokenizer, model_meta = build_model_and_tokenizer(cfg)
    fingerprint = create_config_fingerprint(cfg, dataset_summary["dataset_id"])
    resume_path, resume_decision, resume_manifest = resolve_resume_checkpoint(run_dir, dataset_summary, fingerprint, output_repo)
    lineage = build_lineage(run_dir, resume_decision, resume_manifest)
    save_snapshots(cfg, run_dir, dataset_summary, model_meta, fingerprint, lineage)

    recorder = GenerationRecorder(
        path=run_dir / cfg["logging"]["generations_jsonl_name"],
        reward_order=list(cfg["phase1"]["active_rewards"]),
        reward_weights={name: float(cfg["phase1"]["reward_weights"][name]) for name in cfg["phase1"]["active_rewards"]},
        run_meta=lineage,
    )
    reward_fns, reward_weights, reward_weight_map = build_reward_fns(cfg, run_dir, recorder)
    args = build_grpo_args(cfg, run_dir, mode, reward_weights, output_repo)
    save_json(
        build_effective_runtime_config(mode, cfg, dataset_summary, args, reward_weight_map, output_repo, lineage),
        run_dir / "effective_runtime_config.json",
    )

    runtime_summary = build_runtime_summary(mode, dataset_summary, model_meta, cfg, run_dir, lineage)
    save_json(resume_decision, run_dir / "resume_decision.json")

    metrics_callback = StructuredMetricsCallback(
        metrics_path=run_dir / cfg["logging"]["metrics_jsonl_name"],
        metrics_csv_path=run_dir / cfg["logging"]["metrics_csv_name"],
        logger=logger,
    )
    checkpoint_callback = CheckpointEventCallback(
        run_dir=run_dir,
        events_path=run_dir / cfg["logging"]["checkpoint_events_name"],
        fingerprint=fingerprint,
        runtime_summary=runtime_summary,
        output_repo=output_repo,
    )

    trainer = InstrumentedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[metrics_callback, checkpoint_callback],
        eval_num_generations=int(cfg["generation"]["num_generations_eval"]),
    )

    logger.info("dataset_id=%s train_size=%s eval_size=%s", dataset_summary["dataset_id"], len(train_dataset), len(eval_dataset))
    logger.info("source_dataset_id=%s test_size=%s hard_diagnostic_size=%s", dataset_summary["dataset_source_id"], dataset_summary["test_size"], dataset_summary["hard_diagnostic_size"])
    logger.info("active_rewards=%s", list(cfg["phase1"]["active_rewards"]))
    logger.info("reward_weights=%s", reward_weight_map)
    logger.info("output_repo=%s", output_repo)
    logger.info("train_manifest_path=%s", dataset_summary["train_manifest_path"])
    logger.info("hard_diagnostic_manifest_path=%s", dataset_summary["hard_diagnostic_manifest_path"])
    logger.info("trainable_params=%s total_params=%s trainable_ratio=%.6f", model_meta.get("trainable_params"), model_meta.get("total_params"), float(model_meta.get("trainable_ratio", 0.0)))
    logger.info("resume_decision=%s", json.dumps(resume_decision, ensure_ascii=False))
    logger.info("starting trainer.train()")

    train_result = trainer.train(resume_from_checkpoint=resume_path)
    save_json(json_safe(train_result.metrics), run_dir / "train_result_metrics.json")

    final_adapter_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)
    recorder.close()

    run_summary = {
        "timestamp_utc": utc_now_iso(),
        "mode": mode,
        **json_safe(lineage),
        "run_dir": str(run_dir),
        "output_repo": output_repo,
        "resume_decision": resume_decision,
        "best_model_checkpoint": json_safe(getattr(trainer.state, "best_model_checkpoint", None)),
        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
        "train_metrics": json_safe(train_result.metrics),
        "final_adapter_dir": str(final_adapter_dir),
    }
    save_json(run_summary, run_dir / "run_summary.json")

    try:
        generations_export = export_generations_to_hub(run_dir, output_repo, lineage)
    except Exception as exc:
        generations_export = {
            "enabled": False,
            "repo_id": resolve_generations_repo(output_repo),
            "reason": f"export_failed:{type(exc).__name__}",
            "error": str(exc),
        }
        logger.exception("export_generations_to_hub failed: %s", exc)
    save_json(json_safe(generations_export), run_dir / "generations_export_summary.json")

    if output_repo:
        try:
            logger.info("pushing final adapter to hub repo=%s", output_repo)
            trainer.push_to_hub(commit_message=f"Final adapter for {run_dir.name}")
        except Exception as exc:
            logger.exception("push_to_hub failed: %s", exc)

    if generations_export.get("enabled"):
        logger.info("pushed generations dataset repo=%s", generations_export.get("repo_id"))

    logger.info("done run_dir=%s", run_dir)
    return {
        "run_dir": str(run_dir),
        "resume_path": resume_path,
        "resume_decision": resume_decision,
        "lineage": lineage,
        "dataset_summary": dataset_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Shaer GRPO training or smoke sanity.")
    parser.add_argument("--mode", choices=["train", "sanity"], default=os.getenv("GRPO_MODE", "train"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(mode=args.mode)
