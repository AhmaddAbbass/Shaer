import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi, create_repo, hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DEFAULT_REPO_ID = "Shaer-AI/yehia-sft-qlora"
DEFAULT_DATASET_ID = "Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed"
DEFAULT_BASE_MODEL = "Navid-AI/Yehia-7B-preview"


@dataclass
class SFTConfig:
    mode: str
    repo_id: str
    dataset_id: str
    base_model_id: str
    resume: str
    resume_path: str
    run_name: str
    output_dir: Path
    seed: int
    push_to_hub: bool
    max_seq_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    max_steps: int | None
    save_steps: int | None
    eval_steps: int | None
    logging_steps: int | None
    save_total_limit: int | None
    bf16: str
    gradient_checkpointing: bool
    test_train_rows: int
    test_eval_rows: int
    log_dir: Path
    log_level: str
    heartbeat_seconds: int
    jsonl_metrics: bool
    jsonl_events: bool
    debug_dump_first_batch: bool
    remote_keep_last_checkpoints: int


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    value = v.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(run_dir: Path, level: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sft_qlora")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


class JsonlWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_config_fingerprint(cfg: SFTConfig) -> str:
    payload = {
        "mode": cfg.mode,
        "dataset_id": cfg.dataset_id,
        "base_model_id": cfg.base_model_id,
        "max_seq_length": cfg.max_seq_length,
        "lr": cfg.learning_rate,
        "ga": cfg.gradient_accumulation_steps,
        "lora": {
            "r": 64,
            "alpha": 128,
            "dropout": 0.05,
            "targets": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def resolve_mode_defaults(cfg: SFTConfig) -> None:
    if cfg.mode == "test":
        if cfg.max_steps is None:
            cfg.max_steps = 6
        if cfg.save_steps is None:
            cfg.save_steps = 2
        if cfg.eval_steps is None:
            cfg.eval_steps = 2
        if cfg.logging_steps is None:
            cfg.logging_steps = 1
        if cfg.save_total_limit is None:
            cfg.save_total_limit = 5
    else:
        if cfg.max_steps is None:
            cfg.max_steps = -1
        if cfg.save_steps is None:
            cfg.save_steps = 100
        if cfg.eval_steps is None:
            cfg.eval_steps = 100
        if cfg.logging_steps is None:
            cfg.logging_steps = 5
        if cfg.save_total_limit is None:
            cfg.save_total_limit = 3


def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="QLoRA SFT with auto-resume and HF checkpointing")
    parser.add_argument("--mode", choices=["test", "full"], required=True)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--base_model_id", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--resume", choices=["auto", "fresh", "from_path"], default="auto")
    parser.add_argument("--resume_path", default="")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--output_dir", default="outputs/sft_qlora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", type=str2bool, default=True)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--bf16", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument("--test_train_rows", type=int, default=256)
    parser.add_argument("--test_eval_rows", type=int, default=64)
    parser.add_argument("--log_dir", default="logs/sft")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--heartbeat_seconds", type=int, default=30)
    parser.add_argument("--jsonl_metrics", type=str2bool, default=True)
    parser.add_argument("--jsonl_events", type=str2bool, default=True)
    parser.add_argument("--debug_dump_first_batch", type=str2bool, default=True)
    parser.add_argument("--remote_keep_last_checkpoints", type=int, default=6)

    args = parser.parse_args()
    run_name = args.run_name.strip() or f"{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    cfg = SFTConfig(
        mode=args.mode,
        repo_id=args.repo_id,
        dataset_id=args.dataset_id,
        base_model_id=args.base_model_id,
        resume=args.resume,
        resume_path=args.resume_path,
        run_name=run_name,
        output_dir=Path(args.output_dir).resolve() / args.mode / run_name,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        test_train_rows=args.test_train_rows,
        test_eval_rows=args.test_eval_rows,
        log_dir=Path(args.log_dir).resolve() / args.mode / run_name,
        log_level=args.log_level,
        heartbeat_seconds=args.heartbeat_seconds,
        jsonl_metrics=args.jsonl_metrics,
        jsonl_events=args.jsonl_events,
        debug_dump_first_batch=args.debug_dump_first_batch,
        remote_keep_last_checkpoints=args.remote_keep_last_checkpoints,
    )

    if cfg.resume == "from_path" and not cfg.resume_path.strip():
        raise ValueError("--resume_path is required when --resume from_path")

    resolve_mode_defaults(cfg)
    return cfg


def detect_precision(cfg: SFTConfig) -> tuple[bool, bool]:
    if cfg.bf16 == "true":
        return True, False
    if cfg.bf16 == "false":
        return False, True
    bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if bf16_ok:
        return True, False
    return False, True


def ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    create_repo(repo_id=repo_id, repo_type="model", token=token, exist_ok=True)


def read_remote_json(api: HfApi, repo_id: str, path_in_repo: str, token: str, cache_dir: Path) -> dict[str, Any] | None:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=path_in_repo,
            token=token,
            cache_dir=str(cache_dir),
        )
    except (EntryNotFoundError, HfHubHTTPError):
        return None
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_json(api: HfApi, repo_id: str, path_in_repo: str, payload: dict[str, Any], token: str) -> None:
    blob = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    api.upload_file(
        path_or_fileobj=blob,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )


def find_latest_checkpoint_prefix(api: HfApi, repo_id: str, mode: str, token: str) -> str | None:
    info = api.model_info(repo_id=repo_id, token=token)
    pat = re.compile(rf"^checkpoints/{mode}/([^/]+)/checkpoint-(\d+)/adapter_config\.json$")
    best_step = -1
    best_prefix = None
    for sib in info.siblings or []:
        m = pat.match(sib.rfilename)
        if not m:
            continue
        step = int(m.group(2))
        prefix = f"checkpoints/{mode}/{m.group(1)}/checkpoint-{step}"
        if step > best_step:
            best_step = step
            best_prefix = prefix
    return best_prefix


def download_checkpoint_prefix(repo_id: str, prefix: str, token: str, cache_dir: Path) -> str:
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns=[f"{prefix}/*"],
        cache_dir=str(cache_dir),
    )
    local_path = Path(snapshot_dir) / prefix
    if not local_path.exists():
        raise RuntimeError(f"Checkpoint prefix downloaded but missing local path: {local_path}")
    return str(local_path)


def load_dataset_resilient(dataset_id: str, token: str, logger: logging.Logger, cache_dir: Path) -> Dataset:
    required_cols = ["sft_prompt", "sft_full_text", "base_meter"]
    try:
        ds = load_dataset(dataset_id, split="train", cache_dir=str(cache_dir))
        for c in required_cols:
            if c not in ds.column_names:
                raise RuntimeError(f"Required column missing in dataset: {c}")
        logger.info("Loaded dataset via datasets.load_dataset (%d rows)", len(ds))
        return ds
    except Exception as e:
        logger.warning("load_dataset failed (%s). Falling back to parquet direct load.", e)

    parquet_files = [f for f in list_repo_files(repo_id=dataset_id, repo_type="dataset", token=token) if f.endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for dataset: {dataset_id}")

    frames = []
    for f in sorted(parquet_files):
        local = hf_hub_download(
            repo_id=dataset_id,
            repo_type="dataset",
            filename=f,
            token=token,
            cache_dir=str(cache_dir),
        )
        df = pd.read_parquet(local, columns=required_cols)
        frames.append(df)
        logger.info("Loaded parquet chunk %s with %d rows", f, len(df))
    full = pd.concat(frames, ignore_index=True)
    ds = Dataset.from_pandas(full, preserve_index=False)
    logger.info("Loaded dataset via parquet fallback (%d rows)", len(ds))
    return ds


def build_train_eval_splits(ds: Dataset, cfg: SFTConfig, logger: logging.Logger) -> tuple[Dataset, Dataset]:
    if cfg.mode == "test":
        total_needed = cfg.test_train_rows + cfg.test_eval_rows
        sample_n = min(total_needed, len(ds))
        if sample_n < 4:
            raise RuntimeError("Dataset too small for test mode sampling")
        dss = ds.shuffle(seed=cfg.seed).select(range(sample_n))
        train_n = min(cfg.test_train_rows, sample_n - 1)
        eval_n = min(cfg.test_eval_rows, sample_n - train_n)
        if eval_n <= 0:
            eval_n = 1
            train_n = sample_n - 1
        train_ds = dss.select(range(train_n))
        eval_ds = dss.select(range(train_n, train_n + eval_n))
        logger.info("Test split built: train=%d eval=%d", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    eval_n = max(1, math.ceil(len(ds) * 0.02))
    if "base_meter" not in ds.column_names:
        split = ds.train_test_split(test_size=eval_n, seed=cfg.seed, shuffle=True)
        train_ds = split["train"]
        eval_ds = split["test"]
        logger.info("Full split built (non-stratified fallback): train=%d eval=%d", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    meter_to_indices: dict[str, list[int]] = {}
    for idx, meter in enumerate(ds["base_meter"]):
        key = str(meter)
        meter_to_indices.setdefault(key, []).append(idx)

    rng = random.Random(cfg.seed)
    for idxs in meter_to_indices.values():
        rng.shuffle(idxs)

    # Ensure eval sees at least a few examples from each meter while preserving total eval size.
    min_per_meter = 2
    eval_indices: list[int] = []
    leftovers: list[int] = []
    meter_eval_counts: dict[str, int] = {}
    for meter, idxs in meter_to_indices.items():
        keep_for_train = max(1, len(idxs) - min_per_meter)
        take = max(0, len(idxs) - keep_for_train)
        chosen = idxs[:take]
        eval_indices.extend(chosen)
        leftovers.extend(idxs[take:])
        meter_eval_counts[meter] = len(chosen)

    if len(eval_indices) > eval_n:
        rng.shuffle(eval_indices)
        selected = set(eval_indices[:eval_n])
        eval_indices = sorted(selected)
    else:
        remaining = eval_n - len(eval_indices)
        if remaining > 0:
            rng.shuffle(leftovers)
            eval_indices.extend(leftovers[:remaining])

    eval_set = set(eval_indices)
    train_indices = [i for i in range(len(ds)) if i not in eval_set]
    train_ds = ds.select(train_indices)
    eval_ds = ds.select(sorted(eval_indices))
    eval_meters = set(eval_ds["base_meter"])
    logger.info(
        "Full split built (meter-aware): train=%d eval=%d unique_eval_meters=%d",
        len(train_ds),
        len(eval_ds),
        len(eval_meters),
    )
    return train_ds, eval_ds


def preprocess_dataset(ds: Dataset, tokenizer: AutoTokenizer, cfg: SFTConfig, logger: logging.Logger, desc: str) -> Dataset:
    def fn(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        out_input_ids = []
        out_attn = []
        out_labels = []
        out_prompt_lens = []
        out_supervised = []

        prompts = batch["sft_prompt"]
        full_texts = batch["sft_full_text"]

        for prompt, full_text in zip(prompts, full_texts):
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

            full_ids = full_ids[: cfg.max_seq_length]
            labels = full_ids.copy()
            prompt_len = min(len(prompt_ids), len(labels))
            for i in range(prompt_len):
                labels[i] = -100
            supervised = sum(1 for x in labels if x != -100)

            out_input_ids.append(full_ids)
            out_attn.append([1] * len(full_ids))
            out_labels.append(labels)
            out_prompt_lens.append(prompt_len)
            out_supervised.append(supervised)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
            "prompt_len": out_prompt_lens,
            "supervised_tokens": out_supervised,
        }

    mapped = ds.map(fn, batched=True, desc=desc, remove_columns=ds.column_names)
    filtered = mapped.filter(lambda ex: ex["supervised_tokens"] > 0)
    logger.info("%s rows after preprocessing/filtering: %d", desc, len(filtered))
    return filtered


class Seq2SeqLikeCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            l = len(f["input_ids"])
            pad = max_len - l
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StructuredMetricsCallback(TrainerCallback):
    def __init__(self, metrics_writer: JsonlWriter, events_writer: JsonlWriter, logger: logging.Logger, heartbeat_seconds: int):
        self.metrics_writer = metrics_writer
        self.events_writer = events_writer
        self.logger = logger
        self.heartbeat_seconds = heartbeat_seconds
        self.last_heartbeat = time.time()
        self.metric_rows: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logs = logs or {}
        row = {
            "timestamp_utc": utc_now_iso(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()},
        }
        self.metric_rows.append(row)
        self.metrics_writer.write(row)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        now = time.time()
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            self.last_heartbeat = now
            payload = {
                "timestamp_utc": utc_now_iso(),
                "event_type": "heartbeat",
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }
            self.events_writer.write(payload)
            self.logger.info("Heartbeat | step=%s epoch=%s", payload["global_step"], payload["epoch"])


class HubCheckpointCallback(TrainerCallback):
    def __init__(
        self,
        api: HfApi,
        token: str,
        cfg: SFTConfig,
        logger: logging.Logger,
        events_writer: JsonlWriter,
        fingerprint: str,
    ):
        self.api = api
        self.token = token
        self.cfg = cfg
        self.logger = logger
        self.events_writer = events_writer
        self.fingerprint = fingerprint
        self.latest_checkpoint_repo_path: str | None = None
        self.history: list[dict[str, Any]] = []

    def checkpoint_repo_path(self, step: int) -> str:
        return f"checkpoints/{self.cfg.mode}/{self.cfg.run_name}/checkpoint-{step}"

    def latest_manifest_path(self) -> str:
        return f"manifests/{self.cfg.mode}/latest.json"

    def history_manifest_path(self) -> str:
        return f"manifests/{self.cfg.mode}/history/{self.cfg.run_name}.json"

    def adapters_latest_path(self) -> str:
        return f"adapters/{self.cfg.mode}/latest"

    def _write_manifests(self, step: int, checkpoint_path_in_repo: str, adapter_path_in_repo: str | None = None) -> None:
        payload = {
            "repo_id": self.cfg.repo_id,
            "mode": self.cfg.mode,
            "run_name": self.cfg.run_name,
            "base_model_id": self.cfg.base_model_id,
            "dataset_id": self.cfg.dataset_id,
            "global_step": int(step),
            "checkpoint_path_in_repo": checkpoint_path_in_repo,
            "adapter_latest_path_in_repo": adapter_path_in_repo,
            "created_at_utc": utc_now_iso(),
            "config_fingerprint": self.fingerprint,
            "train_args": {
                "max_seq_length": self.cfg.max_seq_length,
                "learning_rate": self.cfg.learning_rate,
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "max_steps": self.cfg.max_steps,
                "save_steps": self.cfg.save_steps,
                "eval_steps": self.cfg.eval_steps,
                "logging_steps": self.cfg.logging_steps,
            },
        }
        upload_json(self.api, self.cfg.repo_id, self.latest_manifest_path(), payload, self.token)
        self.history.append(payload)
        upload_json(
            self.api,
            self.cfg.repo_id,
            self.history_manifest_path(),
            {"items": self.history},
            self.token,
        )

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if not self.cfg.push_to_hub:
            return

        step = int(state.global_step)
        local_ckpt = Path(args.output_dir) / f"checkpoint-{step}"
        if not local_ckpt.exists():
            self.logger.warning("Expected checkpoint missing on save: %s", local_ckpt)
            return

        repo_path = self.checkpoint_repo_path(step)
        t0 = time.time()
        self.api.upload_folder(
            folder_path=str(local_ckpt),
            repo_id=self.cfg.repo_id,
            repo_type="model",
            token=self.token,
            path_in_repo=repo_path,
        )
        elapsed = int((time.time() - t0) * 1000)

        self.latest_checkpoint_repo_path = repo_path
        self._write_manifests(step=step, checkpoint_path_in_repo=repo_path)
        self._prune_old_remote_checkpoints()

        evt = {
            "timestamp_utc": utc_now_iso(),
            "event_type": "checkpoint_uploaded",
            "global_step": step,
            "hf_path": repo_path,
            "duration_ms": elapsed,
            "status": "ok",
        }
        self.events_writer.write(evt)
        self.logger.info("Uploaded checkpoint step=%d to %s (%d ms)", step, repo_path, elapsed)

    def _prune_old_remote_checkpoints(self) -> None:
        keep_n = max(0, int(self.cfg.remote_keep_last_checkpoints))
        if keep_n <= 0:
            return

        pattern = re.compile(
            rf"^checkpoints/{re.escape(self.cfg.mode)}/{re.escape(self.cfg.run_name)}/checkpoint-(\d+)/.+$"
        )
        grouped: dict[int, list[str]] = {}
        for f in list_repo_files(repo_id=self.cfg.repo_id, repo_type="model", token=self.token):
            m = pattern.match(f)
            if not m:
                continue
            step = int(m.group(1))
            grouped.setdefault(step, []).append(f)

        if len(grouped) <= keep_n:
            return

        sorted_steps = sorted(grouped.keys())
        delete_steps = sorted_steps[: len(sorted_steps) - keep_n]
        delete_files = [f for s in delete_steps for f in grouped[s]]
        if not delete_files:
            return

        ops = [CommitOperationDelete(path_in_repo=f) for f in delete_files]
        self.api.create_commit(
            repo_id=self.cfg.repo_id,
            repo_type="model",
            token=self.token,
            operations=ops,
            commit_message=(
                f"Prune old checkpoints for {self.cfg.mode}/{self.cfg.run_name}: keep last {keep_n}"
            ),
        )
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "checkpoint_pruned",
                "mode": self.cfg.mode,
                "run_name": self.cfg.run_name,
                "keep_last": keep_n,
                "deleted_steps": delete_steps,
                "deleted_files_count": len(delete_files),
            }
        )
        self.logger.info(
            "Pruned remote checkpoints for run=%s. kept_last=%d deleted_steps=%s deleted_files=%d",
            self.cfg.run_name,
            keep_n,
            delete_steps,
            len(delete_files),
        )

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if not self.cfg.push_to_hub:
            return

        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            self.logger.warning("Missing model/tokenizer in on_train_end callback; skipping adapter upload")
            return

        export_dir = Path(args.output_dir) / "_adapter_latest_export"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)

        repo_path = self.adapters_latest_path()
        t0 = time.time()
        self.api.upload_folder(
            folder_path=str(export_dir),
            repo_id=self.cfg.repo_id,
            repo_type="model",
            token=self.token,
            path_in_repo=repo_path,
        )
        elapsed = int((time.time() - t0) * 1000)

        step = int(state.global_step)
        checkpoint_path = self.latest_checkpoint_repo_path or self.checkpoint_repo_path(step)
        self._write_manifests(step=step, checkpoint_path_in_repo=checkpoint_path, adapter_path_in_repo=repo_path)

        evt = {
            "timestamp_utc": utc_now_iso(),
            "event_type": "adapter_latest_uploaded",
            "global_step": step,
            "hf_path": repo_path,
            "duration_ms": elapsed,
            "status": "ok",
        }
        self.events_writer.write(evt)
        self.logger.info("Uploaded adapter latest to %s (%d ms)", repo_path, elapsed)



def resolve_resume_checkpoint(
    cfg: SFTConfig,
    api: HfApi,
    token: str,
    logger: logging.Logger,
    events_writer: JsonlWriter,
    fingerprint: str,
    cache_dir: Path,
) -> tuple[str | None, dict[str, Any]]:
    decision = {
        "timestamp_utc": utc_now_iso(),
        "mode": cfg.mode,
        "resume": cfg.resume,
        "result": "fresh",
        "reason": "",
        "checkpoint_repo_path": None,
        "local_resume_path": None,
    }

    if cfg.resume == "fresh":
        decision["reason"] = "explicit_fresh"
        events_writer.write({"event_type": "resume_decision", **decision})
        return None, decision

    if cfg.resume == "from_path":
        rp = cfg.resume_path.strip()
        if os.path.isdir(rp):
            decision["result"] = "resume"
            decision["reason"] = "explicit_local_path"
            decision["local_resume_path"] = rp
            events_writer.write({"event_type": "resume_decision", **decision})
            return rp, decision

        local = download_checkpoint_prefix(cfg.repo_id, rp, token, cache_dir)
        decision["result"] = "resume"
        decision["reason"] = "explicit_remote_prefix"
        decision["checkpoint_repo_path"] = rp
        decision["local_resume_path"] = local
        events_writer.write({"event_type": "resume_decision", **decision})
        return local, decision

    manifest_path = f"manifests/{cfg.mode}/latest.json"
    manifest = read_remote_json(api, cfg.repo_id, manifest_path, token, cache_dir)
    if manifest:
        compat = (
            manifest.get("mode") == cfg.mode
            and manifest.get("base_model_id") == cfg.base_model_id
            and manifest.get("dataset_id") == cfg.dataset_id
            and manifest.get("config_fingerprint") == fingerprint
            and manifest.get("checkpoint_path_in_repo")
        )
        if compat:
            ckpt_prefix = manifest["checkpoint_path_in_repo"]
            local = download_checkpoint_prefix(cfg.repo_id, ckpt_prefix, token, cache_dir)
            decision["result"] = "resume"
            decision["reason"] = "manifest_latest_compatible"
            decision["checkpoint_repo_path"] = ckpt_prefix
            decision["local_resume_path"] = local
            events_writer.write({"event_type": "resume_decision", **decision})
            return local, decision

        decision["reason"] = "manifest_incompatible"

    fallback = find_latest_checkpoint_prefix(api, cfg.repo_id, cfg.mode, token)
    if fallback:
        local = download_checkpoint_prefix(cfg.repo_id, fallback, token, cache_dir)
        decision["result"] = "resume"
        decision["reason"] = "fallback_latest_checkpoint_scan"
        decision["checkpoint_repo_path"] = fallback
        decision["local_resume_path"] = local
        events_writer.write({"event_type": "resume_decision", **decision})
        return local, decision

    decision["reason"] = decision["reason"] or "no_checkpoint_found"
    events_writer.write({"event_type": "resume_decision", **decision})
    return None, decision


def align_resume_trainer_state(resume_path: str | None, cfg: SFTConfig, logger: logging.Logger) -> None:
    if not resume_path:
        return
    state_path = Path(resume_path) / "trainer_state.json"
    if not state_path.exists():
        return
    try:
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        logger.warning("Could not read trainer_state.json for alignment (%s): %s", state_path, e)
        return

    changed = False
    desired: dict[str, int | None] = {
        "eval_steps": cfg.eval_steps,
        "save_steps": cfg.save_steps,
        "logging_steps": cfg.logging_steps,
    }
    for k, v in desired.items():
        if v is None:
            continue
        if state.get(k) != v:
            state[k] = v
            changed = True

    if not changed:
        return
    try:
        with state_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info(
            "Aligned resume trainer_state scheduling at %s (eval_steps=%s save_steps=%s logging_steps=%s)",
            state_path,
            cfg.eval_steps,
            cfg.save_steps,
            cfg.logging_steps,
        )
    except Exception as e:
        logger.warning("Could not write aligned trainer_state.json (%s): %s", state_path, e)


def first_batch_debug(train_ds: Dataset, collator: Seq2SeqLikeCollator, out_path: Path, logger: logging.Logger) -> dict[str, Any]:
    sample_n = min(4, len(train_ds))
    batch = collator([train_ds[i] for i in range(sample_n)])
    labels = batch["labels"]
    mask = labels.eq(-100)
    supervised = labels.ne(-100)
    payload = {
        "sample_n": sample_n,
        "shape_input_ids": list(batch["input_ids"].shape),
        "shape_labels": list(batch["labels"].shape),
        "masked_count": int(mask.sum().item()),
        "supervised_count": int(supervised.sum().item()),
        "has_masked": bool(mask.any().item()),
        "has_supervised": bool(supervised.any().item()),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("First batch debug: %s", payload)
    if not payload["has_masked"] or not payload["has_supervised"]:
        raise RuntimeError("Masking sanity failed: expected both masked and supervised tokens")
    return payload


def build_model_and_tokenizer(cfg: SFTConfig, logger: logging.Logger, bf16: bool, fp16: bool, cache_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id, trust_remote_code=True, use_fast=False, cache_dir=str(cache_dir)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    logger.info("Loading base model with 4-bit QLoRA setup...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer, bnb_cfg


def is_finite_number(x: Any) -> bool:
    if x is None:
        return False
    try:
        f = float(x)
    except Exception:
        return False
    return math.isfinite(f)


def validate_test_assertions(
    cfg: SFTConfig,
    metrics_rows: list[dict[str, Any]],
    weight_delta: float,
    output_dir: Path,
    callback: HubCheckpointCallback,
    api: HfApi,
    token: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    checks = {}

    losses = [row.get("loss") for row in metrics_rows if "loss" in row]
    eval_losses = [row.get("eval_loss") for row in metrics_rows if "eval_loss" in row]

    checks["train_loss_all_finite"] = bool(losses) and all(is_finite_number(x) for x in losses)
    checks["eval_loss_all_finite"] = bool(eval_losses) and all(is_finite_number(x) for x in eval_losses)
    checks["adapter_weights_changed"] = bool(weight_delta > 0)

    expected = [2, 4, 6]
    checks["local_checkpoints_exist"] = all((output_dir / f"checkpoint-{s}").exists() for s in expected)

    siblings = {f for f in list_repo_files(repo_id=cfg.repo_id, repo_type="model", token=token)}
    checks["remote_checkpoint_files_exist"] = all(
        f"checkpoints/{cfg.mode}/{cfg.run_name}/checkpoint-{s}/adapter_config.json" in siblings for s in expected
    )
    checks["remote_adapter_latest_exists"] = f"adapters/{cfg.mode}/latest/adapter_config.json" in siblings
    checks["manifest_latest_exists"] = f"manifests/{cfg.mode}/latest.json" in siblings

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Test assertions failed: {failed}")

    logger.info("All test assertions passed.")
    return checks


def reload_and_forward_check(
    cfg: SFTConfig,
    token: str,
    bnb_cfg: BitsAndBytesConfig,
    checkpoint_prefix: str,
    logger: logging.Logger,
    cache_dir: Path,
) -> dict[str, Any]:
    text = "اختبار سريع للشعر"
    result = {}

    tok = AutoTokenizer.from_pretrained(
        cfg.base_model_id, trust_remote_code=True, use_fast=False, cache_dir=str(cache_dir)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    logger.info("Reload check: adapters/%s/latest", cfg.mode)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    m = PeftModel.from_pretrained(base, cfg.repo_id, subfolder=f"adapters/{cfg.mode}/latest", token=token)
    device = next(m.parameters()).device
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = m(**inputs, labels=inputs["input_ids"])
    result["adapter_latest_loss_finite"] = bool(torch.isfinite(out.loss).item())
    del m
    del base
    torch.cuda.empty_cache()

    logger.info("Reload check: %s", checkpoint_prefix)
    base2 = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    m2 = PeftModel.from_pretrained(base2, cfg.repo_id, subfolder=checkpoint_prefix, token=token)
    device2 = next(m2.parameters()).device
    inputs2 = tok(text, return_tensors="pt")
    inputs2 = {k: v.to(device2) for k, v in inputs2.items()}
    with torch.no_grad():
        out2 = m2(**inputs2, labels=inputs2["input_ids"])
    result["checkpoint_reload_loss_finite"] = bool(torch.isfinite(out2.loss).item())
    del m2
    del base2
    torch.cuda.empty_cache()

    if not result["adapter_latest_loss_finite"] or not result["checkpoint_reload_loss_finite"]:
        raise RuntimeError(f"Reload forward checks failed: {result}")

    return result


def main() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)

    cfg = parse_args()
    set_seed(cfg.seed)

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")

    hf_home = Path((os.getenv("HF_HOME", "/workspace/.hf_cache").strip() or "/workspace/.hf_cache")).resolve()
    hub_cache = Path((os.getenv("HF_HUB_CACHE", str(hf_home / "hub")).strip() or str(hf_home / "hub"))).resolve()
    hf_home.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_cache)
    cache_dir = hub_cache

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(cfg.log_dir, cfg.log_level)
    metrics_writer = JsonlWriter(cfg.log_dir / "metrics.jsonl", enabled=cfg.jsonl_metrics)
    events_writer = JsonlWriter(cfg.log_dir / "events.jsonl", enabled=cfg.jsonl_events)

    logger.info("Starting SFT QLoRA run | mode=%s run_name=%s", cfg.mode, cfg.run_name)
    logger.info("Output dir: %s", cfg.output_dir)
    logger.info("Log dir: %s", cfg.log_dir)
    logger.info("HF cache dir: %s", cache_dir)

    config_payload = asdict(cfg)
    config_payload["output_dir"] = str(cfg.output_dir)
    config_payload["log_dir"] = str(cfg.log_dir)
    with (cfg.log_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    api = HfApi(token=hf_token)
    ensure_repo(api, cfg.repo_id, hf_token)

    fingerprint = create_config_fingerprint(cfg)
    logger.info("Config fingerprint: %s", fingerprint)

    resume_path, resume_decision = resolve_resume_checkpoint(
        cfg, api, hf_token, logger, events_writer, fingerprint, cache_dir
    )
    with (cfg.log_dir / "resume_decision.json").open("w", encoding="utf-8") as f:
        json.dump(resume_decision, f, ensure_ascii=False, indent=2)
    logger.info("Resume decision: %s", resume_decision)
    align_resume_trainer_state(resume_path, cfg, logger)

    ds = load_dataset_resilient(cfg.dataset_id, hf_token, logger, cache_dir)
    train_raw, eval_raw = build_train_eval_splits(ds, cfg, logger)

    bf16, fp16 = detect_precision(cfg)
    logger.info("Precision selected: bf16=%s fp16=%s", bf16, fp16)

    model, tokenizer, bnb_cfg = build_model_and_tokenizer(cfg, logger, bf16, fp16, cache_dir)

    train_ds = preprocess_dataset(train_raw, tokenizer, cfg, logger, desc="train_preprocess")
    eval_ds = preprocess_dataset(eval_raw, tokenizer, cfg, logger, desc="eval_preprocess")

    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError("Preprocessed train/eval datasets are empty")

    collator = Seq2SeqLikeCollator(tokenizer.pad_token_id)
    if cfg.debug_dump_first_batch:
        first_batch_debug(train_ds, collator, cfg.log_dir / "first_batch_debug.json", logger)

    tracked_name = None
    tracked_before = None
    for n, p in model.named_parameters():
        if p.requires_grad:
            tracked_name = n
            tracked_before = p.detach().float().cpu().clone()
            break
    if tracked_name is None:
        raise RuntimeError("No trainable parameters found")
    logger.info("Tracking trainable tensor for update check: %s", tracked_name)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        run_name=cfg.run_name,
        seed=cfg.seed,
    )

    metrics_cb = StructuredMetricsCallback(metrics_writer, events_writer, logger, cfg.heartbeat_seconds)
    hub_cb = HubCheckpointCallback(api, hf_token, cfg, logger, events_writer, fingerprint)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[metrics_cb, hub_cb],
    )

    logger.info("Running initial eval before training...")
    pre_eval = trainer.evaluate()
    metrics_writer.write({"timestamp_utc": utc_now_iso(), "phase": "pre_train_eval", **{k: float(v) for k, v in pre_eval.items() if isinstance(v, (int, float))}})

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    logger.info("Training finished. global_step=%s", trainer.state.global_step)

    final_eval = trainer.evaluate()
    logger.info("Final eval: %s", final_eval)

    tracked_after = dict(model.named_parameters())[tracked_name].detach().float().cpu()
    weight_delta = float(torch.norm(tracked_after - tracked_before).item())
    logger.info("Tracked tensor delta norm: %.8f", weight_delta)

    test_asserts = {}
    reload_checks = {}
    if cfg.mode == "test":
        test_asserts = validate_test_assertions(
            cfg=cfg,
            metrics_rows=metrics_cb.metric_rows,
            weight_delta=weight_delta,
            output_dir=cfg.output_dir,
            callback=hub_cb,
            api=api,
            token=hf_token,
            logger=logger,
        )

        ckpt_prefix = hub_cb.latest_checkpoint_repo_path
        if not ckpt_prefix:
            raise RuntimeError("No latest checkpoint prefix found after test run")

        del trainer
        del model
        torch.cuda.empty_cache()

        reload_checks = reload_and_forward_check(
            cfg=cfg,
            token=hf_token,
            bnb_cfg=bnb_cfg,
            checkpoint_prefix=ckpt_prefix,
            logger=logger,
            cache_dir=cache_dir,
        )

    summary = {
        "timestamp_utc": utc_now_iso(),
        "mode": cfg.mode,
        "repo_id": cfg.repo_id,
        "run_name": cfg.run_name,
        "output_dir": str(cfg.output_dir),
        "log_dir": str(cfg.log_dir),
        "resume_decision": resume_decision,
        "train_global_step": int(trainer.state.global_step if 'trainer' in locals() else cfg.max_steps),
        "weight_delta_norm": weight_delta,
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "final_eval": {k: float(v) if isinstance(v, (int, float)) else v for k, v in final_eval.items()},
        "latest_checkpoint_repo_path": hub_cb.latest_checkpoint_repo_path,
        "adapter_latest_repo_path": f"adapters/{cfg.mode}/latest",
        "test_assertions": test_asserts,
        "reload_checks": reload_checks,
    }
    with (cfg.log_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    events_writer.write({"timestamp_utc": utc_now_iso(), "event_type": "run_complete", "mode": cfg.mode, "run_name": cfg.run_name, "status": "ok"})
    logger.info("Run completed successfully.")
    logger.info("Summary written to %s", cfg.log_dir / "run_summary.json")


if __name__ == "__main__":
    main()
