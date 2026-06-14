#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the best probe-meter checkpoint for a live SFT run.")
    parser.add_argument("--run-dir", required=True, help="Path to the run directory, e.g. outputs/train/train_YYYY...")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval while watching.")
    parser.add_argument("--watch", action="store_true", help="Keep watching for new best probe-meter checkpoints.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def pick_best_probe(rows: list[dict]) -> dict | None:
    probe_rows = [r for r in rows if r.get("mode") == "probe" and "probe_meter_mean" in r]
    if not probe_rows:
        return None
    return max(probe_rows, key=lambda r: (float(r["probe_meter_mean"]), int(r.get("global_step", -1))))


def hardlink_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_tree_with_links(src_dir: Path, dst_dir: Path) -> None:
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel = root_path.relative_to(src_dir)
        for name in dirs:
            (dst_dir / rel / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            hardlink_or_copy_file(root_path / name, dst_dir / rel / name)


def atomic_replace_dir(src_tmp_dir: Path, final_dir: Path) -> None:
    backup_dir = final_dir.with_name(final_dir.name + ".old")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.replace(backup_dir)
    src_tmp_dir.replace(final_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config_snapshot.json"
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def build_summary(run_dir: Path, run_cfg: dict, best_probe: dict, source_checkpoint: Path, kept_dir: Path) -> dict:
    run_name = run_dir.name
    namespace = str(run_cfg.get("continuation_namespace", "fresh_sft"))
    model_repo_id = str(run_cfg.get("model_repo_id", ""))
    step = int(best_probe["global_step"])
    return {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "selection_metric": "probe_meter_mean",
        "best_probe_meter_step": step,
        "best_probe_meter_mean": float(best_probe["probe_meter_mean"]),
        "best_probe_count_adherence_mean": float(best_probe.get("probe_count_adherence_mean", 0.0)),
        "best_probe_timestamp_utc": best_probe.get("timestamp_utc"),
        "source_checkpoint_dir": str(source_checkpoint),
        "kept_checkpoint_dir": str(kept_dir),
        "model_repo_id": model_repo_id,
        "remote_checkpoint_subfolder": (
            f"checkpoints/{namespace}/train/{run_name}/checkpoint-{step}" if model_repo_id else None
        ),
        "kept_checkpoint_present": kept_dir.exists(),
    }


def ensure_best_probe_checkpoint(run_dir: Path, logger: logging.Logger) -> bool:
    probe_path = run_dir / "probe_metrics.jsonl"
    probe_rows = load_jsonl(probe_path)
    best_probe = pick_best_probe(probe_rows)
    if best_probe is None:
        logger.info("No probe rows found yet in %s", probe_path)
        return False

    step = int(best_probe["global_step"])
    source_checkpoint = run_dir / f"checkpoint-{step}"
    kept_dir = run_dir / "best_probe_meter_checkpoint"
    summary_path = run_dir / "best_probe_meter_summary.json"
    state_path = run_dir / "best_probe_meter_keeper_state.json"
    run_cfg = load_run_config(run_dir)

    if not source_checkpoint.exists():
        logger.info(
            "Best probe-meter step=%s exists in metrics but local checkpoint directory is not present yet: %s",
            step,
            source_checkpoint,
        )
        summary = build_summary(run_dir, run_cfg, best_probe, source_checkpoint, kept_dir)
        summary["local_checkpoint_present"] = False
        write_json(summary_path, summary)
        return False

    previous_step = None
    if state_path.exists():
        try:
            previous_step = json.loads(state_path.read_text(encoding="utf-8")).get("best_probe_meter_step")
        except json.JSONDecodeError:
            previous_step = None

    if kept_dir.exists() and previous_step == step:
        logger.info("Best probe-meter checkpoint already protected at step=%s", step)
        return False

    tmp_parent = Path(tempfile.mkdtemp(prefix="best_probe_meter_", dir=str(run_dir)))
    tmp_dir = tmp_parent / kept_dir.name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    mirror_tree_with_links(source_checkpoint, tmp_dir)
    atomic_replace_dir(tmp_dir, kept_dir)
    shutil.rmtree(tmp_parent, ignore_errors=True)

    summary = build_summary(run_dir, run_cfg, best_probe, source_checkpoint, kept_dir)
    summary["local_checkpoint_present"] = True
    write_json(summary_path, summary)
    write_json(
        state_path,
        {
            "best_probe_meter_step": step,
            "best_probe_meter_mean": float(best_probe["probe_meter_mean"]),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    logger.info(
        "Protected best probe-meter checkpoint | step=%s meter_mean=%.4f source=%s kept=%s",
        step,
        float(best_probe["probe_meter_mean"]),
        source_checkpoint,
        kept_dir,
    )
    return True


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("best_probe_meter_keeper")

    stop_requested = False

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal stop_requested
        logger.info("Received signal %s, stopping.", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    ensure_best_probe_checkpoint(run_dir, logger)
    if not args.watch:
        return 0

    while not stop_requested:
        time.sleep(max(1.0, args.poll_seconds))
        try:
            ensure_best_probe_checkpoint(run_dir, logger)
        except Exception as exc:  # pragma: no cover
            logger.exception("Watcher iteration failed: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
