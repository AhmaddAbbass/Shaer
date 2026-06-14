#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from preprocess_meter_count_common import (
    append_jsonl,
    atomic_write_json,
    difficulty_rule_spec,
    load_dotenv_if_present,
    load_run_config,
    read_json_or_none,
    status_path,
    summarize_scores,
    utc_now_iso,
)
from preprocess_meter_count_status import summarize_run
from watcher import load_state, save_state, send_email, state_path_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send milestone email updates for a meter/count preprocess run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--every-rows", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--full-train-rows", type=int, default=109070)
    parser.add_argument("--stale-minutes", type=float, default=0.0)
    return parser.parse_args()


def event_path(run_dir: Path) -> Path:
    return run_dir / "events" / "email_watcher.jsonl"


def load_watcher_state(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = state_path_for(f"{run_dir.resolve()}::meter_count_email_watcher")
    state = load_state(path)
    state.setdefault("created_at", utc_now_iso())
    state.setdefault("last_sent_milestone", 0)
    state.setdefault("last_sent_scored_rows", 0)
    state.setdefault("last_progress_scored_rows", 0)
    state.setdefault("last_progress_at", utc_now_iso())
    state.setdefault("last_difficulty_counts", {})
    state.setdefault("startup_email_sent", False)
    state.setdefault("stale_alert_active", False)
    state.setdefault("run_complete_email_sent", False)
    state.setdefault("publish_complete_email_sent", False)
    return path, state


def live_pid_summary(run_dir: Path) -> dict[str, int]:
    counts = {"generators": 0, "score_workers": 0, "checkpoint_publishers": 0, "email_watchers": 0, "other": 0}
    pids_dir = run_dir / "pids"
    if not pids_dir.exists():
        return counts
    for pid_file in sorted(pids_dir.glob("*.pid")):
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        alive = True
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
        if not alive:
            continue
        stem = pid_file.stem
        if stem.startswith("gen"):
            counts["generators"] += 1
        elif stem.startswith("score"):
            counts["score_workers"] += 1
        elif "checkpoint_publisher" in stem:
            counts["checkpoint_publishers"] += 1
        elif "email_watcher" in stem:
            counts["email_watchers"] += 1
        else:
            counts["other"] += 1
    return counts


def checkpoint_summary(run_dir: Path) -> dict[str, Any]:
    return read_json_or_none(status_path(run_dir, "checkpoint_publisher")) or {}


def failure_summary(run_dir: Path) -> tuple[int, list[str]]:
    failures_dir = run_dir / "failures"
    if not failures_dir.exists():
        return 0, []
    total = 0
    recent: list[str] = []
    for path in sorted(failures_dir.glob("*.jsonl")):
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        except Exception:
            continue
        total += len(lines)
        for line in lines[-3:]:
            recent.append(f"{path.name}: {line[:400]}")
    return total, recent[-5:]


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    hours = float(seconds) / 3600.0
    if hours < 1.0:
        return f"{seconds / 60.0:.1f}m"
    if hours < 48.0:
        return f"{hours:.2f}h"
    return f"{hours / 24.0:.2f}d"


def difficulty_delta(current: dict[str, int], previous: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in sorted(set(current) | set(previous)):
        out[key] = int(current.get(key, 0)) - int(previous.get(key, 0))
    return out


def row_meter_mean_stats(run_dir: Path) -> dict[str, Any]:
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    by_difficulty_meter: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((run_dir / "scored").glob("*.json")):
        row = read_json_or_none(path)
        if row is None:
            continue
        difficulty = str(row.get("difficulty") or "unknown")
        meter = str(row.get("base_meter") or row.get("meter_label") or "unknown")
        try:
            value = float(row.get("mean_meter_score") or 0.0)
        except Exception:
            value = 0.0
        by_difficulty[difficulty].append(value)
        by_difficulty_meter[difficulty][meter].append(value)

    diff_stats: dict[str, dict[str, float | int]] = {}
    for difficulty, values in sorted(by_difficulty.items()):
        summary = summarize_scores(values)
        diff_stats[difficulty] = {
            "rows": int(len(values)),
            "mean": float(summary["mean"]),
            "min": float(summary["min"]),
            "max": float(summary["max"]),
            "std": float(summary["std"]),
        }

    meter_stats: dict[str, dict[str, dict[str, float | int]]] = {}
    for difficulty, meter_map in sorted(by_difficulty_meter.items()):
        meter_stats[difficulty] = {}
        for meter, values in sorted(meter_map.items()):
            summary = summarize_scores(values)
            meter_stats[difficulty][meter] = {
                "rows": int(len(values)),
                "mean": float(summary["mean"]),
                "min": float(summary["min"]),
                "max": float(summary["max"]),
                "std": float(summary["std"]),
            }
    return {
        "by_difficulty": diff_stats,
        "by_difficulty_meter": meter_stats,
    }


def render_difficulty_rule(num_candidates: int) -> list[str]:
    rule = difficulty_rule_spec(num_candidates)
    easy_n = int(rule["easy_min_strong_exact_count"])
    hard_n = int(rule["hard_max_strong_meter_count"])
    return [
        f"- strong_meter: candidate meter_score >= {float(rule['strong_meter_threshold']):.2f}",
        "- strong_exact: candidate meter_score >= 0.70 and exact bayt count",
        f"- easy: num_strong_exact >= {easy_n} and mean_meter_score > {float(rule['easy_min_mean_meter']):.2f}",
        f"- hard: num_strong_meter == 0 OR (num_strong_meter <= {hard_n} and mean_meter_score < {float(rule['hard_max_mean_meter']):.2f})",
        "- medium: everything else",
    ]


def format_bucket_stats(title: str, rows: dict[str, dict[str, float | int]]) -> list[str]:
    lines = [title] if title else []
    if not rows:
        lines.append("- none yet")
        return lines
    for key, stats in rows.items():
        lines.append(
            f"- {key}: rows={int(stats['rows'])} mean={float(stats['mean']):.4f} min={float(stats['min']):.4f} max={float(stats['max']):.4f} std={float(stats['std']):.4f}"
        )
    return lines


def bucket_average_lines(rows: dict[str, dict[str, float | int]]) -> list[str]:
    if not rows:
        return ["- none yet"]
    lines = []
    for difficulty_name in ("easy", "medium", "hard", "unknown"):
        stats = rows.get(difficulty_name)
        if not stats:
            continue
        lines.append(
            f"- average row mean_meter_score for {difficulty_name}: {float(stats['mean']):.4f} across {int(stats['rows'])} rows"
        )
    for difficulty_name, stats in sorted(rows.items()):
        if difficulty_name in {"easy", "medium", "hard", "unknown"}:
            continue
        lines.append(
            f"- average row mean_meter_score for {difficulty_name}: {float(stats['mean']):.4f} across {int(stats['rows'])} rows"
        )
    return lines


def persist_sent_snapshot(
    *,
    state_file: Path,
    state: dict[str, Any],
    current_milestone: int,
    scored_rows: int,
    difficulty: dict[str, int],
    mark_startup_sent: bool = False,
    mark_run_complete_sent: bool = False,
    mark_publish_complete_sent: bool = False,
    mark_stale_active: bool | None = None,
) -> None:
    if mark_startup_sent:
        state["startup_email_sent"] = True
    if mark_run_complete_sent:
        state["run_complete_email_sent"] = True
    if mark_publish_complete_sent:
        state["publish_complete_email_sent"] = True
    if mark_stale_active is not None:
        state["stale_alert_active"] = bool(mark_stale_active)
    if current_milestone > int(state.get("last_sent_milestone") or 0):
        state["last_sent_milestone"] = int(current_milestone)
    state["last_sent_scored_rows"] = int(scored_rows)
    state["last_difficulty_counts"] = dict(difficulty)
    save_state(state_file, state)


def render_email_body(
    *,
    kind: str,
    run_dir: Path,
    cfg: dict[str, Any],
    summary: dict[str, Any],
    milestone_rows: int,
    pid_counts: dict[str, int],
    checkpoint: dict[str, Any],
    failure_count: int,
    recent_failures: list[str],
    diff_delta: dict[str, int],
    meter_stats: dict[str, Any],
) -> str:
    manifest_rows = int(summary["manifest_rows"])
    scored_rows = int(summary["scored_rows"])
    generated_rows = int(summary["generated_rows"])
    pct = (100.0 * scored_rows / manifest_rows) if manifest_rows else 0.0
    gen_rate = float(summary["generator_rate"]["parallel_active_rate"])
    score_rate = float(summary["score_worker_rate"]["parallel_active_rate"])
    eta = format_eta(summary.get("full_eta_seconds"))
    difficulty = summary.get("difficulty_counts") or {}
    lines = [
        f"Timestamp: {summary['timestamp']}",
        f"Kind: {kind}",
        f"Run dir: {run_dir}",
        f"Source dataset: {cfg.get('dataset_id', '')}",
        f"Output repo: {cfg.get('output_dataset_id', '')}",
        f"Base model: {cfg.get('base_model_id', '')}",
        f"Adapter repo: {cfg.get('sft_adapter_repo', '')}",
        "",
        "Progress",
        f"- milestone_rows: {milestone_rows}",
        f"- generated_rows: {generated_rows}/{manifest_rows}",
        f"- scored_rows: {scored_rows}/{manifest_rows} ({pct:.2f}%)",
        f"- candidate_total_scored: {int(summary['candidate_total_scored'])}",
        "",
        "Difficulty rule",
        *render_difficulty_rule(int(cfg.get("num_candidates") or summary.get("num_candidates") or 0)),
        "",
        "Throughput",
        f"- generator_rate: {gen_rate:.3f} cand/s",
        f"- score_rate: {score_rate:.3f} cand/s",
        f"- bottleneck_rate: {float(summary['bottleneck_candidates_per_second']):.3f} cand/s",
        f"- eta: {eta}",
        "",
        "Difficulty so far",
        f"- easy: {int(difficulty.get('easy', 0))} ({int(diff_delta.get('easy', 0)):+d} since last email)",
        f"- medium: {int(difficulty.get('medium', 0))} ({int(diff_delta.get('medium', 0)):+d} since last email)",
        f"- hard: {int(difficulty.get('hard', 0))} ({int(diff_delta.get('hard', 0)):+d} since last email)",
        "",
        "Process health",
        f"- live_generators: {pid_counts['generators']}",
        f"- live_score_workers: {pid_counts['score_workers']}",
        f"- live_checkpoint_publishers: {pid_counts['checkpoint_publishers']}",
        f"- live_email_watchers: {pid_counts['email_watchers']}",
        "",
        "Checkpoint publish state",
        f"- last_published_rows: {int(checkpoint.get('last_published_rows') or 0)}",
        f"- next_checkpoint_rows: {int(checkpoint.get('next_checkpoint_rows') or 0)}",
        f"- publish_count: {int(checkpoint.get('publish_count') or 0)}",
        f"- last_publish_kind: {checkpoint.get('last_publish_kind') or ''}",
        f"- final_ready: {bool(checkpoint.get('final_ready'))}",
        "",
        "Headline bucket averages",
        *bucket_average_lines(meter_stats.get("by_difficulty") or {}),
        "",
        "Row mean_meter_score by difficulty",
        *format_bucket_stats("", meter_stats.get("by_difficulty") or {}),
        "",
        "Row mean_meter_score by meter within difficulty",
    ]
    for difficulty_name in ("easy", "medium", "hard", "unknown"):
        bucket = (meter_stats.get("by_difficulty_meter") or {}).get(difficulty_name) or {}
        if not bucket:
            continue
        lines.append(f"- {difficulty_name}:")
        for meter_name, stats in bucket.items():
            lines.append(
                f"  - {meter_name}: rows={int(stats['rows'])} mean={float(stats['mean']):.4f} min={float(stats['min']):.4f} max={float(stats['max']):.4f} std={float(stats['std']):.4f}"
            )
    lines.extend(
        [
            "",
        "Failures",
        f"- failure_records_seen: {failure_count}",
        ]
    )
    if recent_failures:
        lines.append("- recent_failure_samples:")
        lines.extend([f"  - {item}" for item in recent_failures])
    else:
        lines.append("- recent_failure_samples: none")
    return "\n".join(lines)


def send_summary_email(
    *,
    subject: str,
    body: str,
    run_dir: Path,
    payload: dict[str, Any],
) -> bool:
    ok = False
    error_text = ""
    try:
        ok = bool(send_email(subject, body))
        if not ok:
            error_text = "send_email returned false"
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
    append_jsonl(
        event_path(run_dir),
        {
            "event": "email_attempt",
            "timestamp": utc_now_iso(),
            "subject": subject,
            "ok": ok,
            "error": error_text,
            **payload,
        },
    )
    return ok


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_run_config(run_dir)
    every_rows = max(1, int(args.every_rows))
    poll_seconds = max(30, int(args.poll_seconds))
    stale_minutes = float(args.stale_minutes or os.getenv("WATCHER_STALE_MINUTES", "20") or 20.0)
    state_file, state = load_watcher_state(run_dir)

    while True:
        summary = summarize_run(run_dir, full_train_rows=int(args.full_train_rows))
        scored_rows = int(summary["scored_rows"])
        difficulty = {str(k): int(v) for k, v in (summary.get("difficulty_counts") or {}).items()}
        checkpoint = checkpoint_summary(run_dir)
        pid_counts = live_pid_summary(run_dir)
        failure_count, recent_failures = failure_summary(run_dir)

        if scored_rows > int(state["last_progress_scored_rows"]):
            state["last_progress_scored_rows"] = scored_rows
            state["last_progress_at"] = utc_now_iso()
            state["stale_alert_active"] = False

        atomic_write_json(
            status_path(run_dir, "email_watcher"),
            {
                "status": "watching",
                "timestamp": utc_now_iso(),
                "scored_rows": scored_rows,
                "last_sent_milestone": int(state["last_sent_milestone"]),
                "last_sent_scored_rows": int(state["last_sent_scored_rows"]),
                "last_progress_at": state["last_progress_at"],
                "stale_alert_active": bool(state["stale_alert_active"]),
                "run_complete_email_sent": bool(state["run_complete_email_sent"]),
                "publish_complete_email_sent": bool(state["publish_complete_email_sent"]),
            },
        )

        current_milestone = (scored_rows // every_rows) * every_rows
        if not bool(state["startup_email_sent"]):
            delta = difficulty_delta(difficulty, state.get("last_difficulty_counts") or {})
            meter_stats = row_meter_mean_stats(run_dir)
            subject = f"[Shaer meter+count] watcher attached | scored={scored_rows}/{summary['manifest_rows']} | eta={format_eta(summary.get('full_eta_seconds'))}"
            body = render_email_body(
                kind="startup",
                run_dir=run_dir,
                cfg=cfg,
                summary=summary,
                milestone_rows=current_milestone,
                pid_counts=pid_counts,
                checkpoint=checkpoint,
                failure_count=failure_count,
                recent_failures=recent_failures,
                diff_delta=delta,
                meter_stats=meter_stats,
            )
            if send_summary_email(subject=subject, body=body, run_dir=run_dir, payload={"kind": "startup", "scored_rows": scored_rows}):
                persist_sent_snapshot(
                    state_file=state_file,
                    state=state,
                    current_milestone=current_milestone if current_milestone >= every_rows else 0,
                    scored_rows=scored_rows,
                    difficulty=difficulty,
                    mark_startup_sent=True,
                )

        if current_milestone >= every_rows and current_milestone > int(state["last_sent_milestone"]):
            delta = difficulty_delta(difficulty, state.get("last_difficulty_counts") or {})
            pct = (100.0 * scored_rows / int(summary["manifest_rows"])) if int(summary["manifest_rows"]) else 0.0
            meter_stats = row_meter_mean_stats(run_dir)
            subject = f"[Shaer meter+count] {current_milestone:,} rows scored | {pct:.2f}% | eta={format_eta(summary.get('full_eta_seconds'))}"
            body = render_email_body(
                kind="milestone",
                run_dir=run_dir,
                cfg=cfg,
                summary=summary,
                milestone_rows=current_milestone,
                pid_counts=pid_counts,
                checkpoint=checkpoint,
                failure_count=failure_count,
                recent_failures=recent_failures,
                diff_delta=delta,
                meter_stats=meter_stats,
            )
            if send_summary_email(
                subject=subject,
                body=body,
                run_dir=run_dir,
                payload={"kind": "milestone", "milestone_rows": current_milestone, "scored_rows": scored_rows},
            ):
                persist_sent_snapshot(
                    state_file=state_file,
                    state=state,
                    current_milestone=current_milestone,
                    scored_rows=scored_rows,
                    difficulty=difficulty,
                )

        try:
            last_progress_epoch = time.mktime(time.strptime(state["last_progress_at"], "%Y-%m-%dT%H:%M:%SZ"))
        except Exception:
            last_progress_epoch = time.time()
        stale_now = (time.time() - last_progress_epoch) / 60.0
        if stale_minutes > 0 and stale_now >= stale_minutes and not bool(state["stale_alert_active"]):
            meter_stats = row_meter_mean_stats(run_dir)
            subject = f"[Shaer meter+count] stale alert | no scored progress for {stale_now:.1f}m"
            body = render_email_body(
                kind="stale_alert",
                run_dir=run_dir,
                cfg=cfg,
                summary=summary,
                milestone_rows=current_milestone,
                pid_counts=pid_counts,
                checkpoint=checkpoint,
                failure_count=failure_count,
                recent_failures=recent_failures,
                diff_delta={"easy": 0, "medium": 0, "hard": 0},
                meter_stats=meter_stats,
            )
            if send_summary_email(subject=subject, body=body, run_dir=run_dir, payload={"kind": "stale_alert", "scored_rows": scored_rows}):
                persist_sent_snapshot(
                    state_file=state_file,
                    state=state,
                    current_milestone=int(state.get("last_sent_milestone") or 0),
                    scored_rows=scored_rows,
                    difficulty=difficulty,
                    mark_stale_active=True,
                )

        if (
            bool(summary["all_generators_done"])
            and scored_rows >= int(summary["manifest_rows"])
            and not bool(state["run_complete_email_sent"])
        ):
            meter_stats = row_meter_mean_stats(run_dir)
            subject = f"[Shaer meter+count] scoring complete | {scored_rows:,} rows | waiting on final publish"
            body = render_email_body(
                kind="run_complete",
                run_dir=run_dir,
                cfg=cfg,
                summary=summary,
                milestone_rows=scored_rows,
                pid_counts=pid_counts,
                checkpoint=checkpoint,
                failure_count=failure_count,
                recent_failures=recent_failures,
                diff_delta={"easy": 0, "medium": 0, "hard": 0},
                meter_stats=meter_stats,
            )
            if send_summary_email(subject=subject, body=body, run_dir=run_dir, payload={"kind": "run_complete", "scored_rows": scored_rows}):
                persist_sent_snapshot(
                    state_file=state_file,
                    state=state,
                    current_milestone=max(current_milestone, int(state.get("last_sent_milestone") or 0)),
                    scored_rows=scored_rows,
                    difficulty=difficulty,
                    mark_run_complete_sent=True,
                )

        if checkpoint.get("status") == "published" and str(checkpoint.get("last_publish_kind") or "") == "final" and not bool(state["publish_complete_email_sent"]):
            meter_stats = row_meter_mean_stats(run_dir)
            subject = f"[Shaer meter+count] final publish complete | repo ready"
            body = render_email_body(
                kind="publish_complete",
                run_dir=run_dir,
                cfg=cfg,
                summary=summary,
                milestone_rows=scored_rows,
                pid_counts=pid_counts,
                checkpoint=checkpoint,
                failure_count=failure_count,
                recent_failures=recent_failures,
                diff_delta={"easy": 0, "medium": 0, "hard": 0},
                meter_stats=meter_stats,
            )
            if send_summary_email(subject=subject, body=body, run_dir=run_dir, payload={"kind": "publish_complete", "scored_rows": scored_rows}):
                persist_sent_snapshot(
                    state_file=state_file,
                    state=state,
                    current_milestone=max(current_milestone, int(state.get("last_sent_milestone") or 0)),
                    scored_rows=scored_rows,
                    difficulty=difficulty,
                    mark_publish_complete_sent=True,
                )
                return 0

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
