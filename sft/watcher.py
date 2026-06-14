from __future__ import annotations

import hashlib
import json
import os
import smtplib
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def send_email(subject: str, body: str) -> bool:
    load_dotenv(ENV_PATH, override=False)
    if os.getenv("WATCHER_EMAIL_ENABLED", "true").lower() != "true":
        return False

    host = os.getenv("WATCHER_SMTP_HOST", "")
    port = int(os.getenv("WATCHER_SMTP_PORT", "587"))
    username = os.getenv("WATCHER_SMTP_USERNAME", "")
    password = os.getenv("WATCHER_SMTP_PASSWORD", "")
    from_email = os.getenv("WATCHER_FROM_EMAIL", "")
    to_email = os.getenv("WATCHER_TO_EMAIL", "")
    if not all([host, username, password, from_email, to_email]):
        return False

    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(from_email, [to_email], msg.as_string())
    return True


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def state_path_for(target: str) -> Path:
    load_dotenv(ENV_PATH, override=False)
    root = os.getenv("WATCHER_STATE_DIR", "").strip()
    base = Path(root).expanduser() if root else (Path(__file__).resolve().parent / "watcher_state")
    base.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    return base / f"{digest}.json"


def load_state(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def save_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def discover_run_dir(target: Path) -> Path | None:
    if target.is_dir():
        return target
    parent = target.parent
    candidates = [child for child in parent.iterdir() if child.is_dir() and (child / "train.log").exists()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def latest_row(rows: list[dict[str, Any]], mode: str) -> dict[str, Any] | None:
    matches = [row for row in rows if str(row.get("mode", "")) == mode]
    if not matches:
        return None
    matches.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return matches[-1]


def latest_probe(run_dir: Path) -> dict[str, Any] | None:
    rows = read_jsonl(run_dir / "probe_metrics.jsonl")
    if not rows:
        return None
    rows.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return rows[-1]


def summarize_examples(run_dir: Path, step: int) -> str:
    rows = [row for row in read_jsonl(run_dir / "probe_generations.jsonl") if int(row.get("global_step", -1) or -1) == step]
    if not rows:
        return "No probe generations yet."
    rows.sort(key=lambda row: float(row.get("probe_meter_score", 0.0)), reverse=True)
    best = rows[:2]
    worst = list(reversed(rows[-2:]))
    parts = ["Best probe generations:"]
    for row in best:
        preview = str(row.get("completion_text", "")).replace("\n", " | ")[:250]
        parts.append(
            f"- meter={float(row.get('probe_meter_score', 0.0)):.3f} "
            f"count={float(row.get('probe_count_adherence', 0.0)):.3f} "
            f"{row.get('base_meter','')} {row.get('requested_bayts','')} bayts | {preview}"
        )
    parts.append("")
    parts.append("Weakest probe generations:")
    for row in worst:
        preview = str(row.get("completion_text", "")).replace("\n", " | ")[:250]
        parts.append(
            f"- meter={float(row.get('probe_meter_score', 0.0)):.3f} "
            f"count={float(row.get('probe_count_adherence', 0.0)):.3f} "
            f"{row.get('base_meter','')} {row.get('requested_bayts','')} bayts | {preview}"
        )
    return "\n".join(parts)


def send_step_summary(run_dir: Path, step: int) -> bool:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    train_row = next((row for row in reversed(metrics) if str(row.get("mode", "")) == "train" and int(row.get("global_step", 0) or 0) == step), None)
    live_eval = latest_row(metrics, "eval")
    probe = latest_probe(run_dir)
    if train_row is None or probe is None:
        return False

    body = [
        f"Run dir: {run_dir}",
        f"Step: {step}",
        "",
        "Train snapshot:",
        f"- loss: {float(train_row.get('loss', 0.0)):.4f}",
        "",
        "Eval snapshot:",
        f"- eval_loss: {float(live_eval.get('eval_loss', 0.0)):.4f}" if live_eval else "- not available yet",
        "",
        "Probe snapshot:",
        f"- probe_meter_mean: {float(probe.get('probe_meter_mean', 0.0)):.4f}",
        f"- probe_count_mean: {float(probe.get('probe_count_adherence_mean', 0.0)):.4f}",
        "",
    ]
    worst = []
    per_meter = probe.get("per_meter_probe_meter_mean", {})
    if isinstance(per_meter, str):
        try:
            per_meter = json.loads(per_meter)
        except Exception:
            per_meter = {}
    if isinstance(per_meter, dict):
        worst = sorted(per_meter.items(), key=lambda item: float(item[1]))[:4]
    if worst:
        body.append("Weakest meters:")
        for meter, value in worst:
            body.append(f"- {meter}: {float(value):.4f}")
        body.append("")
    body.append(summarize_examples(run_dir, int(probe.get("global_step", step) or step)))
    return send_email(f"SFT milestone step {step}", "\n".join(body))


def send_stale_notice(run_dir: Path, log_path: Path, stale_minutes: int) -> bool:
    return send_email(
        "SFT watcher stale alert",
        f"Run dir: {run_dir}\nLog: {log_path}\nNo updates for {stale_minutes} minutes.",
    )


def send_run_started_notice(run_dir: Path, target: Path) -> bool:
    return send_email(
        "SFT run started",
        "\n".join(
            [
                f"Run dir: {run_dir}",
                f"Target: {target}",
                "",
                "Watcher is active and will send milestone summaries with train and eval snapshots.",
            ]
        ),
    )


def send_run_complete_notice(run_dir: Path) -> bool:
    summary = read_json(run_dir / "run_summary.json") or {}
    final_eval = summary.get("final_eval") or summary.get("final_live_eval") or {}
    final_test = summary.get("final_test_eval") or summary.get("final_full_eval") or {}
    final_probe = summary.get("final_probe") or {}
    body = [
        f"Run dir: {run_dir}",
        f"Run name: {summary.get('run_name', '')}",
        f"Train global step: {summary.get('train_global_step', '')}",
        f"Train loss: {float(summary.get('train_loss', 0.0)):.4f}" if summary.get("train_loss") is not None else "Train loss: n/a",
        "",
        "Final eval:",
        f"- eval_loss: {float(final_eval.get('eval_loss', 0.0)):.4f}" if final_eval else "- not available",
        "",
        "Final test:",
        f"- test_loss: {float(final_test.get('test_loss', 0.0)):.4f}" if final_test else "- not available",
        "",
        "Final probe:",
        f"- probe_meter_mean: {float(final_probe.get('probe_meter_mean', 0.0)):.4f}" if final_probe else "- not available",
        f"- probe_count_mean: {float(final_probe.get('probe_count_adherence_mean', 0.0)):.4f}" if final_probe else "- not available",
        "",
        f"Latest adapter path: {summary.get('latest_adapter_path', '')}",
        f"Best adapter path: {summary.get('best_adapter_path', '')}",
        f"Latest checkpoint path: {summary.get('latest_checkpoint_repo_path', '')}",
    ]
    return send_email("SFT run complete", "\n".join(body))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: watcher.py <outer-log-or-run-dir>", file=sys.stderr)
        return 2

    load_dotenv(ENV_PATH, override=False)
    target = Path(sys.argv[1]).resolve()
    state_path = state_path_for(str(target))
    state = load_state(state_path)
    email_every = int(os.getenv("WATCHER_STEP_EMAIL_EVERY", "50"))
    poll_seconds = int(os.getenv("WATCHER_POLL_SECONDS", "30"))
    stale_minutes = int(os.getenv("WATCHER_STALE_MINUTES", "20"))

    while True:
        run_dir = discover_run_dir(target)
        if run_dir is None:
            time.sleep(poll_seconds)
            continue

        if state.get("run_dir") != str(run_dir):
            state["run_dir"] = str(run_dir)
            if state.get("run_started_email") != str(run_dir):
                if send_run_started_notice(run_dir, target):
                    state["run_started_email"] = str(run_dir)
                    save_state(state_path, state)

        metrics_path = run_dir / "metrics.jsonl"
        latest_mtime = metrics_path.stat().st_mtime if metrics_path.exists() else 0.0
        if latest_mtime:
            state["last_seen_mtime"] = latest_mtime

        metrics = read_jsonl(metrics_path)
        train_rows = [row for row in metrics if str(row.get("mode", "")) == "train"]
        if train_rows:
            latest_step = int(train_rows[-1].get("global_step", 0) or 0)
            next_step = ((latest_step // email_every) * email_every)
            if next_step > 0 and latest_step >= next_step and state.get("last_step_email") != next_step:
                if send_step_summary(run_dir, next_step):
                    state["last_step_email"] = next_step
                    save_state(state_path, state)

        last_seen = float(state.get("last_seen_mtime", latest_mtime or 0.0))
        if last_seen and (time.time() - last_seen) >= stale_minutes * 60:
            if state.get("last_stale_email_mtime") != last_seen:
                if send_stale_notice(run_dir, target, stale_minutes):
                    state["last_stale_email_mtime"] = last_seen
                    save_state(state_path, state)

        summary_path = run_dir / "run_summary.json"
        if summary_path.exists() and state.get("run_complete_email") != str(summary_path):
            if send_run_complete_notice(run_dir):
                state["run_complete_email"] = str(summary_path)
                save_state(state_path, state)

        save_state(state_path, state)
        time.sleep(max(10, poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
