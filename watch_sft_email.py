#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import smtplib
import sys
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any


LOG_DIR_RE = re.compile(r"Log dir:\s+(.+)$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch SFT logs and send email alerts for eval loss and model updates."
    )
    parser.add_argument("--detached-log", type=Path, default=Path("logs/sft_full_detached.log"))
    parser.add_argument("--runs-root", type=Path, default=Path("logs/sft/full"))
    parser.add_argument("--state-file", type=Path, default=Path("logs/sft_email_watcher_state.json"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--smtp-host", default="smtp.gmail.com")
    parser.add_argument("--smtp-port", type=int, default=465)
    parser.add_argument("--smtp-email", default="")
    parser.add_argument("--smtp-app-password", default="")
    parser.add_argument("--to-email", default="")
    parser.add_argument("--subject-prefix", default="[SFT Watcher]")
    parser.add_argument("--initial-run-dir", type=Path, default=None)
    parser.add_argument(
        "--read-from-start",
        action="store_true",
        help="Read existing file history on first attach. Default is follow-new-only.",
    )
    return parser.parse_args()


def resolve_path(p: Path) -> Path:
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cursors": {}, "current_run_dir": None, "last_eval_loss_by_run": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("invalid state structure")
        data.setdefault("cursors", {})
        data.setdefault("current_run_dir", None)
        data.setdefault("last_eval_loss_by_run", {})
        return data
    except Exception:
        return {"cursors": {}, "current_run_dir": None, "last_eval_loss_by_run": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def ensure_cursor(
    state: dict[str, Any], path: Path, *, read_from_start: bool
) -> dict[str, int | None]:
    key = str(path)
    cur = state["cursors"].get(key)
    if isinstance(cur, dict) and "offset" in cur:
        return cur

    if path.exists():
        st = path.stat()
        cur = {
            "offset": 0 if read_from_start else int(st.st_size),
            "inode": int(st.st_ino),
        }
    else:
        cur = {"offset": 0, "inode": None}
    state["cursors"][key] = cur
    return cur


def read_new_lines(
    state: dict[str, Any],
    path: Path,
    *,
    read_from_start: bool,
) -> list[str]:
    cur = ensure_cursor(state, path, read_from_start=read_from_start)
    if not path.exists():
        return []

    st = path.stat()
    inode = int(st.st_ino)
    size = int(st.st_size)

    if cur.get("inode") != inode or int(cur.get("offset", 0)) > size:
        cur["offset"] = 0
        cur["inode"] = inode

    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(int(cur.get("offset", 0)))
        lines = f.readlines()
        cur["offset"] = int(f.tell())
        cur["inode"] = inode
    return [ln.rstrip("\n") for ln in lines]


def pick_latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates: list[Path] = []
    for d in runs_root.glob("full_*"):
        if d.is_dir() and (d / "metrics.jsonl").exists() and (d / "events.jsonl").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_name_from_dir(run_dir: Path) -> str:
    return run_dir.name


def send_email(
    *,
    smtp_host: str,
    smtp_port: int,
    smtp_email: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["From"] = smtp_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as smtp:
        smtp.login(smtp_email, smtp_password)
        smtp.send_message(msg)


def detect_log_dir_lines(lines: list[str]) -> Path | None:
    found: Path | None = None
    for ln in lines:
        m = LOG_DIR_RE.search(ln)
        if not m:
            continue
        p = Path(m.group(1).strip())
        found = resolve_path(p)
    return found


def parse_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def read_run_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config_snapshot.json"
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def send_startup_email(
    *,
    args: argparse.Namespace,
    smtp_email: str,
    smtp_password: str,
    to_email: str,
    run_dir: Path | None,
) -> None:
    if run_dir is None:
        subject = f"{args.subject_prefix} Watcher started"
        body = "\n".join(
            [
                "SFT watcher started.",
                f"Timestamp (UTC): {utc_now()}",
                f"Detached log: {args.detached_log}",
                "Run dir: not detected yet",
                "I will start sending emails when a run is detected.",
            ]
        )
    else:
        cfg = read_run_config(run_dir)
        eval_steps = cfg.get("eval_steps")
        save_steps = cfg.get("save_steps")
        run_name = run_name_from_dir(run_dir)
        cadence_line = (
            f"I will send eval-loss emails about every {eval_steps} training steps."
            if isinstance(eval_steps, int) and eval_steps > 0
            else "Eval cadence could not be read from config_snapshot.json."
        )
        subject = f"{args.subject_prefix} Watcher started | run={run_name}"
        body = "\n".join(
            [
                "SFT watcher is now running in detached mode.",
                f"Timestamp (UTC): {utc_now()}",
                f"Run: {run_name}",
                f"Run dir: {run_dir}",
                f"Detached log: {args.detached_log}",
                f"Metrics file: {run_dir / 'metrics.jsonl'}",
                f"Events file: {run_dir / 'events.jsonl'}",
                cadence_line,
                (
                    f"Checkpoints/model-upload alerts are tied to save cadence "
                    f"(current save_steps={save_steps})."
                    if isinstance(save_steps, int) and save_steps > 0
                    else "Model-upload alerts will trigger on checkpoint/adapter upload events."
                ),
                "Alert types: eval_loss, checkpoint_uploaded, adapter_latest_uploaded.",
            ]
        )

    send_email(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_email=smtp_email,
        smtp_password=smtp_password,
        to_email=to_email,
        subject=subject,
        body=body,
    )


def format_eval_body(run_dir: Path, row: dict[str, Any], prev_eval: float | None) -> str:
    step = row.get("global_step")
    epoch = row.get("epoch")
    eval_loss = row.get("eval_loss")
    phase = row.get("phase", "eval")
    ts = row.get("timestamp_utc", utc_now())
    runtime = row.get("eval_runtime")
    sps = row.get("eval_steps_per_second")
    ssps = row.get("eval_samples_per_second")

    delta = None
    trend = "n/a"
    if prev_eval is not None and isinstance(eval_loss, (int, float)):
        delta = float(eval_loss) - float(prev_eval)
        if delta < 0:
            trend = "improved"
        elif delta > 0:
            trend = "worse"
        else:
            trend = "flat"

    lines = [
        f"Run: {run_name_from_dir(run_dir)}",
        f"Run dir: {run_dir}",
        f"Timestamp (UTC): {ts}",
        f"Phase: {phase}",
        f"Step: {step}",
        f"Epoch: {epoch}",
        f"Eval loss: {eval_loss}",
    ]
    if delta is not None:
        lines.append(f"Delta vs previous eval: {delta:+.6f} ({trend})")
    if runtime is not None:
        lines.append(f"Eval runtime (s): {runtime}")
    if sps is not None:
        lines.append(f"Eval steps/sec: {sps}")
    if ssps is not None:
        lines.append(f"Eval samples/sec: {ssps}")
    return "\n".join(lines)


def format_model_update_body(run_dir: Path, evt: dict[str, Any]) -> str:
    event_type = evt.get("event_type", "unknown")
    step = evt.get("global_step")
    ts = evt.get("timestamp_utc", utc_now())
    hf_path = evt.get("hf_path", "")
    duration_ms = evt.get("duration_ms")
    status = evt.get("status")

    lines = [
        f"Run: {run_name_from_dir(run_dir)}",
        f"Run dir: {run_dir}",
        f"Timestamp (UTC): {ts}",
        f"Event: {event_type}",
        f"Step: {step}",
        f"HF path: {hf_path}",
        f"Status: {status}",
    ]
    if duration_ms is not None:
        lines.append(f"Upload duration (ms): {duration_ms}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.detached_log = resolve_path(args.detached_log)
    args.runs_root = resolve_path(args.runs_root)
    args.state_file = resolve_path(args.state_file)
    args.initial_run_dir = resolve_path(args.initial_run_dir) if args.initial_run_dir else None

    smtp_email = args.smtp_email.strip()
    smtp_password = args.smtp_app_password.strip().replace(" ", "")
    to_email = args.to_email.strip()

    if not smtp_email or not smtp_password or not to_email:
        print(
            "Missing SMTP config. Provide --smtp-email, --smtp-app-password, and --to-email.",
            file=sys.stderr,
        )
        return 2

    state = load_state(args.state_file)
    current_run_dir: Path | None = None

    if args.initial_run_dir and args.initial_run_dir.exists():
        current_run_dir = args.initial_run_dir
    elif state.get("current_run_dir"):
        p = Path(state["current_run_dir"])
        if p.exists():
            current_run_dir = p
    if current_run_dir is None:
        current_run_dir = pick_latest_run_dir(args.runs_root)

    if current_run_dir is not None:
        state["current_run_dir"] = str(current_run_dir)
        print(f"[{utc_now()}] Initial run dir: {current_run_dir}", flush=True)
    else:
        print(f"[{utc_now()}] Waiting for run dir...", flush=True)

    try:
        send_startup_email(
            args=args,
            smtp_email=smtp_email,
            smtp_password=smtp_password,
            to_email=to_email,
            run_dir=current_run_dir,
        )
        print(f"[{utc_now()}] Sent startup email.", flush=True)
    except Exception as e:
        print(f"[{utc_now()}] Email send failed (startup): {e}", file=sys.stderr, flush=True)

    try:
        while True:
            detached_lines = read_new_lines(
                state,
                args.detached_log,
                read_from_start=args.read_from_start,
            )
            detected = detect_log_dir_lines(detached_lines)
            if detected and detected.exists():
                if current_run_dir is None or detected != current_run_dir:
                    current_run_dir = detected
                    state["current_run_dir"] = str(current_run_dir)
                    print(f"[{utc_now()}] Switched run dir: {current_run_dir}", flush=True)

            if current_run_dir is None:
                guess = pick_latest_run_dir(args.runs_root)
                if guess is not None:
                    current_run_dir = guess
                    state["current_run_dir"] = str(current_run_dir)
                    print(f"[{utc_now()}] Discovered run dir: {current_run_dir}", flush=True)

            if current_run_dir is not None:
                run_key = str(current_run_dir)
                run_name = run_name_from_dir(current_run_dir)

                metrics_path = current_run_dir / "metrics.jsonl"
                for line in read_new_lines(
                    state,
                    metrics_path,
                    read_from_start=args.read_from_start,
                ):
                    row = parse_json_line(line)
                    if row is None:
                        continue
                    if "eval_loss" not in row:
                        continue

                    prev_eval = state["last_eval_loss_by_run"].get(run_key)
                    subject = (
                        f"{args.subject_prefix} Eval loss | run={run_name} "
                        f"step={row.get('global_step')}"
                    )
                    body = format_eval_body(current_run_dir, row, prev_eval)
                    try:
                        send_email(
                            smtp_host=args.smtp_host,
                            smtp_port=args.smtp_port,
                            smtp_email=smtp_email,
                            smtp_password=smtp_password,
                            to_email=to_email,
                            subject=subject,
                            body=body,
                        )
                        print(f"[{utc_now()}] Sent eval email: {subject}", flush=True)
                    except Exception as e:
                        print(f"[{utc_now()}] Email send failed (eval): {e}", file=sys.stderr, flush=True)

                    if isinstance(row.get("eval_loss"), (int, float)):
                        state["last_eval_loss_by_run"][run_key] = float(row["eval_loss"])

                events_path = current_run_dir / "events.jsonl"
                for line in read_new_lines(
                    state,
                    events_path,
                    read_from_start=args.read_from_start,
                ):
                    evt = parse_json_line(line)
                    if evt is None:
                        continue
                    if evt.get("event_type") not in {"checkpoint_uploaded", "adapter_latest_uploaded"}:
                        continue

                    event_type = evt.get("event_type")
                    subject = (
                        f"{args.subject_prefix} Model updated | run={run_name} "
                        f"event={event_type} step={evt.get('global_step')}"
                    )
                    body = format_model_update_body(current_run_dir, evt)
                    try:
                        send_email(
                            smtp_host=args.smtp_host,
                            smtp_port=args.smtp_port,
                            smtp_email=smtp_email,
                            smtp_password=smtp_password,
                            to_email=to_email,
                            subject=subject,
                            body=body,
                        )
                        print(f"[{utc_now()}] Sent model-update email: {subject}", flush=True)
                    except Exception as e:
                        print(
                            f"[{utc_now()}] Email send failed (model update): {e}",
                            file=sys.stderr,
                            flush=True,
                        )

            save_state(args.state_file, state)
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        save_state(args.state_file, state)
        print(f"[{utc_now()}] Watcher stopped.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
