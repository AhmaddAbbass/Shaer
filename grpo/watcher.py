import hashlib
import json
import os
import smtplib
import sys
import time
from collections import defaultdict
from email.mime.text import MIMEText
from pathlib import Path
from statistics import pstdev

from dotenv import load_dotenv


def send_email(subject, body):
    load_dotenv(override=False)
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


def read_tail(path, max_lines=60):
    path = Path(path)
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path):
    rows = []
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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def state_path_for(target: str) -> Path:
    state_root = os.getenv("WATCHER_STATE_DIR", "").strip()
    if state_root:
        root = Path(state_root).expanduser()
    else:
        root = Path.cwd() / ".watcher_state"
    ensure_dir(root)
    digest = hashlib.sha256(target.encode("utf-8")).hexdigest()[:16]
    return root / f"{digest}.json"


def load_state(path: Path):
    data = read_json(path)
    if isinstance(data, dict):
        return data
    return {}


def save_state(path: Path, payload: dict):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def discover_run_dir(target: Path):
    if target.is_dir():
        return target
    root = target.parent
    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        train_log = child / "train.log"
        if not train_log.exists():
            continue
        candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def generation_step(row):
    mode = str(row.get("mode", "train"))
    if mode == "train" and row.get("logical_step") is not None:
        return row.get("logical_step")
    return row.get("global_step")


def latest_metric_row(metrics_rows, mode: str, step: int | None = None):
    matches = []
    for row in metrics_rows:
        if str(row.get("mode", "")) != mode:
            continue
        if step is not None and int(row.get("global_step", -1) or -1) != int(step):
            continue
        matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return matches[-1]


def previous_metric_row(metrics_rows, mode: str, current_step: int):
    matches = []
    for row in metrics_rows:
        if str(row.get("mode", "")) != mode:
            continue
        step = int(row.get("global_step", -1) or -1)
        if step >= int(current_step):
            continue
        matches.append(row)
    if not matches:
        return None
    matches.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return matches[-1]


def train_rows(metrics_rows):
    rows = [row for row in metrics_rows if str(row.get("mode", "")) == "train"]
    rows.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return rows


def eval_rows(metrics_rows):
    rows = [row for row in metrics_rows if str(row.get("mode", "")) == "eval"]
    rows.sort(key=lambda row: int(row.get("global_step", 0) or 0))
    return rows


def latest_train_step(metrics_rows):
    rows = train_rows(metrics_rows)
    if not rows:
        return 0
    return int(rows[-1].get("global_step", 0) or 0)


def milestone_ready(step_interval: int, milestone: int, metrics_rows, checkpoint_events):
    train_row = latest_metric_row(metrics_rows, "train", milestone)
    if not train_row:
        return False
    if latest_metric_row(metrics_rows, "eval", milestone):
        return True
    for event in checkpoint_events:
        if int(event.get("global_step", -1) or -1) != milestone:
            continue
        if str(event.get("event_type", "")) in {"evaluation_completed", "checkpoint_saved"}:
            return True
    later_train = latest_metric_row(metrics_rows, "train", milestone + 1)
    if later_train:
        return True
    return False


def window_average(rows, metric_name: str, last_n: int = 10):
    values = []
    for row in rows[-last_n:]:
        value = row.get(metric_name)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None, None
    return sum(values) / len(values), (pstdev(values) if len(values) > 1 else 0.0)


def metric_value(row, *names, default=0.0):
    if not row:
        return default
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except Exception:
                continue
    return default


def record_status(rows, metric_name: str, current_step: int):
    points = []
    for row in rows:
        value = row.get(metric_name)
        step = row.get("global_step")
        if value is None or step is None:
            continue
        try:
            points.append((int(step), float(value)))
        except Exception:
            continue
    if not points:
        return []
    current_value = None
    prior = []
    for step, value in points:
        if step == int(current_step):
            current_value = value
        elif step < int(current_step):
            prior.append((step, value))
    if current_value is None:
        return []
    messages = []
    if not prior:
        messages.append("first recorded eval point")
        return messages
    prior_values = [value for _, value in prior]
    if current_value > max(prior_values):
        prev_step, prev_val = max(prior, key=lambda item: item[1])
        messages.append(f"new max vs previous best step {prev_step} ({prev_val:.4f})")
    if current_value < min(prior_values):
        prev_step, prev_val = min(prior, key=lambda item: item[1])
        messages.append(f"new min vs previous worst step {prev_step} ({prev_val:.4f})")
    return messages


def format_delta(current: float, previous: float | None):
    if previous is None:
        return "n/a"
    return f"{current - previous:+.4f}"


def eval_per_meter_summary(run_dir: Path, step: int):
    buckets = defaultdict(lambda: {"meter": [], "count": [], "repeat": [], "total": []})
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", "")) != "eval":
            continue
        row_step = generation_step(row)
        if row_step is None or int(row_step) != int(step):
            continue
        meter_name = str(row.get("base_meter", "") or row.get("meter_label", "")).strip()
        if not meter_name:
            continue
        buckets[meter_name]["meter"].append(float(row.get("reward_meter", 0.0) or 0.0))
        buckets[meter_name]["count"].append(float(row.get("reward_count_adherence", row.get("reward_exact_count_bonus", 0.0)) or 0.0))
        buckets[meter_name]["repeat"].append(float(row.get("reward_repeat_soft", row.get("reward_repeat_penalty", 0.0)) or 0.0))
        buckets[meter_name]["total"].append(float(row.get("reward_total", 0.0) or 0.0))
    out = {}
    for meter_name, values in buckets.items():
        out[meter_name] = {
            "meter": sum(values["meter"]) / len(values["meter"]) if values["meter"] else 0.0,
            "count": sum(values["count"]) / len(values["count"]) if values["count"] else 0.0,
            "repeat": sum(values["repeat"]) / len(values["repeat"]) if values["repeat"] else 0.0,
            "total": sum(values["total"]) / len(values["total"]) if values["total"] else 0.0,
        }
    return out


def eval_quality_summary(run_dir: Path, step: int):
    buckets = {
        "exact_count_rate": [],
        "hard_gate_failure": [],
        "contamination_fail": [],
        "digit_leakage": [],
        "contamination_ratio": [],
        "artifact_free": [],
        "lexical_plausibility": [],
        "near_duplicate_penalty": [],
        "opening_diversity": [],
        "distinct_2": [],
    }
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", "")) != "eval":
            continue
        row_step = generation_step(row)
        if row_step is None or int(row_step) != int(step):
            continue
        extra = row.get("total_composite_extra") or row.get("repeat_penalty_extra") or row.get("arabic_clean_extra") or {}
        buckets["exact_count_rate"].append(1.0 if bool(extra.get("exact_count", False)) else 0.0)
        buckets["hard_gate_failure"].append(1.0 if bool(extra.get("hard_gate_blocked", False)) else 0.0)
        buckets["contamination_fail"].append(
            1.0
            if (
                (not bool(extra.get("has_arabic", False)))
                or bool(extra.get("contains_latin", False))
                or bool(extra.get("contains_digits", False))
                or bool(extra.get("contains_forbidden_artifacts", False))
            )
            else 0.0
        )
        buckets["digit_leakage"].append(1.0 if bool(extra.get("contains_digits", False)) else 0.0)
        buckets["contamination_ratio"].append(float(extra.get("contamination_ratio", 0.0) or 0.0))
        buckets["artifact_free"].append(float(extra.get("artifact_free_score", 0.0) or 0.0))
        buckets["lexical_plausibility"].append(float(extra.get("lexical_plausibility_score", 0.0) or 0.0))
        buckets["near_duplicate_penalty"].append(float(extra.get("near_duplicate_score", 0.0) or 0.0))
        buckets["opening_diversity"].append(float(extra.get("opening_diversity_score", 0.0) or 0.0))
        buckets["distinct_2"].append(float(extra.get("distinct_2_score", 0.0) or 0.0))
    return {
        key: (sum(values) / len(values) if values else None)
        for key, values in buckets.items()
    }


def per_meter_lines(current_step: int, current_summary: dict, previous_step: int | None, previous_summary: dict):
    meters = sorted(set(current_summary) | set(previous_summary))
    if not meters:
        return ["- no per-meter eval rows yet"]
    lines = []
    for meter_name in meters:
        cur = current_summary.get(meter_name)
        prev = previous_summary.get(meter_name)
        if cur and prev:
            lines.append(
                f"- {meter_name}: meter {cur['meter']:.3f} ({cur['meter'] - prev['meter']:+.3f}), "
                f"count {cur['count']:.3f} ({cur['count'] - prev['count']:+.3f}), "
                f"repeat {cur['repeat']:.3f} ({cur['repeat'] - prev['repeat']:+.3f}), "
                f"total {cur['total']:.3f} ({cur['total'] - prev['total']:+.3f})"
            )
        elif cur and not prev:
            lines.append(
                f"- {meter_name}: meter {cur['meter']:.3f}, count {cur['count']:.3f}, repeat {cur['repeat']:.3f}, total {cur['total']:.3f} (new)"
            )
        else:
            lines.append(f"- {meter_name}: no longer updated after step {previous_step}")
    return lines


def candidate_examples(run_dir: Path, milestone: int, max_examples: int = 1):
    rows = []
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", "")) != "train":
            continue
        step = generation_step(row)
        if step is None or int(step) != milestone:
            continue
        rows.append(row)
    if not rows:
        return []
    rows.sort(key=lambda row: float(row.get("reward_total", 0.0)), reverse=True)
    examples = []
    best = rows[:max_examples]
    worst = list(reversed(rows[-max_examples:]))
    for label, group in [("Best candidates", best), ("Weakest candidates", worst)]:
        lines = [label + ":"]
        for row in group:
            preview = str(row.get("completion_text", "")).replace("\n", " | ")[:220]
            lines.append(
                f"- total={float(row.get('reward_total', 0.0)):.3f} "
                f"meter={float(row.get('reward_meter', 0.0)):.3f} "
                f"count={float(row.get('reward_count_adherence', row.get('reward_exact_count_bonus', 0.0))):.3f} "
                f"repeat={float(row.get('reward_repeat_soft', row.get('reward_repeat_penalty', 0.0))):.3f} "
                f"text={preview}"
            )
        examples.append("\n".join(lines))
    return examples


def count_generation_patterns(run_dir: Path, milestone: int):
    counts = defaultdict(int)
    for row in read_jsonl(run_dir / "all_generations.jsonl"):
        if bool(row.get("incomplete_reward_batch", False)):
            continue
        if str(row.get("mode", "")) != "train":
            continue
        step = generation_step(row)
        if step is None or int(step) != milestone:
            continue
        meter = float(row.get("reward_meter", 0.0))
        count_score = float(row.get("reward_count_adherence", row.get("reward_exact_count_bonus", 0.0)))
        repeat = float(row.get("reward_repeat_soft", row.get("reward_repeat_penalty", 0.0)))
        if meter >= 0.8 and count_score >= 0.8:
            counts["high_meter_good_count"] += 1
        if meter >= 0.8 and count_score < 0.5:
            counts["high_meter_bad_count"] += 1
        if meter >= 0.8 and repeat < 0.5:
            counts["high_meter_repetition_risk"] += 1
        if repeat <= 0.2:
            counts["severe_repeat_hack"] += 1
    return counts


def describe_step(train_row):
    components = {
        "meter": float(train_row.get("reward_meter_mean", 0.0)),
        "count_adherence": float(train_row.get("reward_count_adherence_mean", train_row.get("reward_exact_count_bonus_mean", 0.0))),
        "hard_gate": float(train_row.get("reward_hard_gate_mean", 0.0)),
        "judge_quality": float(train_row.get("reward_judge_quality_mean", 0.0)),
        "repeat_soft": float(train_row.get("reward_repeat_soft_mean", train_row.get("reward_repeat_penalty_mean", 0.0))),
    }
    if train_row.get("reward_judge_meaning_fit_mean") is not None:
        components["judge_meaning_fit"] = float(train_row.get("reward_judge_meaning_fit_mean", 0.0))
    if train_row.get("reward_judge_naturalness_mean") is not None:
        components["judge_naturalness"] = float(train_row.get("reward_judge_naturalness_mean", 0.0))
    weakest = min(components.items(), key=lambda item: item[1])
    strongest = max(components.items(), key=lambda item: item[1])
    return strongest, weakest


def build_step_summary(run_dir: Path, milestone: int, metrics_rows, step_interval: int):
    train_row = latest_metric_row(metrics_rows, "train", milestone)
    latest_eval = None
    for row in eval_rows(metrics_rows):
        if int(row.get("global_step", 0) or 0) <= milestone:
            latest_eval = row
    previous_train = latest_metric_row(metrics_rows, "train", max(0, milestone - step_interval))
    train_history = [row for row in train_rows(metrics_rows) if int(row.get("global_step", 0) or 0) <= milestone]
    recent_mean, recent_std = window_average(train_history, "reward_total_mean", last_n=10)
    strongest, weakest = describe_step(train_row)
    patterns = count_generation_patterns(run_dir, milestone)
    examples = candidate_examples(run_dir, milestone, max_examples=1)

    lines = []
    lines.append(f"Run: {run_dir.name}")
    lines.append(f"Run dir: {run_dir}")
    lines.append(f"Milestone step: {milestone}")
    lines.append("")
    lines.append("Train summary:")
    lines.append(f"- total={float(train_row.get('reward_total_mean', 0.0)):.4f}")
    lines.append(f"- meter={float(train_row.get('reward_meter_mean', 0.0)):.4f}")
    lines.append(
        f"- count adherence={metric_value(train_row, 'reward_count_adherence_mean', 'reward_exact_count_bonus_mean'):.4f}"
    )
    if train_row.get("reward_hard_gate_mean") is not None:
        lines.append(f"- hard gate={metric_value(train_row, 'reward_hard_gate_mean'):.4f}")
    if train_row.get("reward_judge_quality_mean") is not None:
        lines.append(f"- judge quality={metric_value(train_row, 'reward_judge_quality_mean'):.4f}")
    if train_row.get("reward_judge_meaning_fit_mean") is not None:
        lines.append(f"- judge meaning fit={metric_value(train_row, 'reward_judge_meaning_fit_mean'):.4f}")
    if train_row.get("reward_judge_naturalness_mean") is not None:
        lines.append(f"- judge naturalness={metric_value(train_row, 'reward_judge_naturalness_mean'):.4f}")
    if train_row.get("reward_repeat_soft_mean") is not None:
        lines.append(f"- repeat soft={metric_value(train_row, 'reward_repeat_soft_mean'):.4f}")
    if train_row.get("loss") is not None:
        lines.append(f"- loss={float(train_row.get('loss', 0.0)):.4f}")
    lines.append(f"- strongest component: {strongest[0]} ({strongest[1]:.4f})")
    lines.append(f"- weakest component: {weakest[0]} ({weakest[1]:.4f})")
    if recent_mean is not None:
        lines.append(f"- last-10-step mean total={recent_mean:.4f}, std={recent_std:.4f}")
    if previous_train:
        delta = float(train_row.get("reward_total_mean", 0.0)) - float(previous_train.get("reward_total_mean", 0.0))
        lines.append(f"- delta vs step {milestone - step_interval}: {delta:+.4f}")
    lines.append("")

    if latest_eval:
        eval_history = eval_rows(metrics_rows)
        prev_eval = previous_metric_row(metrics_rows, "eval", int(latest_eval.get("global_step", 0) or 0))
        current_eval_step = int(latest_eval.get("global_step", 0) or 0)
        previous_eval_step = int(prev_eval.get("global_step", 0) or 0) if prev_eval else None
        current_per_meter = eval_per_meter_summary(run_dir, current_eval_step)
        previous_per_meter = eval_per_meter_summary(run_dir, previous_eval_step) if previous_eval_step is not None else {}
        quality_summary = eval_quality_summary(run_dir, current_eval_step)

        lines.append("Latest eval summary:")
        lines.append(f"- eval step={current_eval_step}")
        lines.append(
            f"- total={metric_value(latest_eval, 'eval_reward_total_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_total_mean'), metric_value(prev_eval, 'eval_reward_total_mean', default=None))})"
        )
        lines.append(
            f"- meter={metric_value(latest_eval, 'eval_reward_meter_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_meter_mean'), metric_value(prev_eval, 'eval_reward_meter_mean', default=None))})"
        )
        lines.append(
            f"- count adherence={metric_value(latest_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean'), metric_value(prev_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean', default=None))})"
        )
        if latest_eval.get("eval_reward_hard_gate_mean") is not None:
            lines.append(
                f"- hard gate={metric_value(latest_eval, 'eval_reward_hard_gate_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_hard_gate_mean'), metric_value(prev_eval, 'eval_reward_hard_gate_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_repeat_soft_mean") is not None:
            lines.append(
                f"- repeat soft={metric_value(latest_eval, 'eval_reward_repeat_soft_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_repeat_soft_mean'), metric_value(prev_eval, 'eval_reward_repeat_soft_mean', default=None))})"
            )
        lines.append(
            f"- arabic clean={metric_value(latest_eval, 'eval_reward_arabic_clean_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_arabic_clean_mean'), metric_value(prev_eval, 'eval_reward_arabic_clean_mean', default=None))})"
        )
        if latest_eval.get("eval_reward_judge_quality_mean") is not None:
            lines.append(
                f"- judge quality={metric_value(latest_eval, 'eval_reward_judge_quality_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_judge_quality_mean'), metric_value(prev_eval, 'eval_reward_judge_quality_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_judge_meaning_fit_mean") is not None:
            lines.append(
                f"- judge meaning fit={metric_value(latest_eval, 'eval_reward_judge_meaning_fit_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_judge_meaning_fit_mean'), metric_value(prev_eval, 'eval_reward_judge_meaning_fit_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_judge_naturalness_mean") is not None:
            lines.append(
                f"- judge naturalness={metric_value(latest_eval, 'eval_reward_judge_naturalness_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_judge_naturalness_mean'), metric_value(prev_eval, 'eval_reward_judge_naturalness_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_lexical_plausibility_mean") is not None:
            lines.append(
                f"- lexical plausibility={metric_value(latest_eval, 'eval_reward_lexical_plausibility_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_lexical_plausibility_mean'), metric_value(prev_eval, 'eval_reward_lexical_plausibility_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_near_duplicate_penalty_mean") is not None:
            lines.append(
                f"- near-duplicate penalty={metric_value(latest_eval, 'eval_reward_near_duplicate_penalty_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_near_duplicate_penalty_mean'), metric_value(prev_eval, 'eval_reward_near_duplicate_penalty_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_opening_diversity_mean") is not None:
            lines.append(
                f"- opening diversity={metric_value(latest_eval, 'eval_reward_opening_diversity_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_opening_diversity_mean'), metric_value(prev_eval, 'eval_reward_opening_diversity_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_distinct_2_mean") is not None:
            lines.append(
                f"- distinct-2={metric_value(latest_eval, 'eval_reward_distinct_2_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_distinct_2_mean'), metric_value(prev_eval, 'eval_reward_distinct_2_mean', default=None))})"
            )
        if latest_eval.get("eval_loss") is not None:
            lines.append(f"- eval loss={float(latest_eval.get('eval_loss', 0.0)):.4f}")
        for metric_name, label in [
            ("eval_reward_total_mean", "total"),
            ("eval_reward_meter_mean", "meter"),
            ("eval_reward_count_adherence_mean", "count adherence"),
            ("eval_reward_judge_quality_mean", "judge quality"),
            ("eval_reward_judge_meaning_fit_mean", "judge meaning fit"),
            ("eval_reward_judge_naturalness_mean", "judge naturalness"),
            ("eval_reward_repeat_soft_mean", "repeat soft"),
        ]:
            for note in record_status(eval_history, metric_name, current_eval_step):
                lines.append(f"- {label}: {note}")
        if prev_eval:
            lines.append(f"- previous eval step={previous_eval_step}")
        lines.append("")
        lines.append("Quality diagnostics:")
        lines.append("- repeat soft blends exact-repeat, near-duplicate, opening-diversity, and distinct-2 into a light anti-template guardrail.")
        if quality_summary["exact_count_rate"] is not None:
            lines.append(f"- exact-count rate={quality_summary['exact_count_rate']:.4f}")
        if quality_summary["hard_gate_failure"] is not None:
            lines.append(f"- hard-gate failure rate={quality_summary['hard_gate_failure']:.4f}")
        if quality_summary["contamination_fail"] is not None:
            lines.append(f"- contamination fail rate={quality_summary['contamination_fail']:.4f}")
        if quality_summary["digit_leakage"] is not None:
            lines.append(f"- digit leakage rate={quality_summary['digit_leakage']:.4f}")
        if quality_summary["contamination_ratio"] is not None:
            lines.append(f"- contamination ratio={quality_summary['contamination_ratio']:.4f}")
        if quality_summary["artifact_free"] is not None:
            lines.append(f"- artifact-free score={quality_summary['artifact_free']:.4f}")
        lines.append("")
        lines.append("Per-meter eval averages:")
        lines.extend(per_meter_lines(current_eval_step, current_per_meter, previous_eval_step, previous_per_meter))
        lines.append("")

    lines.append("Interesting patterns:")
    lines.append(f"- high-meter good-count candidates at this step: {patterns['high_meter_good_count']}")
    lines.append(f"- high-meter bad-count candidates at this step: {patterns['high_meter_bad_count']}")
    lines.append(f"- high-meter repetition-risk candidates at this step: {patterns['high_meter_repetition_risk']}")
    lines.append(f"- severe repeat-hack candidates at this step: {patterns['severe_repeat_hack']}")
    lines.append("")
    lines.extend(examples)
    lines.append("")
    lines.append("Recent log tail:")
    lines.append(read_tail(run_dir / "train.log", max_lines=30))
    return "\n".join(lines)


def send_step_email(run_dir: Path, milestone: int, metrics_rows, step_interval: int):
    subject = f"[Shaer GRPO] step {milestone} summary for {run_dir.name}"
    body = build_step_summary(run_dir, milestone, metrics_rows, step_interval)
    return send_email(subject, body)


def build_periodic_update(run_dir: Path, metrics_rows):
    lines = [f"Run: {run_dir.name}", f"Run dir: {run_dir}", ""]
    latest_train = latest_metric_row(metrics_rows, "train")
    latest_eval = latest_metric_row(metrics_rows, "eval")
    eval_history = eval_rows(metrics_rows)
    if latest_train:
        lines.append("Latest train:")
        lines.append(f"- step={int(latest_train.get('global_step', 0) or 0)}")
        lines.append(f"- total={metric_value(latest_train, 'reward_total_mean'):.4f}")
        lines.append(f"- meter={metric_value(latest_train, 'reward_meter_mean'):.4f}")
        lines.append(f"- count adherence={metric_value(latest_train, 'reward_count_adherence_mean', 'reward_exact_count_bonus_mean'):.4f}")
        if latest_train.get("reward_hard_gate_mean") is not None:
            lines.append(f"- hard gate={metric_value(latest_train, 'reward_hard_gate_mean'):.4f}")
        if latest_train.get("reward_repeat_soft_mean") is not None:
            lines.append(f"- repeat soft={metric_value(latest_train, 'reward_repeat_soft_mean'):.4f}")
        if latest_train.get("reward_lexical_plausibility_mean") is not None:
            lines.append(f"- lexical plausibility={metric_value(latest_train, 'reward_lexical_plausibility_mean'):.4f}")
        if latest_train.get("reward_near_duplicate_penalty_mean") is not None:
            lines.append(f"- near-duplicate penalty={metric_value(latest_train, 'reward_near_duplicate_penalty_mean'):.4f}")
        if latest_train.get("reward_opening_diversity_mean") is not None:
            lines.append(f"- opening diversity={metric_value(latest_train, 'reward_opening_diversity_mean'):.4f}")
        if latest_train.get("reward_distinct_2_mean") is not None:
            lines.append(f"- distinct-2={metric_value(latest_train, 'reward_distinct_2_mean'):.4f}")
        lines.append("")
    if latest_eval:
        prev_eval = previous_metric_row(metrics_rows, "eval", int(latest_eval.get("global_step", 0) or 0))
        current_eval_step = int(latest_eval.get("global_step", 0) or 0)
        previous_eval_step = int(prev_eval.get("global_step", 0) or 0) if prev_eval else None
        current_per_meter = eval_per_meter_summary(run_dir, current_eval_step)
        previous_per_meter = eval_per_meter_summary(run_dir, previous_eval_step) if previous_eval_step is not None else {}
        quality_summary = eval_quality_summary(run_dir, current_eval_step)
        lines.append("Latest eval:")
        lines.append(f"- step={current_eval_step}")
        lines.append(
            f"- total={metric_value(latest_eval, 'eval_reward_total_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_total_mean'), metric_value(prev_eval, 'eval_reward_total_mean', default=None))})"
        )
        lines.append(
            f"- meter={metric_value(latest_eval, 'eval_reward_meter_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_meter_mean'), metric_value(prev_eval, 'eval_reward_meter_mean', default=None))})"
        )
        lines.append(
            f"- count adherence={metric_value(latest_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean'):.4f} "
            f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean'), metric_value(prev_eval, 'eval_reward_count_adherence_mean', 'eval_reward_exact_count_bonus_mean', default=None))})"
        )
        if latest_eval.get("eval_reward_hard_gate_mean") is not None:
            lines.append(
                f"- hard gate={metric_value(latest_eval, 'eval_reward_hard_gate_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_hard_gate_mean'), metric_value(prev_eval, 'eval_reward_hard_gate_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_repeat_soft_mean") is not None:
            lines.append(
                f"- repeat soft={metric_value(latest_eval, 'eval_reward_repeat_soft_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_repeat_soft_mean'), metric_value(prev_eval, 'eval_reward_repeat_soft_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_arabic_clean_mean") is not None:
            lines.append(
                f"- arabic clean={metric_value(latest_eval, 'eval_reward_arabic_clean_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_arabic_clean_mean'), metric_value(prev_eval, 'eval_reward_arabic_clean_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_lexical_plausibility_mean") is not None:
            lines.append(
                f"- lexical plausibility={metric_value(latest_eval, 'eval_reward_lexical_plausibility_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_lexical_plausibility_mean'), metric_value(prev_eval, 'eval_reward_lexical_plausibility_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_near_duplicate_penalty_mean") is not None:
            lines.append(
                f"- near-duplicate penalty={metric_value(latest_eval, 'eval_reward_near_duplicate_penalty_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_near_duplicate_penalty_mean'), metric_value(prev_eval, 'eval_reward_near_duplicate_penalty_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_opening_diversity_mean") is not None:
            lines.append(
                f"- opening diversity={metric_value(latest_eval, 'eval_reward_opening_diversity_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_opening_diversity_mean'), metric_value(prev_eval, 'eval_reward_opening_diversity_mean', default=None))})"
            )
        if latest_eval.get("eval_reward_distinct_2_mean") is not None:
            lines.append(
                f"- distinct-2={metric_value(latest_eval, 'eval_reward_distinct_2_mean'):.4f} "
                f"(delta vs prev {format_delta(metric_value(latest_eval, 'eval_reward_distinct_2_mean'), metric_value(prev_eval, 'eval_reward_distinct_2_mean', default=None))})"
            )
        for metric_name, label in [
            ("eval_reward_total_mean", "total"),
            ("eval_reward_meter_mean", "meter"),
            ("eval_reward_count_adherence_mean", "count adherence"),
            ("eval_reward_repeat_soft_mean", "repeat soft"),
        ]:
            for note in record_status(eval_history, metric_name, current_eval_step):
                lines.append(f"- {label}: {note}")
        lines.append("")
        lines.append("Quality diagnostics:")
        lines.append("- repeat soft blends exact-repeat, near-duplicate, opening-diversity, and distinct-2 into a light anti-template guardrail.")
        if quality_summary["exact_count_rate"] is not None:
            lines.append(f"- exact-count rate={quality_summary['exact_count_rate']:.4f}")
        if quality_summary["hard_gate_failure"] is not None:
            lines.append(f"- hard-gate failure rate={quality_summary['hard_gate_failure']:.4f}")
        if quality_summary["contamination_fail"] is not None:
            lines.append(f"- contamination fail rate={quality_summary['contamination_fail']:.4f}")
        if quality_summary["digit_leakage"] is not None:
            lines.append(f"- digit leakage rate={quality_summary['digit_leakage']:.4f}")
        if quality_summary["contamination_ratio"] is not None:
            lines.append(f"- contamination ratio={quality_summary['contamination_ratio']:.4f}")
        if quality_summary["artifact_free"] is not None:
            lines.append(f"- artifact-free score={quality_summary['artifact_free']:.4f}")
        lines.append("")
        lines.append("Per-meter eval averages:")
        lines.extend(per_meter_lines(current_eval_step, current_per_meter, previous_eval_step, previous_per_meter))
        lines.append("")
    lines.append("Recent log tail:")
    lines.append(read_tail(run_dir / "train.log", max_lines=60))
    return "\n".join(lines)


def main():
    load_dotenv(override=False)
    if len(sys.argv) < 2:
        print("usage: python watcher.py <log_path_or_run_dir>")
        sys.exit(1)

    target = Path(sys.argv[1]).expanduser().resolve()
    interval_min = int(os.getenv("WATCHER_INTERVAL_MINUTES", "30"))
    stale_min = int(os.getenv("WATCHER_STALE_MINUTES", "20"))
    step_interval = int(os.getenv("WATCHER_STEP_EMAIL_EVERY", "50"))

    state_path = state_path_for(str(target))
    state = load_state(state_path)
    last_sent_at = float(state.get("last_sent_at", 0.0))
    last_size = int(state.get("last_size", -1))
    last_growth_time = float(state.get("last_growth_time", time.time()))
    last_sent_milestone = int(state.get("last_sent_milestone", 0))
    last_run_id = str(state.get("last_run_id", ""))

    while True:
        exists = target.exists()
        if exists and target.is_file():
            size = target.stat().st_size
            if size != last_size:
                last_growth_time = time.time()
                last_size = size

        now = time.time()

        run_dir = discover_run_dir(target)
        if run_dir is not None:
            if run_dir.name != last_run_id:
                last_run_id = run_dir.name
                last_sent_milestone = 0
            metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
            checkpoint_events = read_jsonl(run_dir / "checkpoint_events.jsonl")
            latest_step = latest_train_step(metrics_rows)
            next_milestone = ((last_sent_milestone // step_interval) + 1) * step_interval
            if next_milestone > 0 and latest_step >= next_milestone:
                if milestone_ready(step_interval, next_milestone, metrics_rows, checkpoint_events):
                    if send_step_email(run_dir, next_milestone, metrics_rows, step_interval):
                        last_sent_milestone = next_milestone

        # periodic email
        if now - last_sent_at >= interval_min * 60:
            if run_dir is not None:
                body = build_periodic_update(run_dir, metrics_rows)
            else:
                body = f"Watcher update for target: {target}\n\n" + read_tail(target, max_lines=80)
            if send_email("[Shaer GRPO] periodic update", body):
                last_sent_at = now

        # stale alert
        if now - last_growth_time >= stale_min * 60:
            body = f"Log appears stale for >= {stale_min} minutes.\n\nTarget: {target}\n"
            if run_dir is not None:
                body += f"Run dir: {run_dir}\n\n"
                body += read_tail(run_dir / "train.log", max_lines=80)
            else:
                body += "\n" + read_tail(target, max_lines=80)
            if send_email("[Shaer GRPO] stale log alert", body):
                last_growth_time = now

        save_state(
            state_path,
            {
                "last_sent_at": last_sent_at,
                "last_size": last_size,
                "last_growth_time": last_growth_time,
                "last_sent_milestone": last_sent_milestone,
                "last_run_id": last_run_id,
            },
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
