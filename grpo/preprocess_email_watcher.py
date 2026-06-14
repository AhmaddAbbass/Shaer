import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watcher import send_email, load_state, save_state, state_path_for


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith('Z'):
            return datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except Exception:
        return None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def ensure_state_shape(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault('created_at', utc_now_iso())
    state.setdefault('last_summary_sent_at', None)
    state.setdefault('alert_flags', {})
    state.setdefault('file_offsets', {})
    state.setdefault('generated_counts', {'train': 0, 'eval': 0, 'test': 0})
    state.setdefault('judged_counts', {'train': 0, 'eval': 0, 'test': 0})
    state.setdefault('difficulty_counts', {'train': {}, 'eval': {}, 'test': {}})
    state.setdefault('meter_stats', {'train': {}, 'eval': {}, 'test': {}})
    state.setdefault('history', [])
    return state


def append_history(state: dict[str, Any], generated_total: int, judged_total: int):
    history = state.setdefault('history', [])
    now_epoch = int(time.time())
    history.append({'t': now_epoch, 'generated_total': int(generated_total), 'judged_total': int(judged_total)})
    cutoff = now_epoch - 48 * 3600
    state['history'] = [row for row in history if int(row.get('t', 0)) >= cutoff][-256:]


LINE_ENDINGS_RE = re.compile(rb'\n')


def count_new_lines(path: Path, start_offset: int) -> tuple[int, int]:
    if not path.exists():
        return 0, start_offset
    size = path.stat().st_size
    if start_offset > size:
        start_offset = 0
    with path.open('rb') as f:
        f.seek(start_offset)
        data = f.read()
        end = f.tell()
    return len(LINE_ENDINGS_RE.findall(data)), end


def iter_new_jsonl_rows(path: Path, start_offset: int):
    if not path.exists():
        return [], start_offset
    size = path.stat().st_size
    if start_offset > size:
        start_offset = 0
    rows = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        end = f.tell()
    return rows, end


def update_meter_stats(state: dict[str, Any], split_name: str, row: dict[str, Any]):
    split_stats = state['meter_stats'].setdefault(split_name, {})
    meter = str(row.get('base_meter') or row.get('meter_label') or 'unknown')
    bucket = split_stats.setdefault(
        meter,
        {
            'rows': 0,
            'meter_sum': 0.0,
            'weighted_total_sum': 0.0,
            'fit_sum': 0.0,
            'substance_sum': 0.0,
            'count_sum': 0.0,
        },
    )
    bucket['rows'] += 1
    bucket['meter_sum'] += float(row.get('meter_mean', 0.0) or 0.0)
    bucket['weighted_total_sum'] += float(row.get('weighted_total_mean', 0.0) or 0.0)
    bucket['fit_sum'] += float(row.get('fit_mean', 0.0) or 0.0)
    bucket['substance_sum'] += float(row.get('substance_mean', 0.0) or 0.0)
    bucket['count_sum'] += float(row.get('count_mean', 0.0) or 0.0)


def update_from_judged_rows(state: dict[str, Any], split_name: str, rows: list[dict[str, Any]]):
    for row in rows:
        state['judged_counts'][split_name] = int(state['judged_counts'].get(split_name, 0)) + 1
        bucket = str(row.get('difficulty_bucket', 'unknown'))
        split_counts = state['difficulty_counts'].setdefault(split_name, {})
        split_counts[bucket] = int(split_counts.get(bucket, 0)) + 1
        update_meter_stats(state, split_name, row)


ERROR_PATTERNS = (
    'Traceback',
    'ERROR',
    'generation_failed',
    'judge_failed',
)


def scan_new_error_lines(path: Path, start_offset: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], start_offset
    size = path.stat().st_size
    if start_offset > size:
        start_offset = 0
    matches: list[str] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        f.seek(start_offset)
        for line in f:
            if any(token in line for token in ERROR_PATTERNS):
                matches.append(line.rstrip())
        end = f.tell()
    return matches[-50:], end


def process_count(pattern: str) -> int:
    result = subprocess.run(['bash', '-lc', f"pgrep -fc {json.dumps(pattern)} || true"], capture_output=True, text=True)
    try:
        return int((result.stdout or '0').strip() or '0')
    except Exception:
        return 0


def aggregate_meter_rows(meter_stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for meter, stats in meter_stats.items():
        n = max(1, int(stats.get('rows', 0)))
        rows.append(
            {
                'meter': meter,
                'rows': int(stats.get('rows', 0)),
                'avg_meter': float(stats.get('meter_sum', 0.0)) / n,
                'avg_total': float(stats.get('weighted_total_sum', 0.0)) / n,
                'avg_fit': float(stats.get('fit_sum', 0.0)) / n,
                'avg_substance': float(stats.get('substance_sum', 0.0)) / n,
                'avg_count': float(stats.get('count_sum', 0.0)) / n,
            }
        )
    return rows


def format_meter_table(rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        lines.append(
            f"- {row['meter']}: rows={row['rows']} total={row['avg_total']:.3f} meter={row['avg_meter']:.3f} fit={row['avg_fit']:.3f} substance={row['avg_substance']:.3f} count={row['avg_count']:.3f}"
        )
    return '\n'.join(lines) if lines else '- none yet'


def slope_eta(state: dict[str, Any], total_target: int, kind: str) -> str:
    history = state.get('history', [])
    if len(history) < 2:
        return 'unknown'
    latest = history[-1]
    cutoff = latest['t'] - 1800
    baseline = None
    for row in history:
        if int(row.get('t', 0)) <= cutoff:
            baseline = row
    if baseline is None:
        baseline = history[0]
    delta_t = max(1, int(latest['t']) - int(baseline['t']))
    key = f'{kind}_total'
    delta_n = int(latest.get(key, 0)) - int(baseline.get(key, 0))
    if delta_n <= 0:
        return 'unknown'
    remaining = max(0, int(total_target) - int(latest.get(key, 0)))
    seconds = remaining / (delta_n / delta_t)
    hours = seconds / 3600.0
    if hours < 1:
        return f'{seconds/60.0:.1f}m'
    if hours < 48:
        return f'{hours:.1f}h'
    return f'{hours/24.0:.1f}d'


def stale_minutes_for(path: Path, fallback_minutes: int) -> float:
    if not path.exists():
        return 1e9
    return max(0.0, (time.time() - path.stat().st_mtime) / 60.0)


def body_for_summary(run_dir: Path, state: dict[str, Any], current: dict[str, Any], new_errors: list[str]) -> str:
    cfg = read_json(run_dir / 'config_snapshot.json') or {}
    prep_cfg = cfg.get('preprocess_grpo_pipeline', {})
    gen_total_target = 0
    gen_rows = []
    for split in ('train', 'eval', 'test'):
        generated = int(state['generated_counts'].get(split, 0))
        judged = int(state['judged_counts'].get(split, 0))
        total = int(current['totals'].get(split, 0))
        gen_total_target += total
        backlog = max(0, generated - judged)
        gen_rows.append(f"- {split}: generated={generated}/{total} judged={judged} backlog={backlog}")

    gen_eta = slope_eta(state, gen_total_target, 'generated')
    judge_eta = slope_eta(state, gen_total_target, 'judged')

    all_meter_rows = []
    for split in ('train', 'eval', 'test'):
        all_meter_rows.extend(aggregate_meter_rows(state['meter_stats'].get(split, {})))
    merged = {}
    for row in all_meter_rows:
        meter = row['meter']
        bucket = merged.setdefault(meter, {'rows': 0, 'meter_sum': 0.0, 'total_sum': 0.0, 'fit_sum': 0.0, 'substance_sum': 0.0, 'count_sum': 0.0})
        bucket['rows'] += row['rows']
        bucket['meter_sum'] += row['avg_meter'] * row['rows']
        bucket['total_sum'] += row['avg_total'] * row['rows']
        bucket['fit_sum'] += row['avg_fit'] * row['rows']
        bucket['substance_sum'] += row['avg_substance'] * row['rows']
        bucket['count_sum'] += row['avg_count'] * row['rows']
    merged_rows = []
    for meter, stats in merged.items():
        n = max(1, stats['rows'])
        merged_rows.append({'meter': meter, 'rows': n, 'avg_meter': stats['meter_sum']/n, 'avg_total': stats['total_sum']/n, 'avg_fit': stats['fit_sum']/n, 'avg_substance': stats['substance_sum']/n, 'avg_count': stats['count_sum']/n})
    merged_rows.sort(key=lambda row: (-row['rows'], row['meter']))
    strongest = sorted([row for row in merged_rows if row['rows'] >= 3], key=lambda row: row['avg_meter'], reverse=True)[:5]
    weakest = sorted([row for row in merged_rows if row['rows'] >= 3], key=lambda row: row['avg_meter'])[:5]

    diff_lines = []
    for split in ('train', 'eval', 'test'):
        counts = Counter(state['difficulty_counts'].get(split, {}))
        total = sum(counts.values())
        if total == 0:
            continue
        diff_lines.append(f"- {split}: easy={counts.get('easy', 0)} medium={counts.get('medium', 0)} hard={counts.get('hard', 0)}")

    lines = [
        f'Run: {run_dir}',
        f'Time: {utc_now_iso()}',
        '',
        'Pipeline status:',
        f"- generator processes: {current['generator_proc_count']}",
        f"- judge processes: {current['judge_proc_count']}",
        f"- vllm engine processes: {current['engine_proc_count']}",
        f"- generator file stale: {current['generator_stale_minutes']:.1f} min",
        f"- judged file stale: {current['judged_stale_minutes']:.1f} min",
        '',
        'Config:',
        f"- num_generations: {prep_cfg.get('num_generations', 'unknown')}",
        f"- generator_batch_size: {prep_cfg.get('generator_batch_size', 'unknown')}",
        '',
        'Progress by split:',
        *gen_rows,
        '',
        'ETA (rolling, approximate):',
        f'- generation: {gen_eta}',
        f'- judging: {judge_eta}',
        '',
        'Difficulty buckets so far:',
        *(diff_lines or ['- none yet']),
        '',
        'Strongest meters so far by avg meter score:',
        format_meter_table(strongest),
        '',
        'Weakest meters so far by avg meter score:',
        format_meter_table(weakest),
    ]
    if new_errors:
        lines.extend(['', 'New errors seen since last email:', *[f'- {line[:500]}' for line in new_errors[-10:]]])
    return '\n'.join(lines)


def collect_current(run_dir: Path) -> dict[str, Any]:
    totals = {}
    status = read_json(run_dir / 'generator_status.json') or {}
    current_split = status.get('split')
    if current_split and status.get('total_rows'):
        totals[str(current_split)] = int(status['total_rows'])
    return {
        'generator_proc_count': process_count(f"generate_grpo_candidates.py --run-dir {run_dir}"),
        'judge_proc_count': process_count(f"judge_grpo_candidates.py --run-dir {run_dir}"),
        'engine_proc_count': process_count('VLLM::EngineCore'),
        'generator_stale_minutes': stale_minutes_for(run_dir / 'train_generations.jsonl', int(os.getenv('WATCHER_STALE_MINUTES', '20'))),
        'judged_stale_minutes': stale_minutes_for(run_dir / 'train.jsonl', int(os.getenv('WATCHER_STALE_MINUTES', '20'))),
        'totals': totals,
        'status': status,
    }


def maybe_send(subject: str, body: str, always_print: bool = True) -> bool:
    ok = send_email(subject, body)
    if always_print:
        print(f'[{utc_now_iso()}] email_sent={ok} subject={subject}', flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--summary-minutes', type=int, default=None)
    parser.add_argument('--stale-minutes', type=int, default=None)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary_minutes = args.summary_minutes or int(os.getenv('WATCHER_INTERVAL_MINUTES', '30'))
    stale_minutes = args.stale_minutes or int(os.getenv('WATCHER_STALE_MINUTES', '20'))
    state_path = state_path_for(str(run_dir) + '::preprocess_email_watcher')
    state = ensure_state_shape(load_state(state_path))
    state['run_dir'] = str(run_dir)

    watched_logs = [
        run_dir / 'generator.log',
        run_dir / 'generator_detached.log',
        run_dir / 'judge.log',
        run_dir / 'judge_detached.log',
    ] + sorted(run_dir.glob('judge*_detached.log')) + sorted(run_dir.glob('judge_*.log'))

    split_generated_paths = {split: run_dir / f'{split}_generations.jsonl' for split in ('train', 'eval', 'test')}
    split_judged_paths = {split: run_dir / f'{split}.jsonl' for split in ('train', 'eval', 'test')}

    while True:
        new_errors: list[str] = []
        for split, path in split_generated_paths.items():
            key = f'generated::{split}'
            old_offset = int(state['file_offsets'].get(key, 0))
            new_lines, new_offset = count_new_lines(path, old_offset)
            state['file_offsets'][key] = new_offset
            state['generated_counts'][split] = int(state['generated_counts'].get(split, 0)) + int(new_lines)

        for split, path in split_judged_paths.items():
            key = f'judged::{split}'
            old_offset = int(state['file_offsets'].get(key, 0))
            new_rows, new_offset = iter_new_jsonl_rows(path, old_offset)
            state['file_offsets'][key] = new_offset
            update_from_judged_rows(state, split, new_rows)

        for path in watched_logs + [run_dir / 'train_generation_failures.jsonl', run_dir / 'train_judge_failures.jsonl', run_dir / 'eval_generation_failures.jsonl', run_dir / 'eval_judge_failures.jsonl', run_dir / 'test_generation_failures.jsonl', run_dir / 'test_judge_failures.jsonl']:
            key = f'log::{path}'
            old_offset = int(state['file_offsets'].get(key, 0))
            matches, new_offset = scan_new_error_lines(path, old_offset)
            state['file_offsets'][key] = new_offset
            if matches:
                new_errors.extend([f'{path.name}: {line}' for line in matches[-10:]])

        generated_total = sum(int(v) for v in state['generated_counts'].values())
        judged_total = sum(int(v) for v in state['judged_counts'].values())
        append_history(state, generated_total, judged_total)

        current = collect_current(run_dir)
        subject_prefix = '[Shaer][GRPO Preprocess]'
        flags = state.setdefault('alert_flags', {})

        generator_stale = current['generator_stale_minutes'] >= stale_minutes and current['generator_proc_count'] <= 0
        if generator_stale and not flags.get('generator_stale_active', False):
            body = body_for_summary(run_dir, state, current, new_errors)
            maybe_send(f"{subject_prefix} Generator Stalled", body)
            flags['generator_stale_active'] = True
        elif not generator_stale and flags.get('generator_stale_active', False):
            body = body_for_summary(run_dir, state, current, new_errors)
            maybe_send(f"{subject_prefix} Generator Recovered", body)
            flags['generator_stale_active'] = False

        judges_stale = current['judged_stale_minutes'] >= stale_minutes and current['judge_proc_count'] <= 0
        if judges_stale and not flags.get('judges_stale_active', False):
            body = body_for_summary(run_dir, state, current, new_errors)
            maybe_send(f"{subject_prefix} Judges Stalled", body)
            flags['judges_stale_active'] = True
        elif not judges_stale and flags.get('judges_stale_active', False):
            body = body_for_summary(run_dir, state, current, new_errors)
            maybe_send(f"{subject_prefix} Judges Recovered", body)
            flags['judges_stale_active'] = False

        if new_errors:
            body = body_for_summary(run_dir, state, current, new_errors)
            maybe_send(f"{subject_prefix} New Errors ({len(new_errors)})", body)

        last_summary_at = parse_ts(state.get('last_summary_sent_at'))
        due = last_summary_at is None or (utc_now() - last_summary_at).total_seconds() >= summary_minutes * 60
        if due or args.once:
            body = body_for_summary(run_dir, state, current, new_errors)
            if maybe_send(f"{subject_prefix} Progress Update", body):
                state['last_summary_sent_at'] = utc_now_iso()

        save_state(state_path, state)
        if args.once:
            break
        time.sleep(max(15, int(args.poll_seconds)))


if __name__ == '__main__':
    main()
