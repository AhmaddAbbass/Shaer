import argparse
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg: str, log_path: Path):
    line = f"{utc_now_iso()} | {msg}\n"
    with log_path.open('a', encoding='utf-8') as f:
        f.write(line)
    print(line, end='', flush=True)


def pids_for(pattern: str) -> list[int]:
    result = subprocess.run(['bash', '-lc', f"pgrep -f {repr(pattern)} || true"], capture_output=True, text=True)
    pids = []
    for part in (result.stdout or '').split():
        try:
            pids.append(int(part))
        except Exception:
            pass
    return pids


def file_stale_minutes(path: Path) -> float:
    if not path.exists():
        return 1e9
    return max(0.0, (time.time() - path.stat().st_mtime) / 60.0)


def spawn_generator(root: Path, run_dir: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env.setdefault('SFT_ADAPTER_REPO', 'Shaer-AI/Shaer-adapters')
    env.setdefault('SFT_ADAPTER_MODE', 'fresh_sft/train')
    out = (run_dir / 'generator_detached.log').open('ab')
    proc = subprocess.Popen(
        ['../sft/.venv/bin/python', '-u', 'generate_grpo_candidates.py', '--run-dir', str(run_dir)],
        cwd=str(root),
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    (run_dir / 'generator_supervised.pid').write_text(str(proc.pid), encoding='utf-8')
    log(f'started generator pid={proc.pid}', log_path)
    return proc.pid


def spawn_judge(root: Path, run_dir: Path, worker_id: str, publish: bool, log_path: Path) -> int:
    out = (run_dir / f'{worker_id}_detached.log').open('ab')
    cmd = ['../sft/.venv/bin/python', '-u', 'judge_grpo_candidates.py', '--run-dir', str(run_dir), '--worker-id', worker_id]
    if publish:
        cmd.append('--publish')
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=os.environ.copy(),
    )
    log(f'started {worker_id} pid={proc.pid} publish={publish}', log_path)
    return proc.pid


def kill_pids(pids: list[int], log_path: Path, label: str):
    if not pids:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    log(f'sent SIGTERM to {label} pids={pids}', log_path)
    time.sleep(2)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    log(f'ensured stopped {label} pids={pids}', log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--judge-workers', type=int, default=8)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--generator-stale-minutes', type=int, default=10)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    root = Path(__file__).resolve().parent
    sup_log = run_dir / 'preprocess_supervisor.log'
    desired_workers = [f'judge{i:02d}' for i in range(args.judge_workers)]

    while True:
        gen_pattern = f'generate_grpo_candidates.py --run-dir {run_dir}'
        gen_pids = pids_for(gen_pattern)
        engine_pids = pids_for('VLLM::EngineCore')
        gen_stale = file_stale_minutes(run_dir / 'train_generations.jsonl')

        if (not gen_pids) and gen_stale >= args.generator_stale_minutes:
            if engine_pids:
                kill_pids(engine_pids, sup_log, 'orphan_vllm_engine')
            spawn_generator(root, run_dir, sup_log)

        for i, worker_id in enumerate(desired_workers):
            pattern = f'judge_grpo_candidates.py --run-dir {run_dir} --worker-id {worker_id}'
            pids = pids_for(pattern)
            if not pids:
                spawn_judge(root, run_dir, worker_id, publish=(i == 0), log_path=sup_log)

        time.sleep(max(15, args.poll_seconds))


if __name__ == '__main__':
    main()
