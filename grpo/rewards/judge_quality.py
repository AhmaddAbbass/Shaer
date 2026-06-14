from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rewards.common import clamp01, extract_judge_json, load_yaml, JudgeClient


def score_judge_quality(
    description: str,
    generated_poem: str,
    prompt_file: str,
    cache_dir: str,
    cache_namespace: str = "judge_quality",
):
    cfg = load_yaml(prompt_file)
    client = JudgeClient(cache_dir=cache_dir, cache_namespace=cache_namespace)

    system_prompt = cfg["system"]
    user_prompt = cfg["user_template"].format(
        description=description,
        generated_poem=generated_poem,
    )

    raw, cache_hit = client.judge(system_prompt, user_prompt)
    score_key = cfg.get("score_key", "judge_quality")
    parsed = extract_judge_json(raw, score_key=score_key)
    return {
        "score": clamp01(parsed.get(score_key, 0.0)),
        "score_key": score_key,
        "failure_mode": str(parsed.get("failure_mode", "") or ""),
        "notes": str(parsed.get("notes", "") or ""),
        "cache_hit": cache_hit,
        "latency_sec": raw.get("latency_sec", None),
        "error": parsed.get("error"),
        "mode_respected": parsed.get("mode_respected"),
        "reasoning_tokens": parsed.get("reasoning_tokens"),
        "raw": parsed,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_name": cfg.get("name", Path(prompt_file).stem),
        "prompt_file": str(prompt_file),
    }


def batch_score_judge_quality(
    *,
    descriptions: list[str],
    poems: list[str],
    prompt_file: str,
    cache_dir: str,
    cache_namespace: str = "judge_quality",
    max_workers: int = 8,
):
    size = max(len(descriptions), len(poems))
    if size == 0:
        return []
    if len(descriptions) != size or len(poems) != size:
        raise ValueError("descriptions and poems must have the same length")

    max_workers = max(1, min(int(max_workers or 1), size))
    results = [None] * size

    def worker(index: int):
        return index, score_judge_quality(
            description=descriptions[index],
            generated_poem=poems[index],
            prompt_file=prompt_file,
            cache_dir=cache_dir,
            cache_namespace=cache_namespace,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, idx) for idx in range(size)]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result

    return results
