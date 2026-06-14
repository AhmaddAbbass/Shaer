from rewards.common import append_jsonl, clamp01, extract_judge_json, extract_text, load_yaml, JudgeClient


def score_meaning_substance(generated_poem: str, prompt_file: str, cache_dir: str):
    cfg = load_yaml(prompt_file)
    client = JudgeClient(cache_dir=cache_dir, cache_namespace="meaning_substance")

    system_prompt = cfg["system"]
    user_prompt = cfg["user_template"].format(generated_poem=generated_poem)

    raw, cache_hit = client.judge(system_prompt, user_prompt)
    parsed = extract_judge_json(raw, score_key=cfg.get("score_key", "substance_score"))
    score_key = cfg.get("score_key", "substance_score")
    return {
        "score": clamp01(parsed.get(score_key, 0.0)),
        "score_key": score_key,
        "notes": parsed.get("notes", ""),
        "cache_hit": cache_hit,
        "latency_sec": raw.get("latency_sec", None),
        "error": parsed.get("error"),
        "mode_respected": parsed.get("mode_respected"),
        "reasoning_tokens": parsed.get("reasoning_tokens"),
        "raw": parsed,
    }


def make_meaning_substance_reward(run_dir: str, prompt_file: str, cache_dir: str, recorder=None):
    debug_path = f"{run_dir}/reward_meaning_substance_debug.jsonl"

    def reward_fn(prompts=None, completions=None, dataset_split=None, **kwargs):
        completions = completions or []
        dataset_split = dataset_split or kwargs.get("dataset_split", [])
        if not isinstance(dataset_split, list):
            dataset_split = [dataset_split] * len(completions)

        rewards = []
        extras = []
        for i, completion in enumerate(completions):
            poem = extract_text(completion)
            out = score_meaning_substance(poem, prompt_file=prompt_file, cache_dir=cache_dir)
            rewards.append(out["score"])
            extra = {
                "notes": out["notes"],
                "cache_hit": out["cache_hit"],
                "latency_sec": out["latency_sec"],
                "error": out["error"],
                "mode_respected": out["mode_respected"],
                "reasoning_tokens": out["reasoning_tokens"],
                "poem_preview": poem[:400],
                "dataset_split": dataset_split[i] if i < len(dataset_split) else "",
            }
            extras.append(extra)
            append_jsonl(debug_path, {"i": i, "score": out["score"], **extra})

        if recorder is not None:
            recorder.record_reward_batch(
                reward_name="meaning_substance",
                completions=completions,
                prompts=prompts,
                scores=rewards,
                extras=extras,
                kwargs={
                    **kwargs,
                    "dataset_split": dataset_split,
                },
            )
        return rewards

    reward_fn.__name__ = "meaning_substance"
    return reward_fn
