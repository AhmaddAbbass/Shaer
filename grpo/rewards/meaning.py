from rewards.common import append_jsonl, clamp01, extract_judge_json, extract_text, load_yaml, JudgeClient


def score_meaning(description: str, generated_poem: str, prompt_file: str, cache_dir: str):
    cfg = load_yaml(prompt_file)
    client = JudgeClient(cache_dir=cache_dir)

    system_prompt = cfg["system"]
    user_prompt = cfg["user_template"].format(
        description=description,
        generated_poem=generated_poem,
    )

    raw, cache_hit = client.judge(system_prompt, user_prompt)
    parsed = extract_judge_json(raw)
    return {
        "score": clamp01(parsed.get("score", 0.0)),
        "notes": parsed.get("notes", ""),
        "cache_hit": cache_hit,
        "latency_sec": raw.get("latency_sec", None),
        "raw": parsed,
    }


def make_meaning_reward(run_dir: str, prompt_file: str, cache_dir: str):
    debug_path = f"{run_dir}/reward_meaning_debug.jsonl"

    def reward_fn(prompts=None, completions=None, description=None, **kwargs):
        completions = completions or []
        description = description or kwargs.get("description", [])
        if not isinstance(description, list):
            description = [description] * len(completions)

        rewards = []
        for i, completion in enumerate(completions):
            poem = extract_text(completion)
            desc = description[i] if i < len(description) else ""
            out = score_meaning(desc, poem, prompt_file=prompt_file, cache_dir=cache_dir)
            rewards.append(out["score"])
            append_jsonl(debug_path, {
                "i": i,
                "score": out["score"],
                "notes": out["notes"],
                "cache_hit": out["cache_hit"],
                "latency_sec": out["latency_sec"],
                "description_preview": str(desc)[:300],
                "poem_preview": poem[:400],
            })
        return rewards

    reward_fn.__name__ = "meaning"
    return reward_fn
