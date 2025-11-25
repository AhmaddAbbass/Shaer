# utils/rewards/meaning_reward.py

import os
import re

from vllm import LLM, SamplingParams

from .form_reward import _extract_text_from_completion


_llm = None
_llm_tokenizer = None
_judge_sampling = None


def _get_llm():
    """
    Lazy-init Yehia vLLM instance once, reuse across reward calls.
    """
    global _llm, _llm_tokenizer, _judge_sampling

    if _llm is not None:
        return _llm, _llm_tokenizer, _judge_sampling

    model_path = os.environ.get("YEHIA_VLLM_MODEL_PATH", "Navid-AI/Yehia-7B-preview")
    gpu_util = float(os.environ.get("YEHIA_VLLM_GPU_UTIL", "0.9"))

    print(f"[meaning_reward] Initializing vLLM LLM from: {model_path}")
    print(f"[meaning_reward] vLLM GPU utilization: {gpu_util}")

    _llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_util,
        max_model_len=4096,
    )

    _llm_tokenizer = _llm.get_tokenizer()

    _judge_sampling = SamplingParams(
        max_tokens=8,             # just want a scalar like 0.83
        temperature=0.0,          # deterministic judge
        top_p=1.0,
        truncate_prompt_tokens=512,
    )

    return _llm, _llm_tokenizer, _judge_sampling


_SCORE_RE = re.compile(r"([01](?:\.\d+)?)")


def _parse_score(text):
    """
    Extract a float in [0, 1] from judge output.
    If it fails, return 0.0.
    """
    if not text:
        return 0.0

    m = _SCORE_RE.search(text)
    if not m:
        return 0.0

    try:
        val = float(m.group(1))
    except Exception:
        return 0.0

    # clamp just in case
    if val < 0.0:
        val = 0.0
    if val > 1.0:
        val = 1.0
    return val


def _normalize_arabic(s):
    if not isinstance(s, str):
        return ""
    t = s.replace("ـ", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def meaning_reward(completions, prompts, poem_description=None, trainer_state=None, **kwargs):
    """
    Meaning / semantic alignment reward.

    Idea:
      - Use Yehia (vLLM) as an Arabic judge.
      - For each sample, ask:
          "Here is a description of the poem, and here is the generated bayt.
           Return a single number in [0, 1] measuring semantic alignment."
      - Parse that scalar and use it as reward.

    Inputs:
      - completions: list of generated answers (strings or chat-style dicts)
      - prompts: list of prompts (we don't really use them here)
      - poem_description: list[str] from dataset column

    Returns:
      - list[float] in [0, 1]
    """
    llm, tokenizer, sampling = _get_llm()

    n = len(completions)
    rewards = [0.0] * n

    # Normalize descriptions to list
    if poem_description is None:
        descs = [None] * n
    else:
        # TRL passes dataset columns as lists already
        descs = list(poem_description)
        if len(descs) != n:
            # just in case, pad/trim
            descs = (descs + [None] * n)[:n]

    # Build prompts for vLLM in one shot (batched)
    prompts_text = []

    system_prompt = (
        "أنت ناقد شعري عربي، متخصّص في تقييم مدى التزام الأبيات بمضمون الوصف المطلوب. "
        "مهمّتك أن تعطي درجة واحدة بين 0 و 1 فقط."
    )

    for i in range(n):
        verse_raw = _extract_text_from_completion(completions[i])
        verse = _normalize_arabic(verse_raw)

        if not verse:
            # empty answer → we keep reward 0
            prompts_text.append(
                "أرجع العدد 0 فقط بلا شرح."
            )
            continue

        desc = descs[i]
        desc_norm = _normalize_arabic(desc) if desc is not None else ""

        if not desc_norm:
            # no description, we can only give a weak judgment
            desc_norm = "الوصف غير متوفر، لكن حاول قياس منطقية البيت بشكل عام."

        user_content = (
            "الوصف المطلوب للقصيدة:\n"
            f"{desc_norm}\n\n"
            "البيت المقترح:\n"
            f"{verse}\n\n"
            "المطلوب:\n"
            "- قيّم مدى انسجام هذا البيت مع الوصف من حيث المعنى والجو الشعوري.\n"
            "- أرجع عدداً حقيقياً واحداً بين 0 و 1 فقط:\n"
            "  0 يعني أن البيت لا علاقة له تقريباً بالوصف.\n"
            "  1 يعني أن البيت منسجم جداً مع الوصف ومعناه مناسب بالكامل.\n"
            "اكتب العدد فقط بدون أي كلمات أو شرح إضافي."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts_text.append(prompt)

    # Run batched generation
    outputs = llm.generate(prompts_text, sampling, use_tqdm=False)

    for i, out in enumerate(outputs):
        if out.outputs:
            raw_text = (out.outputs[0].text or "").strip()
        else:
            raw_text = ""
        rewards[i] = _parse_score(raw_text)

    return rewards
