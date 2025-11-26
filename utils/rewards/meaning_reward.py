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

    # Default: push vLLM to GPU 1 so GPU 0 stays free for the main model.
    worker_devices = os.environ.setdefault("VLLM_WORKER_CUDA_DEVICES", "1")
    # Also mask visibility for vLLM to avoid allocating on GPU 0.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", worker_devices)
    model_path = os.environ.get("YEHIA_VLLM_MODEL_PATH", "Navid-AI/Yehia-7B-preview")
    gpu_util = float(os.environ.get("YEHIA_VLLM_GPU_UTIL", "0.6"))
    max_model_len = int(os.environ.get("YEHIA_VLLM_MAX_MODEL_LEN", "1024"))

    def _init_llm(gu, mlen):
        print(f"[meaning_reward] Initializing vLLM LLM from: {model_path}")
        print(f"[meaning_reward] vLLM GPU utilization: {gu} | max_model_len: {mlen}")
        return LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=gu,
            max_model_len=mlen,
        )

    try:
        _llm = _init_llm(gpu_util, max_model_len)
    except ValueError as e:
        # Recover from insufficient cache allocation by using more GPU and/or shorter context.
        msg = str(e)
        if "No available memory for the cache blocks" not in msg:
            raise

        fallback_gu = min(0.95, gpu_util + 0.2)
        fallback_len = max(1024, max_model_len // 2)
        print(
            f"[meaning_reward] Retry vLLM init with higher gpu_memory_utilization={fallback_gu} "
            f"and max_model_len={fallback_len} due to: {e}"
        )
        _llm = _init_llm(fallback_gu, fallback_len)

    _llm_tokenizer = _llm.get_tokenizer()

    # Judge: we only need a small numeric answer
    _judge_sampling = SamplingParams(
        max_tokens=8,             # enough for "9.3" or "10"
        temperature=0.0,          # deterministic judge
        top_p=1.0,
        truncate_prompt_tokens=512,
    )

    return _llm, _llm_tokenizer, _judge_sampling


def _normalize_arabic(s):
    if not isinstance(s, str):
        return ""
    t = s.replace("ـ", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _parse_score_0_to_10(text):
    """
    Extract a float in [0, 10] from judge output.
    If parsing fails, return 0.0.
    """
    if not text:
        return 0.0

    # Find all numbers like 7, 7.5, 10, 3.25, etc.
    candidates = re.findall(r"(\d+(?:\.\d+)?)", text)
    for c in candidates:
        try:
            val = float(c)
        except Exception:
            continue

        if 0.0 <= val <= 10.0:
            return val

    return 0.0


def meaning_reward(completions, prompts, poem_description=None, trainer_state=None, **kwargs):
    """
    Meaning / semantic alignment reward.

    Uses Yehia (via vLLM) as a judge.

    For each sample:
      - description = poem_description[i]
      - verse       = completions[i]
      - Ask Yehia: "Give me a single number between 0 and 10 describing semantic alignment."

    Returns:
      - list[float] in [0, 10]
    """
    llm, tokenizer, sampling = _get_llm()

    n = len(completions)
    rewards = [0.0] * n

    # Normalize descriptions to a list aligned with completions
    if poem_description is None:
        descs = [None] * n
    else:
        descs = list(poem_description)
        if len(descs) < n:
            descs = descs + [None] * (n - len(descs))
        elif len(descs) > n:
            descs = descs[:n]

    system_prompt = (
        "أنت ناقد شعري عربي متخصّص في تقييم مدى التزام الأبيات بالمضمون المطلوب. "
        "مهمّتك أن تعطي رقمًا واحدًا بين 0 و 10 يعبّر عن مدى انسجام البيت مع الوصف من حيث المعنى والموضوع والجو الشعوري."
    )

    prompts_text = []

    for i in range(n):
        verse_raw = _extract_text_from_completion(completions[i])
        verse = _normalize_arabic(verse_raw)

        desc = descs[i]
        desc_norm = _normalize_arabic(desc) if desc is not None else ""

        if not desc_norm:
            # No description → we evaluate only the clarity and coherence of the verse
            desc_norm = "الوصف غير متوفر، قيّم فقط مدى وضوح المعنى وتماسك الموضوع في هذا البيت."

        if not verse:
            # Empty answer → force 0
            user_content = (
                "لا يوجد بيت مقترح، أرجِع الرقم 0 فقط بدون أي كلام إضافي."
            )
        else:
            user_content = (
                "الوصف المطلوب:\n"
                f"{desc_norm}\n\n"
                "البيت المقترح:\n"
                f"{verse}\n\n"
                "المطلوب:\n"
                "- قيّم مدى انسجام هذا البيت مع الوصف من حيث المعنى والموضوع والجو الشعوري.\n"
                "- أعطِ رقمًا واحدًا حقيقيًا بين 0 و 10 فقط، يمكن أن يكون عددًا كسريًا مثل 7.5 أو 9.0.\n"
                "- 0 يعني أن البيت لا علاقة له تقريبًا بالوصف.\n"
                "- 10 يعني انسجامًا عاليًا جدًا مع الوصف.\n\n"
                "اكتب الرقم فقط بدون أي كلمات أو شرح أو رموز أخرى."
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

    # Use vLLM in batch
    outputs = llm.generate(prompts_text, sampling, use_tqdm=False)

    for i, out in enumerate(outputs):
        if out.outputs:
            raw_text = (out.outputs[0].text or "").strip()
        else:
            raw_text = ""
        rewards[i] = _parse_score_0_to_10(raw_text)

    return rewards
