# judge_server.py
import os
import re
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn

# --------------------------------------------------
# GPU pinning: this *process* uses GPU 1 only
# --------------------------------------------------
# You can ALSO launch with: CUDA_VISIBLE_DEVICES=1 python judge_server.py
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# vLLM will see this as device 0 (its own world)
os.environ.setdefault("VLLM_WORKER_CUDA_DEVICES", "0")

MODEL_ID = os.environ.get("YEHIA_VLLM_MODEL_PATH", "Navid-AI/Yehia-7B-preview")
GPU_UTIL = float(os.environ.get("YEHIA_VLLM_GPU_UTIL", "0.6"))
MAX_MODEL_LEN = int(os.environ.get("YEHIA_VLLM_MAX_MODEL_LEN", "1024"))

app = FastAPI()


class ScoreItem(BaseModel):
    verse: str
    description: str


class ScoreBatchRequest(BaseModel):
    items: List[ScoreItem]


class ScoreBatchResponse(BaseModel):
    scores: List[float]


_llm = None
_tokenizer = None
_sampling = None


def _normalize_arabic(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.replace("ـ", "")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _parse_score_0_to_10(text: str) -> float:
    """
    Extract a float in [0, 10] from judge output.
    If parsing fails, return 0.0.
    """
    if not text:
        return 0.0

    candidates = re.findall(r"(\d+(?:\.\d+)?)", text)
    for c in candidates:
        try:
            val = float(c)
        except Exception:
            continue

        if 0.0 <= val <= 10.0:
            return val

    return 0.0


def _get_llm():
    global _llm, _tokenizer, _sampling
    if _llm is not None:
        return _llm, _tokenizer, _sampling

    def _init_llm(gu, mlen):
        print(f"[judge] Initializing vLLM LLM from: {MODEL_ID}")
        print(f"[judge] vLLM GPU utilization: {gu} | max_model_len: {mlen}")
        return LLM(
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            trust_remote_code=True,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=gu,
            max_model_len=mlen,
        )

    try:
        _llm = _init_llm(GPU_UTIL, MAX_MODEL_LEN)
    except ValueError as e:
        msg = str(e)
        if "No available memory for the cache blocks" not in msg:
            raise
        fallback_gu = min(0.95, GPU_UTIL + 0.2)
        fallback_len = max(1024, MAX_MODEL_LEN // 2)
        print(
            f"[judge] Retry vLLM init with higher gpu_memory_utilization={fallback_gu} "
            f"and max_model_len={fallback_len} due to: {e}"
        )
        _llm = _init_llm(fallback_gu, fallback_len)

    _tokenizer = _llm.get_tokenizer()

    _sampling = SamplingParams(
        max_tokens=8,             # enough for "9.3" or "10"
        temperature=0.0,          # deterministic judge
        top_p=1.0,
        truncate_prompt_tokens=512,
    )

    print("[judge] Model + tokenizer ready.")
    return _llm, _tokenizer, _sampling


_SYSTEM_PROMPT = (
    "أنت ناقد شعري عربي متخصّص في تقييم مدى التزام الأبيات بالمضمون المطلوب. "
    "مهمّتك أن تعطي رقمًا واحدًا بين 0 و 10 يعبّر عن مدى انسجام البيت مع الوصف من حيث المعنى والموضوع والجو الشعوري."
)


def _build_user_content(desc_norm: str, verse: str) -> str:
    if not desc_norm:
        desc_norm = (
            "الوصف غير متوفر، قيّم فقط مدى وضوح المعنى وتماسك الموضوع في هذا البيت."
        )

    if not verse:
        return "لا يوجد بيت مقترح، أرجِع الرقم 0 فقط بدون أي كلام إضافي."

    return (
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


@app.post("/score_batch", response_model=ScoreBatchResponse)
def score_batch(req: ScoreBatchRequest):
    llm, tokenizer, sampling = _get_llm()

    prompts = []
    for item in req.items:
        verse_norm = _normalize_arabic(item.verse)
        desc_norm = _normalize_arabic(item.description)

        user_content = _build_user_content(desc_norm, verse_norm)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling, use_tqdm=False)

    scores: List[float] = []
    for out in outputs:
        if out.outputs:
            raw_text = (out.outputs[0].text or "").strip()
        else:
            raw_text = ""
        score = _parse_score_0_to_10(raw_text)
        # clip to [0, 10] for safety
        score = max(0.0, min(10.0, score))
        scores.append(score)

    return ScoreBatchResponse(scores=scores)


if __name__ == "__main__":
    # Expose on 0.0.0.0 so notebook container / pod can reach it
    uvicorn.run(app, host="0.0.0.0", port=8009)
