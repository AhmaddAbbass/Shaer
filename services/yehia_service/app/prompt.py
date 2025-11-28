from __future__ import annotations

import json
from typing import List

from services.common_schemas.schemas import LLMMessage, PoemSpec, RagSearchHit, YehiaFeedback


SPEC_SYSTEM_PROMPT = (
    "أنت يحيى، شاعر وناقد عربي. استخرج مواصفات قصيدة جديدة من طلب المستخدم "
    "وأعدها في كائن JSON بوضوح دون شرح إضافي. الحقول المطلوبة: "
    "{poem_meter, poem_description, num_verses, poem_theme?, poem_era?, poet_name?, poem_title?}. "
    "احرص أن تكون poem_meter نصاً عربياً واضحاً، poem_description جملة/فقرة قصيرة، "
    "num_verses > 0."
)

FEEDBACK_SYSTEM_PROMPT = (
    "أنت يحيى، ناقد شعري. قيم البيت مقابل المواصفات وأعد كائناً JSON فقط "
    "بالشكل {ok: bool, score: int?, feedback: str}. لا تضف شرحاً خارج JSON."
)


def build_spec_messages(user_query: str, rag_hits: List[RagSearchHit]) -> List[LLMMessage]:
    rag_block = ""
    if rag_hits:
        parts = []
        for hit in rag_hits[:5]:
            parts.append(
                f"- عنوان غير معروف | شاعر: {hit.poet_name or 'غير محدد'} | بحر: {hit.poem_meter or 'غير محدد'} | وصف: {hit.poem_description}"
            )
        rag_block = "أمثلة مسترجعة:\n" + "\n".join(parts)

    user_content = f"طلب المستخدم:\n{user_query}\n\n{rag_block}\n\nأعد كائناً JSON فقط."
    return [
        LLMMessage(role="system", content=SPEC_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


def build_feedback_messages(verse_text: str, spec: PoemSpec) -> List[LLMMessage]:
    spec_json = json.dumps(spec.model_dump(), ensure_ascii=False, indent=2)
    user_content = (
        f"المواصفات:\n{spec_json}\n\nالبيت:\n{verse_text}\n\n"
        "أعد كائناً JSON فقط بالشكل {ok, score, feedback}."
    )
    return [
        LLMMessage(role="system", content=FEEDBACK_SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_content),
    ]


def build_chat_messages(messages: List[LLMMessage]) -> List[LLMMessage]:
    return messages


__all__ = [
    "build_spec_messages",
    "build_feedback_messages",
    "build_chat_messages",
    "SPEC_SYSTEM_PROMPT",
    "FEEDBACK_SYSTEM_PROMPT",
]
