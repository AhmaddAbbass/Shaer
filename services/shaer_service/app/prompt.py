from __future__ import annotations

from typing import Iterable, List

from .schemas import BaytGenerationRequest

SYSTEM_PROMPT = (
    "أنت شاعر عربي متمكن من مختلف البحور والأغراض الشعرية، وقادر على محاكاة أساليب "
    "الشعراء عبر العصور مع الحفاظ على سلامة اللغة، والوزن، والقافية."
)

DEFAULT_ERA = "أسلوب عربي فصيح مناسب لأي عصر ما لم يُذكر غير ذلك."
DEFAULT_POET = "أسلوب عربي عام منسجم مع موضوع القصيدة."


def _format_previous_verses(previous_verses: Iterable[str]) -> str:
    verses = [verse.strip() for verse in previous_verses if verse.strip()]
    if not verses:
        return "لا توجد أبيات سابقة في القصيدة؛ اكتب أول بيت بانسجام مع المواصفات."
    numbered = [f"{idx}. {verse}" for idx, verse in enumerate(verses, start=1)]
    return "\n".join(numbered)


def build_messages(request: BaytGenerationRequest) -> List[dict[str, str]]:
    """Build the Shaer training-style messages array."""
    era = request.poem_era.strip() if request.poem_era and request.poem_era.strip() else DEFAULT_ERA
    poet = request.poet_name.strip() if request.poet_name and request.poet_name.strip() else DEFAULT_POET
    previous_block = _format_previous_verses(request.previous_verses)
    guidance_lines = [
        "- أخرج بيتًا واحدًا مكوّنًا من صدر وعجز في سطر واحد.",
        "- التزم بالبحر الشعري، وبجوّ ومعنى القصيدة كما في الوصف والأبيات السابقة (إن وُجدت).",
        "- لا تكرّر أي بيت سابق ولا تضف شروحًا أو عناوين أو علامات خاصة؛ الناتج هو البيت فقط.",
    ]
    extra_tips = []
    for tip in request.extra_guidance:
        clean_tip = tip.strip()
        if clean_tip:
            extra_tips.append(f"- {clean_tip}")
    if extra_tips:
        guidance_lines.extend(extra_tips)

    user_content = (
        "المطلوب منك في هذه المهمة أن تولّد بيتًا شعريًا واحدًا فقط، وفق المواصفات التالية:\n\n"
        f"- البحر: {request.poem_meter}\n"
        f"- الوصف العام لموضوع القصيدة: {request.poem_description}\n"
        f"- العصر: {era}\n"
        f"- الشاعر: {poet}\n"
        f"- عدد أبيات القصيدة الكلي: {request.num_verses}\n"
        f"- ترتيب البيت المطلوب داخل القصيدة: {request.sequence_number}\n\n"
        "الأبيات السابقة في القصيدة:\n"
        f"{previous_block}\n\n"
        "إرشادات مهمة:\n"
        + "\n".join(guidance_lines)
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


__all__ = ["build_messages", "SYSTEM_PROMPT"]
