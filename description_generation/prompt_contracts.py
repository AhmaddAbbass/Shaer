from __future__ import annotations

from typing import Iterable


DESCRIPTION_PROMPT_VERSION = "pure_description_v8_poem_only_antidrift"
DESCRIPTION_SYSTEM_PROMPT = """أنت تكتب وصفًا عربيًا لقصيدة.

المطلوب:
اكتب وصفًا عربيًا واحدًا يكون وصفًا فقط، لا طلبًا ولا أمرًا ولا قائمة تعليمات.

الهدف:
نريد وصفًا يلتقط أفكار القصيدة ومعناها وصورها ونبرتها العامة، من غير نسخ، ومن غير اختراع معانٍ غير موجودة.

أخرج النتيجة في JSON فقط بالشكل:
{"new_description":"..."}

القواعد:
- ابدأ الوصف بـ "القصيدة تتحدث عن..."
- حافظ على المعنى الموجود في الأبيات فقط.
- اذكر الفكرة الأساسية وبعض الصور أو العناصر الملموسة المهمة إذا كانت مفيدة.
- قد تحتاج أحيانًا إلى قدر يسير من الفهم أو التحليل لتكتب وصفًا جيدًا، لكن لا تتوسع في ذلك أكثر مما يحتمله النص.
- إذا كان النص يحتمل أكثر من فهم، فالتزم بالمعنى الأقرب إلى ظاهر الأبيات ولا تحسم تأويلًا زائدًا.
- لا تحوّل الصور أو الألفاظ المتفرقة إلى قصة كاملة أو مشهد متماسك ما لم يكن ذلك ظاهرًا في النص.
- لا تستنتج نية نفسية أو حكمًا أخلاقيًا أو خلفية تاريخية إلا إذا كانت ظاهرة بوضوح في الأبيات.
- إذا كان النص قصيرًا أو يقوم على صورة واحدة أو موقف واحد، فلا تبنِ عليه معنى أكبر من حجمه.
- إذا كان النص غريبًا أو خشنًا أو شديد الخصوصية، فاحتفظ بهذه الخصوصية ولا تهذبه إلى معنى عام مألوف.
- إذا كان النص غامضًا أو يحتمل أكثر من قراءة، فالتزم بالمعنى الأقرب إلى ظاهر الأبيات ولا تحسم تأويلًا زائدًا.
- لا تخترع مشاهد أو دوافع أو علاقات سببية غير ظاهرة.
- لا تجعل الوصف شرحًا بيتًا بيتًا.
- في النصوص القصيرة أو الغريبة، صف ما يظهر في النص أكثر مما تفسره.
- اجعل الوصف موجزًا نسبيًا، وخاصة في النصوص القصيرة، ويفضل غالبًا أن يكون جملة واحدة أو جملتين قصيرتين.
- لا تذكر البحر أو القافية أو عدد الأبيات.
- لا تستخدم صيغ الطلب أو الأمر مثل: أريد، أكتب، اذكر، ركز، اجعل، يجب أن.
- لا تستبدل الغريب أو الخاص في النص بتعبير عام باهت.
- اكتب فقرة واحدة طبيعية مكتملة.
"""


FINAL_SFT_PROMPT_VERSION = "final_sft_meter_emphasis_v2_num_lines"
FINAL_SFT_SYSTEM_PROMPT = """أنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.
التزم بالبحر المحدد في كل شطر، واستلهم من الموضوع دون نقله حرفياً.
أخرج الأبيات فقط دون مقدمة أو تعليق.
التزم التزاماً صارماً بالبحر المطلوب، ولا تخرج عنه.
"""

FINAL_SFT_USER_TEMPLATE = """البحر الأساسي: {base_meter}
الصيغة: {form}
عدد الأشطر المطلوب: {num_lines}
الموضوع: {description}

اكتب {num_lines} أشطارًا ملتزمة بصيغة {form} من بحر {base_meter} دون أي شرح إضافي."""


def normalize_verses(verses: Iterable[object]) -> list[str]:
    return [str(item).strip() for item in verses if str(item).strip()]


def build_description_user_prompt(verses: Iterable[object]) -> str:
    poem = "\n".join(normalize_verses(verses))
    return (
        "الأبيات:\n"
        f"{poem}\n\n"
        "اكتب وصفًا واحدًا يلتقط أفكار القصيدة ومعناها وصورها ونبرتها العامة."
    )


def build_meter_label(base_meter: str, form: str) -> str:
    if str(form).strip() == "تام":
        return str(base_meter).strip()
    return f"{str(form).strip()} {str(base_meter).strip()}".strip()


def build_sft_prompt(base_meter: str, form: str, description: str, num_lines: int) -> str:
    meter_label = build_meter_label(base_meter, form)
    user_prompt = FINAL_SFT_USER_TEMPLATE.format(
        base_meter=str(base_meter).strip(),
        form=str(form).strip(),
        meter_label=meter_label,
        description=str(description).strip(),
        num_lines=int(num_lines),
    )
    return f"<s> [INST] <<SYS>>\n{FINAL_SFT_SYSTEM_PROMPT}\n<</SYS>>\n\n{user_prompt.strip()} [/INST]"


def build_sft_completion(verses: Iterable[object]) -> str:
    return "\n".join(normalize_verses(verses)).strip()


def build_sft_full_text(prompt: str, completion: str) -> str:
    return f"{prompt} {completion} </s>"
