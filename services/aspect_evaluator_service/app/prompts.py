from __future__ import annotations

ASPECTS = ("meaning", "cohesion", "fluency", "poeticness")

ASPECT_TITLES = {
    "meaning": "جودة المعنى",
    "cohesion": "الترابط",
    "fluency": "الفصاحة",
    "poeticness": "الشعرية",
}

JUDGE_SYSTEM_PROMPTS = {
    "meaning": (
        "أنت ناقد متخصص في تقييم مدى توافق البيت الشعري مع الوصف المطلوب وعمق الفكرة. "
        "أعطِ درجة بين 0 و1 (0 سيئ جدًا، 1 ممتاز) وقدم ملاحظة مختصرة تشرح السبب."
    ),
    "cohesion": (
        "أنت ناقد يقيّم مدى ترابط البيت مع الأبيات السابقة والوصف العام. ركّز على الانسجام أو الانقطاع، "
        "ثم أعطِ درجة 0..1 وتعليقًا قصيرًا يوضح السبب."
    ),
    "fluency": (
        "أنت ناقد لغوي يهتم بالسلامة النحوية والصوتية وسلاسة الجملة. أعطِ درجة 0..1 "
        "واذكر بإيجاز أهم الملاحظات اللغوية."
    ),
    "poeticness": (
        "أنت ناقد شعري يركز على الصور والخيال والإيقاع المعنوي. أعطِ درجة 0..1 مع تعليق قصير يشرح"
        "ما يدعم أو يضعف الشعرية."
    ),
}

JUDGE_RESPONSE_SCHEMA = {
    "name": "AspectJudgeResponse",
    "schema": {
        "type": "object",
        "properties": {
            "score_0_1": {"type": "number", "minimum": 0, "maximum": 1},
            "notes": {"type": "string"},
        },
        "required": ["score_0_1", "notes"],
        "additionalProperties": False,
    },
}
