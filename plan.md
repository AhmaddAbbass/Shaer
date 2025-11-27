# Agentic Workflow Plan (Living Document)

Goal: given a user’s raw Arabic query, retrieve the most similar poem descriptions, use them to reshape the query in Yehia’s style, then extract/complete structured attributes (meter, era, poet, theme, etc.) to build the final instruction payload our generation model expects.

## Current Retrieval Step
- Function: `agent.retrieve_similar_descriptions(query, top_k)`
- Behavior: returns top-k poem descriptions from Chroma (IDs = `poem_id`) to serve as exemplars and metadata hints (meters, eras, poets, etc.).

## Planned Two-Call Yehia Flow
1) **Rewrite**: Prompt Yehia with (user query + retrieved descriptions) to rewrite the user description in the same style/level of detail as the retrieved items while preserving user intent and adding missing details.
2) **Attribute Extraction**: Prompt Yehia again with the rewritten description + retrieved metadata to extract:
   - meter, era, poet name, theme, description (enriched)
   - if any field is missing, choose plausible values from the retrieved set (e.g., pick a meter from the retrieved meters, era from retrieved eras).

## Target Instruction Format (what the generation model expects)
```
[
  {"role": "system", "content": "أنت شاعر عربي متمكن من مختلف البحور والأغراض الشعرية، وقادر على محاكاة أساليب الشعراء عبر العصور مع الحفاظ على سلامة اللغة، والوزن، والقافية."},
  {"role": "user", "content": "المطلوب منك ...\n- البحر: <meter>\n- الوصف العام لموضوع القصيدة: <description>\n- العصر: <era>\n- الشاعر: <poet>\n- عدد أبيات القصيدة الكلي: <total_verses>\n- ترتيب البيت المطلوب داخل القصيدة: <index>\n\nالأبيات السابقة في القصيدة:\n<previous lines or none>\n\nإرشادات مهمة:\n- ..."}
]
```
- Only `meter/theme/poet/era/description` vary; everything else is stable scaffolding.

## Next Steps
- Add Yehia prompt templates (rewrite + extract) using the retrieved metadata.
- Wire an orchestrator that calls: retrieve → rewrite → extract → emit final instruction JSON.
- Log provenance (retrieved poem_ids) alongside the generated instruction for traceability.
