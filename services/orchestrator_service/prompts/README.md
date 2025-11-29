This folder is reserved for orchestrator prompt scaffolding (router/planner/summarizer).
The current implementation does not load these files yet, but the expected layout
matches the design doc:

- 00_global_system.md
- 01_router_route_select.md
- 02_router_clarify_question.md
- 10_poem_generate_toolplan.md
- 11_bayt_fix_toolplan.md
- 11b_bayt_score_toolplan.md
- 12_library_search_toolplan.md
- 13_bayt_explain_toolplan.md
- 20_enhancer_feedback_summary.md
- 21_rag_query_rewrite.md

Populate these with the prompts from orchestrator_service.md when adding
router/planner logic that consumes them.
