Here’s a full agents.md you can drop into the repo and hand to your teammate.
It assumes all 4 services are already working (yehia_service, shaer_service, rag_service, meter_service) and focuses only on the agent logic that sits on top.

Shaer Agents Design

This document describes how the agent layer on the backend behaves.

The frontend only hits one endpoint: POST /api/chat.

Internally, /api/chat is handled by an Orchestrator Agent.

The Orchestrator delegates to a small set of specialized agents, which in turn call the microservices:

yehia_service – general Arabic LLM.

shaer_service – Shaer-7B poetry generator (bayt-by-bayt).

rag_service – Neo4j + Chroma poetry RAG.

meter_service – bayt-level scansion (meter classifier).

All agents exchange typed objects defined in services/common_schemas/schemas.py (PoemSpec, PoemVersion, AgentStep, LibraryItem, MeterEvalResult, etc.).

1. High-level Agents

At runtime we use four main agents:

OrchestratorAgent
Entry point for /api/chat. Detects intent, chooses the right workflow, and assembles the final ChatResponse.

PoetryAgent
Handles everything that creates or edits poetry: building specs, generating verses, evaluating quality, and optionally improving poems via retries.

LibraryAgent
Wraps rag_service to search and fetch existing poems from the Ashaar corpus.

ExplainAgent
Handles theoretical / explanatory questions (“ما هو بحر الكامل؟”, “اشرح هذا البيت…”).

Inside PoetryAgent we still conceptually keep the sub-roles from the earlier plan:

Spec Builder (builds PoemSpec using Yehia + RAG).

Generation (calls Shaer bayt-by-bayt with the correct training prompt).

Critic / Teacher (meter + semantic/style evaluation).

Enhancer (optional loop to regenerate bad verses).

They live as methods inside PoetryAgent rather than separate top-level agents.

2. Shared Schemas (mental model)

These are not redefined here, just referred to:

ChatRequest

messages: ChatMessage[]

mode?: "generate" | "fix" | "search" | "explain" | "other"

ChatResponse

reply: string (required)

mode?: ...

poem_spec?: PoemSpec | null

poem_version?: PoemVersion | null

agent_trace?: AgentStep[]

library_context?: LibraryItem[]

warnings?: string[]

error_code?: string | null

PoemSpec
Target “design” of the poem (meter, theme, era, poet_name, description, num_verses, etc.).

PoemVersion
Actual verses plus the spec used; may also contain original_verses (for fix mode) and notes.

AgentStep
One entry in the right-sidebar timeline:

step: number

agent: string (e.g. "Orchestrator", "Poetry", "Library")

tool: string (e.g. "parse_intent", "build_spec", "generate_poem")

summary: string (Arabic text)

LibraryItem
Small card for a poem retrieved from RAG (poem_id, title, poet_name, poem_meter, poem_era, poem_theme).

MeterEvalResult (bayt-level)

text: string (the bayt that was evaluated)

expected_meter?: string | null

predicted_meter: string

score: number (0–1 confidence)

on_meter: boolean

notes?: string (human-readable hints)

PoetryAgent internally groups meter + Yehia feedback into a high-level EvalScores / Feedback object for its own use.

3. OrchestratorAgent
3.1 Responsibility

The Orchestrator is the only agent directly tied to /api/chat.

For each request it must:

Parse the stateless full chat history + mode hint.

Detect the user’s intent for the current turn.

Route to the appropriate flow:

generate → PoetryAgent (generate new poem)

fix → PoetryAgent (fix poem)

search → LibraryAgent

explain → ExplainAgent

no clear hint → try to infer, fall back to explain / general chat.

Maintain a global step counter for agent_trace and collect steps from sub-agents.

Build a final ChatResponse matching the frontend spec.

3.2 Inputs

ChatRequest:

messages including all previous context.

optional mode hint.

3.3 Outputs

ChatResponse with:

reply (user-facing answer),

mode (resolved),

optional poem_spec,

optional poem_version,

agent_trace (merged from all agents),

library_context (if RAG used).

3.4 Typical flows

Generate:

Step 1: log an AgentStep for parse_intent.

Step 2: call PoetryAgent.generate_poem_flow(...).

Merge:

poem_spec,

poem_version,

agent_steps returned by PoetryAgent,

optionally library_context (from RAG used during spec building).

Fix:

Same pattern, but call PoetryAgent.fix_poem_flow(...).

Search:

Step 1: parse_intent.

Step 2: LibraryAgent.search_flow(...).

Response:

reply summarising retrieved poems,

library_context filled,

no poem_version unless one poem is highlighted.

Explain:

Step 1: parse_intent.

Step 2: ExplainAgent.explain_flow(...).

Usually only reply + agent_trace.

4. PoetryAgent

This is the heavy one. It owns all workflows that involve generating or modifying poems.

Internally it has four phases:

Spec Builder – build_spec(...)

Generation – compose_poem(...) / compose_bayt(...)

Critic / Teacher – evaluate_bayt(...), evaluate_poem(...)

Enhancer – improve_poem_until_threshold(...)

PoetryAgent does not talk to the frontend. It only returns structured objects to the Orchestrator.

4.1 Services it calls

rag_service

search for similar poems and gather metadata.

yehia_service

build a clean poem_description and infer missing attributes (meter, era, theme, poet_name, num_verses).

optionally critique verses / poems for semantics and style.

shaer_service

generate one bayt at a time using the frozen Shaer training prompt skeleton.

meter_service

evaluate a bayt’s meter and provide MeterEvalResult.

4.2 generate_poem_flow(...)

Inputs:

User’s latest message (string).

Optional mode hint, but Orchestrator already decided this is a generate task.

Global step_counter + agent_trace list to append to.

Outputs:

PoemSpec

PoemVersion (final verses)

Optional library_context (poems used as exemplars)

A list of AgentStep entries describing all internal actions.

Internal phases:

Build Spec

Log AgentStep(step++, agent="Poetry", tool="build_spec", ...).

Use RAG to get top-k relevant poems:

call rag_service.search(text=query, top_k=k).

Send user_query + retrieved_descriptions + retrieved_metadata to yehia_service with a special “spec building” prompt.

Yehia returns:

refined Ashaar-style poem_description,

inferred poem_meter, poem_theme, poem_era, poet_name (optional),

suggested num_verses,

optional poem_title.

PoetryAgent assembles a final PoemSpec, deciding missing values (e.g. choose a meter from the retrieved set if Yehia left it unspecified).

Compose Poem

Log AgentStep(step++, agent="Poetry", tool="compose_poem", ...).

Initialise PoemVersion:

spec = PoemSpec,

verses = [].

Loop i = 1 .. num_verses:

Build the canonical Shaer prompt:

system message fixed (“أنت شاعر عربي متمكن…”).

user message: bullet list with:

البحر, الوصف العام, العصر, الشاعر, عدد الأبيات, ترتيب البيت المطلوب…

block of previous verses (or “لا توجد أبيات سابقة”).

“إرشادات مهمة” with standard bullets.

Important constraint: only values change, structure stays identical to SFT.

Call shaer_service.generate_bayt(spec, previous_verses, sequence_number).

Append bayt to PoemVersion.verses and to previous_verses.

Evaluate Poem

Log AgentStep(step++, agent="Poetry", tool="evaluate_poem", ...).

For each bayt:

Call meter_service.eval_bayt(text=bayt, expected_meter=spec.poem_meter) → MeterEvalResult.

Optionally call yehia_service.feedback_bayt(bayt, spec) → textual feedback + simple score.

Aggregate:

meter accuracy across verses,

optional semantic/style scores from Yehia,

produce a high-level EvalScores:

meter_ok, overall_score, flags, etc.

Enhance (optional)

If EvalScores are good enough:

Skip this phase.

Else:

Log AgentStep(step++, agent="Poetry", tool="enhance_poem", ...).

Decide which bayts to regenerate:

e.g. those with on_meter == False or very low semantic score.

For each problematic bayt:

Add a tiny extra bullet under “إرشادات مهمة” (e.g. “حاول تحسين الوزن في هذا البيت مع الحفاظ على نفس المعنى.”).

Call Shaer again for that bayt with the same template.

Optionally re-run evaluate_poem and stop after small number of enhancement rounds (e.g. max 2).

PoetryAgent then returns the final PoemSpec, PoemVersion, and all steps.

4.3 fix_poem_flow(...)

This is similar but starts from user-provided verses instead of an empty poem.

Inputs:

Raw poem text from the user (maybe multiple bayts).

Optional hints (desired meter, theme…).

step counter + trace list.

Phases:

Infer Spec

Build a PoemSpec either by:

asking Yehia to describe the poem and infer meter/theme/era, or

combining user hints with classifier outputs (optional).

Log a build_spec step (mention that this is “fix” mode).

Evaluate Existing Verses

For each verse:

call meter_service.eval_bayt,

optionally yehia_service.feedback_bayt.

Identify which bayts are “broken” (bad meter) or weak (semantic issues).

Log evaluate_poem step.

Regenerate bad verses

For each problematic bayt i:

Build Shaer prompt with:

same PoemSpec,

previous_verses including neighbouring context,

a small instruction like “أعد صياغة هذا البيت مع الحفاظ على نفس المعنى وتحسين الوزن.”

Call shaer_service.generate_bayt to propose a new verse.

Assemble corrected verses into a new PoemVersion:

verses = final corrected poem.

original_verses = the user’s original poem.

Optionally re-run evaluation.

Return

PoetryAgent returns:

PoemSpec,

PoemVersion (with original_verses and verses),

list of AgentSteps.

The Orchestrator wraps this into a ChatResponse with mode = "fix".

5. LibraryAgent

LibraryAgent is a thin wrapper around rag_service.

5.1 Responsibilities

Serve all “search / recommend / retrieve” use cases:

“هات قصيدة عن الصبر في العصر العباسي”

“أريد قصيدة حماسية للمتنبي”

“أعطني قصائد مشابهة لهذه القصيدة…”

Provide both:

library_context for the Inspector (cards),

text snippets for reply.

5.2 Services it calls

rag_service.search(text, top_k) – Chroma semantic search.

rag_service.filter(poet, meter, era, theme, limit) – Neo4j filter queries.

rag_service.get_poem(poem_id) – full poem text + metadata.

5.3 Main flows

search_flow(query, filters)

Use sematic search and/or filters to get relevant poems.

Map to:

LibraryItem[] for library_context,

textual summary for reply (e.g. list of titles + poets + first bayt).

Add one or more AgentSteps, e.g.:

agent = "Library", tool = "search_poems".

similar_poems_flow(poem_id) (optional)

Use Chroma + graph neighbors to find similar poems.

Same output pattern: LibraryItem[] + summary.

PoetryAgent may also call LibraryAgent internally during build_spec (to fetch exemplars). In that case, the library_context can be reused in the final response.

6. ExplainAgent

ExplainAgent is used for non-generative questions or when the user just wants understanding.

6.1 Typical queries

“ما هو بحر الطويل؟”

“اشرح لي هذا البيت: …”

“ما معنى القافية؟”

6.2 Services it calls

Primarily yehia_service:

generic /chat endpoint with a system prompt like:

“أنت مدرس عروض وشعر عربي، تشرح المفاهيم بأسلوب واضح ومبسط وبأمثلة من الشعر.”

Optionally rag_service:

if the question mentions a famous poem or verse that exists in the corpus, it can fetch context.

6.3 Outputs

Mostly:

reply = explanation text (bullets, examples…).

mode = "explain" or "other".

No need for poem_spec / poem_version unless we explicitly analyze a particular poem and want to attach it.

agent_trace usually has 1–2 steps:

parse_intent from Orchestrator.

Explain agent step like tool="explain_meter" or tool="explain_bayt".

7. Agent Trace Conventions

All agents cooperate to produce a single agent_trace: AgentStep[] that drives the Inspector timeline in the frontend.

Rules:

Global step counter

Orchestrator maintains current_step (integer).

Whenever an agent performs a semantically meaningful action, it:

increments current_step,

appends an AgentStep.

Agent names

"Orchestrator" – intent parsing, high-level routing, final assembly.

"Poetry" – any step related to poem spec, generation, evaluation, enhancement.

"Library" – RAG search / retrieval.

"Explain" – theoretical explanations.

Tool names

Examples (non-exhaustive):

parse_intent

build_spec

compose_poem

compose_bayt

evaluate_poem

enhance_poem

search_poems

get_poem

explain_meter

explain_bayt

Summary text

Always short, human-readable Arabic:

e.g.
"تم استنتاج مواصفات القصيدة (بحر الكامل، 8 أبيات) اعتمادًا على وصف المستخدم والقصائد المسترجعة."
"تم توليد 4 أبيات باستخدام نموذج Shaer-7B ومحاولة الحفاظ على جو الحنين."
"تم استرجاع 5 قصائد عن الصبر من العصر العباسي عبر RAG."

The frontend just displays these entries in order.

8. How everything fits under /api/chat

Putting it all together:

Frontend sends { messages, mode? } to POST /api/chat.

OrchestratorAgent:

builds a ChatRequest,

determines effective mode,

initializes agent_trace = [], step = 1.

Depending on mode:

generate: call PoetryAgent.generate_poem_flow.

fix: call PoetryAgent.fix_poem_flow.

search: call LibraryAgent.search_flow.

explain/other: call ExplainAgent.explain_flow.

Each called agent:

uses working microservices (yehia_service, shaer_service, rag_service, meter_service),

appends its own AgentSteps,

returns structured objects (PoemSpec, PoemVersion, LibraryItem[]…).

Orchestrator assembles the final ChatResponse:

reply (the main Arabic answer),

mode,

poem_spec / poem_version (for generate/fix modes),

agent_trace (full timeline),

library_context (if RAG used).

Response is sent back to the frontend, which:

shows reply in chat,

shows poem and metadata in the studio,

shows agent_trace and library_context in the right Inspector.

9. End-to-end I/O contract (single backend input)

The backend exposes only one externally visible input: `POST /api/chat`. All functionality passes through the Orchestrator, which uses ChatGPT (GPT‑4o or GPT‑4.1) as the intent parser / planner. The Orchestrator never exposes service-specific APIs. Instead, it decides which internal “tools” (agents or microservices) to invoke:

- `generate_poem` → PoetryAgent + Shaer + meter
- `fix_poem` → PoetryAgent (fix flow) + meter
- `search_poems` → LibraryAgent (RAG search)
- `explain` → ExplainAgent

ChatGPT orchestrator logic:

1. Receive the entire chat history and optional `mode` hint from the frontend.
2. Run an intent-classification prompt inside the Orchestrator to pick one of {generate, fix, search, explain}.
3. Emit an `AgentStep` (`tool="parse_intent"`) and call the matching agent method.

10. Detailed tool choreography per mode

10.1 Generate poem workflow

1. **Collect user spec**  
   - Extract raw desire: meter, theme, poet, tone, verse count.  
   - Anything missing is tagged as `None`.
2. **RAG assist**  
   - Call `rag_service.search` for 3–5 similar descriptions using the user text.  
   - Include the matched descriptions + metadata in an `AgentStep` summary.  
   - These snippets are fed to Yehia to enrich the spec.
3. **Fill blanks via Yehia**  
   - Prompt template instructs Yehia to propose: refined description, explicit meter, theme, era, poet voice, verse count, rhyming hints.  
   - If Yehia still omits meter (current bug), fall back to defaults (e.g. بحر الكامل, 6 أبيات) and log a warning in `agent_trace`.
4. **Build Shaer prompt**  
   - Canonical skeleton: system message + user instructions (meter, description, poet, verse index, previous verses block, “إرشادات مهمة”).  
   - `PoemSpec` saved for the response.
5. **Bayt-by-bayt generation loop**  
   - For each verse index: call `shaer_service.generate_bayt`.  
   - Append to PoemVersion and context.  
   - Emit `AgentStep(tool="compose_bayt")`.
6. **Feedback stage**  
   - For every generated bayt: call `meter_service.eval_bayt`. If meter service fails, mark `on_meter=false`, set `predicted_meter="default"`, and continue (meter defaults).  
   - Optional semantic/style feedback via Yehia.  
   - Aggregate into EvalScores.
7. **Enhancement loop (≤3 tries)**  
   - If a bayt score < threshold, add instruction “حسّن الوزن مع الحفاظ على المعنى” and regenerate only that bayt.  
   - Re-evaluate meter after each regeneration.  
   - Break after 3 failed attempts; flag the verse in `PoemVersion.notes`.
8. **Finalize**  
   - Return PoemSpec, PoemVersion, agent trace, and any RAG exemplars (library_context).  
   - Orchestrator sends a single response to the frontend.

10.2 Fix poem workflow

1. Orchestrator intent → PoetryAgent.fix.  
2. Parse user poem into verses and optional spec hints.  
3. Build PoemSpec: use user hints + Yehia inference + RAG (if needed).  
4. Evaluate all verses (meter + semantics).  
5. For verses failing evaluation, call Shaer with instruction “أعد صياغة البيت رقم X محافظًا على الوزن والمعنى”.  
6. Limit regeneration per bayt to 3 attempts (same enhancer loop).  
7. Meter fallback if service unavailable: mark `predicted_meter="unknown"` and move on.  
8. Return corrected PoemVersion with `original_verses`.

10.3 Search / library workflow

1. Orchestrator intent → LibraryAgent.search_flow.  
2. LibraryAgent queries RAG:  
   - If user specified filters (meter/theme/era), call graph filters first; otherwise semantic search.  
   - Fetch each poem’s title, poet, era, meter, and first bayt.  
3. Format reply:  
   - Sort ascending by era (abc logic), display “العصر – الشاعر – عنوان القصيدة – مطلعها”.  
   - Provide `library_context` (top K items).  
4. If no results, use fallback message and include a warning step.

10.4 Explain workflow

1. Orchestrator intent → ExplainAgent.explain_flow.  
2. Prompt Yehia (teacher persona).  
3. Optionally augment with RAG excerpt if user referenced a known poem.  
4. Return reply + agent trace (no poem spec).

11. Logging & agent trace

- Every tool call adds an `AgentStep` with `agent`, `tool`, `summary`.  
- Step numbers increment globally.  
- Failures are logged as steps with `summary` describing the fallback (e.g. “تعذر تشغيل خدمة البحر، تم استخدام بحر الكامل كقيمة افتراضية”).  
- The frontend uses `agent_trace` to show the execution timeline.

12. Default / fallback values

- Meter: default to بحر الكامل when meter_service or Yehia fails, but keep a flag `spec.meter_is_default=true`.  
- Theme: default “شعر وجداني” if not inferred.  
- Verse count: default 6.  
- When any fallback is used, append a warning string to `ChatResponse.warnings` so UI can surface it.

13. Tooling summary

| Intent     | Tools (in order)                                                                 | Notes                                                           |
|-----------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------|
| generate  | RAG search → Yehia spec builder → Shaer generator → Meter eval → Enhancer loop    | Enhancer stops after 3 tries per bayt.                          |
| fix       | Yehia + RAG spec inference → Meter eval → Shaer regenerate (per bayt)             | Skip initial Shaer generation; focus on bad verses.             |
| search    | RAG semantic search / Neo4j filters → format results                              | Reply includes sorted poems with titles and poets.              |
| explain   | Yehia teacher persona (+ optional RAG context)                                    | No Shaer/meter usage.                                          |

This expanded section ensures a single input (chat) powers all capabilities, clarifies tool ordering, introduces the fallback defaults, and codifies the agent trace logging so the UI can visualize which tools were triggered during each request.
