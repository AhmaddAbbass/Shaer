
# Backend Requirements for Shaer-AI Frontend (Updated)

This document describes **exactly** what the Shaer-AI frontend expects from the backend.

The goal:
One **orchestrator endpoint** that:

* Receives the **full chat history** and an optional **mode hint**.
* Runs internal logic (orchestrator, RAG, Yehia, Shaer, scansion, etc.).
* Returns:

  * A **final reply** for the user,
  * Plus **structured metadata** so the UI can show:

    * Poem metadata (spec),
    * The actual verses,
    * Which **agents/tools** ran (timeline),
    * Which poems were retrieved via RAG.

This metadata is what powers the **right sidebar (Inspector)** on the Studio page.

---

## 1. API Overview

* **Single endpoint**: `POST /api/chat`
* Backend is **stateless**:
  Frontend sends the **entire message history** each time.
* No session tokens / auth in v1.

---

## 2. Endpoint & Transport

### URL

* Base URL: from backend config / `.env`.
  Frontend uses: `VITE_API_BASE_URL` (default `http://localhost:8000`).

* Full endpoint:

```text
POST {VITE_API_BASE_URL}/api/chat
e.g. POST http://localhost:8000/api/chat
```

### Headers

```http
Content-Type: application/json
```

---

## 3. Request Format

### Top-level body

```ts
{
  messages: ChatMessage[];
  mode?: string;
}
```

#### `ChatMessage` type

```ts
interface ChatMessage {
  id: string;              // unique per message (frontend uses timestamp-ish strings)
  role: 'user' | 'assistant';
  content: string;         // the message text (Arabic / English / mixed)
}
```

* **Messages are ordered** oldest → newest.
* Frontend sends **all previous turns**, including the assistant replies.

#### `mode` hint

```ts
type ModeHint = 'generate' | 'fix' | 'search' | 'explain';
```

* `mode` is **optional**.
* Orchestrator can:

  * ignore it,
  * treat it as a hint / prior,
  * or override it based on its own intent detection.

#### Example request (first turn)

```json
{
  "messages": [
    {
      "id": "1701234567890",
      "role": "user",
      "content": "اكتب لي قصيدة عن الحنين للوطن"
    }
  ],
  "mode": "generate"
}
```

#### Example request (second turn, history included)

```json
{
  "messages": [
    {
      "id": "1",
      "role": "user",
      "content": "اكتب قصيدة عن الربيع"
    },
    {
      "id": "2",
      "role": "assistant",
      "content": "بالطبع! إليك قصيدة عن الربيع..."
    },
    {
      "id": "3",
      "role": "user",
      "content": "اجعلها أطول"
    }
  ],
  "mode": "generate"
}
```

---

## 4. Response Format (High-Level)

The frontend expects this shape:

```ts
interface ChatResponse {
  reply: string;                    // REQUIRED – text shown in the chat
  mode?: 'generate' | 'fix' | 'search' | 'explain' | 'other';

  poem_spec?: PoemSpec | null;      // metadata about a poem
  poem_version?: PoemVersion | null; // structured verses

  agent_trace?: AgentStep[];        // CRITICAL – timeline of agents/tools
  library_context?: LibraryItem[];  // poems retrieved from RAG / library

  // optional extra fields for debugging / future
  warnings?: string[];
  error_code?: string | null;
}
```

Only `reply` is strictly required; everything else is optional but highly recommended.

---

## 5. Type Definitions

### 5.1 `PoemSpec` – poem metadata

Describes **how the orchestrator conceptualized the poem**.

```ts
interface PoemSpec {
  poem_title?: string;        // e.g., "الحنين إلى الوطن"
  poem_meter?: string;        // e.g., "الكامل", "البسيط", "الطويل"
  poem_theme?: string;        // e.g., "الحنين", "الغزل", "المدح"
  poem_era?: string;          // e.g., "عباسي", "جاهلي", "أندلسي", "حديث"
  poet_name?: string;         // e.g., "المتنبي", "نزار قباني", or "نمط المتنبي"
  poem_description?: string;  // prose description; can be ~1–3 sentences
  num_verses?: number;        // intended number of أبيات
}
```

**When to fill:**

* Any time the system is **generating a poem**, **fixing a poem**, or clearly **discussing a specific poem**.
* For purely explanatory answers (e.g., “ما هو بحر الطويل؟”) you can omit or return `null`.

---

### 5.2 `PoemVersion` – structured poem output

This wraps the actual verses plus the spec used.

```ts
interface PoemVersion {
  spec: PoemSpec;          // the spec actually used
  verses: string[];        // each entry = one بيت (صدر + عجز في سطر واحد)

  // optional extras (good for "fix" mode):
  original_verses?: string[]; // the user's original verses (if the task is to fix them)
  notes?: string;             // short note about changes / quality, if any
}
```

**Example:**

```json
{
  "spec": {
    "poem_meter": "الكامل",
    "poem_theme": "الحنين",
    "num_verses": 4,
    "poem_description": "قصيدة تعبر عن الشوق والحنين للوطن"
  },
  "verses": [
    "أحنُّ إلى بلادي والديار",
    "وتشتاقُ الفؤادَ بها الأزهار",
    "فكم ليلٍ بعيدٍ قد مضى",
    "وفي القلب اشتياقٌ واحترار"
  ]
}
```

**When to fill:**

* **generate** mode: verses are the newly generated poem.
* **fix** mode:

  * `verses` = corrected/final version.
  * `original_verses` (optional) = what the user originally provided, if you want to keep a structured record.

---

### 5.3 `AgentStep` – 🔥 tool & agent logging (Inspector)

This is **CRITICAL** for the right sidebar (Agent Steps tab).

It tells the frontend **what the orchestrator actually did**, step by step:

```ts
interface AgentStep {
  step: number;      // 1, 2, 3, ... (sequential)
  agent: string;     // e.g., "Orchestrator", "Library", "SpecBuilder", "Yehia", "Shaer", "Scansion"
  tool: string;      // e.g., "parse_intent", "rag.search", "build_spec", "yehia.rewrite", "shaer.generate_verses"
  summary: string;   // short human-readable description (Arabic is perfect)
}
```

**Minimum expectation:**

For each meaningful internal operation, emit one `AgentStep`. Examples:

* Intent detection
* RAG search
* Poem spec building
* Generation via Shaer
* Meter checking
* Description consistency check via Yehia
* Final response assembly

**Example array:**

```json
[
  {
    "step": 1,
    "agent": "Orchestrator",
    "tool": "parse_intent",
    "summary": "تم اكتشاف نية \"توليد قصيدة\" عن الحنين للوطن"
  },
  {
    "step": 2,
    "agent": "Library",
    "tool": "rag.search",
    "summary": "تم استرجاع 5 قصائد عن الحنين من العصر العباسي باستخدام Chroma/Neo4j"
  },
  {
    "step": 3,
    "agent": "SpecBuilder",
    "tool": "build_spec",
    "summary": "تم اختيار بحر الكامل و4 أبيات مع أسلوب حديث اعتماداً على القصائد المسترجعة"
  },
  {
    "step": 4,
    "agent": "Shaer",
    "tool": "generate_verses",
    "summary": "تم توليد 4 أبيات باستخدام نموذج Shaer-7B باتباع الـ PoemSpec"
  },
  {
    "step": 5,
    "agent": "Scansion",
    "tool": "check_meter",
    "summary": "تم التحقق من الوزن: الأبيات موزونة على بحر الكامل"
  }
]
```

**Frontend behavior:**

* Shows these in a **vertical timeline** with gold dots and lines.
* Displays `agent · tool` as title, `summary` as description.

If nothing is logged, pass `[]` or omit the field → sidebar will say “No internal steps logged.”

---

### 5.4 `LibraryItem` – RAG / library context

Represents **poems retrieved** from your RAG system (Neo4j + Chroma).

```ts
interface LibraryItem {
  poem_id: string;      // should map to HF dataset or Neo4j node ID, e.g. "ashaar:12345"
  title: string;        // poem title
  poet_name?: string;
  poem_meter?: string;
  poem_era?: string;
  poem_theme?: string;
}
```

* `poem_id` should be stable (e.g. `poem_id` in `ashaar-full-desc-preprocessed`).
* These identify the **evidence the orchestrator used**.

**Example:**

```json
[
  {
    "poem_id": "ashaar_12345",
    "title": "أراك عصي الدمع",
    "poet_name": "أبو فراس الحمداني",
    "poem_meter": "الطويل",
    "poem_era": "عباسي",
    "poem_theme": "الأسْر"
  },
  {
    "poem_id": "ashaar_67890",
    "title": "قف بالديار",
    "poet_name": "ابن الرومي",
    "poem_meter": "الطويل",
    "poem_era": "عباسي",
    "poem_theme": "الرثاء"
  }
]
```

**When to fill:**

* Any time you **call RAG** (Chroma, Neo4j) to retrieve similar poems, topics, etc.
* For pure “explain meter” answers that don’t use RAG, you can omit or send `[]`.

**Frontend behavior:**

* Displays a list of cards showing title, poet, and chips for meter/era/theme in the “Library” tab.

---

## 6. Behavior by Mode (Semantics)

These are not enforced by the frontend, but they describe **how we intend the backend to behave**.

### 6.1 `mode = "generate"`

User wants a new poem.

* Orchestrator should:

  * Parse user request (topic, mood, length).
  * Optionally call RAG to get similar poems.
  * Build a `PoemSpec`.
  * Call Shaer (and maybe Yehia scaffolding) to generate verses.
  * Run scansion & evaluator.
* Response:

  * `reply`: poem + short explanation if you like.
  * `mode`: `"generate"`.
  * `poem_spec`: filled.
  * `poem_version`: filled with verses.
  * `agent_trace`: full sequence (parse_intent, rag.search, build_spec, generate, check_meter, etc.).
  * `library_context`: list of retrieved poems (if any).

### 6.2 `mode = "fix"`

User gives a poem and asks to fix it.

* Orchestrator:

  * Detect that this is a “fix my poem” task.
  * Run scansion, error detection, maybe RAG (optional).
  * Fix verses using Shaer / repair pipeline.
* Response:

  * `reply`: explain issues & show corrected poem.
  * `mode`: `"fix"`.
  * `poem_spec`: meter/theme/etc. (if inferred).
  * `poem_version`:

    * `verses`: corrected poem.
    * optionally `original_verses`: original poem.
  * `agent_trace`: steps like `scansion.check_meter`, `yehia.explain_errors`, `shaer.rewrite_verses`.

### 6.3 `mode = "search"`

User wants to **retrieve** poems, not generate.

* Orchestrator:

  * Run RAG / graph queries.
  * Possibly ask Yehia to rank or summarize.
* Response:

  * `reply`: explanation + maybe a short snippet of each retrieved poem.
  * `mode`: `"search"`.
  * `library_context`: **MUST** be filled (this is the relevant data).
  * `poem_spec` / `poem_version`: optional; you can include for a highlighted poem if you want.
  * `agent_trace`: e.g. `parse_intent`, `rag.search`, `yehia.rank_results`.

### 6.4 `mode = "explain"` (or `"other"`)

User asks theory questions: what is بحر الكامل, what is qaafiyah, etc.

* Orchestrator:

  * Use Yehia or any LLM to answer.
  * RAG optional (if you want examples).
* Response:

  * `reply`: explanation.
  * `mode`: `"explain"` or `"other"`.
  * `poem_spec` / `poem_version`: likely omitted.
  * `library_context`: optional (if RAG used).
  * `agent_trace`: still useful (`yehia.explain_concept` etc.).

---

## 7. Complete Example Response (Generate)

**Request:**

```json
{
  "messages": [
    {
      "id": "1701234567890",
      "role": "user",
      "content": "اكتب لي قصيدة عن الحنين للوطن"
    }
  ],
  "mode": "generate"
}
```

**Response:**

```json
{
  "reply": "بالطبع! إليك قصيدة عن الحنين للوطن على بحر الكامل:\n\nأحنُّ إلى بلادي والديار\nوتشتاقُ الفؤادَ بها الأزهار\nفكم ليلٍ بعيدٍ قد مضى\nوفي القلب اشتياقٌ واحترار\n\nهذه القصيدة موزونة على بحر الكامل وتعبر عن الشوق العميق للوطن.",

  "mode": "generate",

  "poem_spec": {
    "poem_title": "الحنين إلى الوطن",
    "poem_meter": "الكامل",
    "poem_theme": "الحنين",
    "poem_era": "حديث",
    "poet_name": "نمط حديث عام",
    "num_verses": 4,
    "poem_description": "قصيدة تعبر عن الشوق والحنين للوطن"
  },

  "poem_version": {
    "spec": {
      "poem_meter": "الكامل",
      "poem_theme": "الحنين",
      "poem_era": "حديث",
      "num_verses": 4
    },
    "verses": [
      "أحنُّ إلى بلادي والديار",
      "وتشتاقُ الفؤادَ بها الأزهار",
      "فكم ليلٍ بعيدٍ قد مضى",
      "وفي القلب اشتياقٌ واحترار"
    ]
  },

  "agent_trace": [
    {
      "step": 1,
      "agent": "Orchestrator",
      "tool": "parse_intent",
      "summary": "تم اكتشاف نية \"توليد قصيدة\" عن الحنين للوطن"
    },
    {
      "step": 2,
      "agent": "Library",
      "tool": "rag.search",
      "summary": "تم استرجاع 5 قصائد عن الحنين من مختلف العصور عبر Chroma/Neo4j"
    },
    {
      "step": 3,
      "agent": "SpecBuilder",
      "tool": "build_spec",
      "summary": "تم اختيار بحر الكامل و4 أبيات اعتمادًا على النتائج المسترجعة ووصف المستخدم"
    },
    {
      "step": 4,
      "agent": "Shaer",
      "tool": "generate_verses",
      "summary": "تم توليد 4 أبيات باستخدام نموذج Shaer-7B بالاعتماد على الـ PoemSpec"
    },
    {
      "step": 5,
      "agent": "Scansion",
      "tool": "check_meter",
      "summary": "تم التحقق من الوزن: الأبيات موزونة بشكل صحيح على بحر الكامل"
    }
  ],

  "library_context": [
    {
      "poem_id": "ashaar_12345",
      "title": "الغريب",
      "poet_name": "محمود سامي البارودي",
      "poem_meter": "الكامل",
      "poem_era": "حديث",
      "poem_theme": "الحنين"
    },
    {
      "poem_id": "ashaar_67890",
      "title": "يا نائح الطلح",
      "poet_name": "ابن زيدون",
      "poem_meter": "البسيط",
      "poem_era": "أندلسي",
      "poem_theme": "الحنين"
    }
  ]
}
```

---

## 8. Error Handling

If something goes wrong, respond with:

* Proper HTTP status (e.g. `500`).
* JSON body:

```json
{
  "error": "Internal server error",
  "message": "Failed to generate poem"
}
```

The frontend will show a generic Arabic message like:

> "عذراً، حدث خطأ. يرجى المحاولة مرة أخرى."

---

## 9. CORS

Frontend runs on Vite default: `http://localhost:5173`.

Enable CORS for that origin.

### Example (FastAPI)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 10. Minimum vs Ideal

### Minimum viable response

```json
{
  "reply": "النص هنا"
}
```

Everything else can be `undefined` / missing.

### Ideal response

* Fills:

  * `reply`
  * `mode`
  * `agent_trace` (so Inspector shows meaningful steps)
  * `library_context` (if RAG used)
  * `poem_spec` & `poem_version` (for generate/fix modes)

---

## 11. Quick checklist for backend

To be fully compatible with the frontend:

* ✅ Implement `POST /api/chat`
* ✅ Accept `{ messages: ChatMessage[], mode?: string }`
* ✅ Use full message history (stateless backend)
* ✅ Always return at least `{ reply: string }`
* ✅ Prefer returning:

  * `agent_trace` (for logs of tools/agents)
  * `library_context` (if RAG used)
  * `poem_spec` + `poem_version` (when dealing with poems)
* ✅ Enable CORS for `http://localhost:5173`
* ✅ Handle multi-turn conversations (history in each request)

And yes:
👉 **The logging of which tools/agents were used is exactly `agent_trace`.**
For every internal step (RAG, spec builder, Shaer, Yehia judge, scansion…), push an `AgentStep` entry so the right sidebar can show a transparent, human-readable trace of what the system did.
