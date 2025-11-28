# common_schemas – Shared Data Models for Shaer

This package defines the **core data shapes** used across all backend services:

- `yehia_client`
- `shaer_client`
- `rag_service`
- `meter_service` (BiLSTM meter checks)
- `ashaar_meter_service` (Ashaar structural scoring)
- future agents / orchestrator

Having a single source of truth here avoids “stringly-typed JSON” and keeps
the services aligned on field names and semantics.

---

## 1. Files

- `__init__.py`  
  Minimal; just makes the package importable and can re-export the models.

- `schemas.py`  
  Contains all shared Pydantic `BaseModel` classes.

All other service-level `schemas.py` files should be **thin wrappers** that
import and embed these core models in HTTP request/response shapes.

---

## 2. Core Poem Models

### 2.1 `PoemSpec`

**What it represents**

A **structured plan** for a poem the system should generate.

This is the bridge between fuzzy user intent and the strict prompt format
Shaer-7B was trained on.

**Fields**

- `poem_title: Optional[str]`  
  Optional title; nice-to-have, not required.

- `poem_meter: str`  
  Target meter (e.g. `"البسيط"`, `"الكامل"`).  
  This is **required** and is what we pass to:
  - Shaer (in the generation prompt),
  - meter_service (as `target_meter`).

- `poem_theme: Optional[str]`  
  High-level theme, e.g. `غزل`, `رثاء`, `حماسة`, etc.

- `poem_era: Optional[str]`  
  Era such as `العصر الجاهلي`, `العصر العباسي`, `العصر الحديث`.

- `poet_name: Optional[str]`  
  Either a real poet (“المتنبي”) or a style hint
  (“أسلوب قريب من نزار قباني”).

- `poem_description: str`  
  Arabic prose summary of **the whole poem** (similar style to the
  Ashaar dataset descriptions). This field is crucial because Shaer
  conditions heavily on it.

- `num_verses: int`  
  Number of bayts to generate in the poem.

- `poem_language_type: Optional[str]`  
  e.g. `فصحى`, `عامية`, etc. Currently optional.

- `style_notes: Optional[str]`  
  Extra stylistic instructions (“صور مكثفة”, “لغة بسيطة”, ...).

**Who uses it**

- **Produced by**:  
  - `yehia_client` (`/build-spec`) – from user query + RAG hits.  

- **Consumed by**:  
  - `shaer_client` (`/generate-bayt`, `/generate-poem`)  
- `meter_service` (uses `poem_meter`)  
  - Orchestrator agent (for reasoning about constraints)  
  - `yehia_client` (`/feedback`) as context when critiquing a bayt.

---

### 2.2 `Verse`

Represents **one bayt** produced or processed by the system.

- `index: int` – 1-based position in the poem.  
- `text: str` – full bayt on one line (“الصدر ... العجز”).

**Usage**

- `shaer_client` returns `Verse` from `/generate-bayt`.  
- Orchestrator builds `previous_verses: list[Verse]` to feed Shaer.  
- `meter_service` takes `verse.text` plus `PoemSpec.poem_meter`.

---

### 2.3 `Poem`

A complete poem instance in the system.

- `spec: PoemSpec` – the plan used to generate it (or inferred from RAG).  
- `verses: list[Verse]` – ordered bayts.  
- `poem_id: Optional[str]` – ID if it exists in the corpus / Neo4j.  
- `source_url: Optional[str]` – original link.  
- `detected_meter: Optional[str]` – meter predicted by some classifier.  
- `rhyme: Optional[str]` – rhyme letter if known.

**Usage**

- Output of `/generate-poem` in `shaer_client` (optional endpoint).  
- Wrapper around `RagPoemRecord` when converting retrieved poems into the
  generation format.

---

## 3. Evaluation & Feedback Models

### 3.1 `BaytMeterEval`

Output of the **meter_service** for a single bayt.

Fields:

- `target_meter: str`  
  The meter we expected (usually `PoemSpec.poem_meter`).

- `meter_score: int (0–100)`  
  Quantitative score of how well the bayt matches the meter.

- `on_meter: bool`  
  High-level decision: acceptable or not (based on a threshold).

- `notes: str`  
  Short explanation in Arabic highlighting where the rhythm breaks, etc.

**Usage**

- **Produced by**: `meter_service /eval-bayt`.  
- **Consumed by**:  
  - Orchestrator / Enhancer agent to decide whether to regenerate a bayt.  
  - Any downstream analytics / UI that wants to show meter quality.

---

### 3.2 `YehiaFeedback`

High-level semantic/style feedback from Yehia about a bayt.

Fields:

- `ok: bool`  
  Whether the bayt is acceptable overall (meaning, coherence, adherence
  to instructions).

- `score: Optional[int (0–100)]`  
  Optional scalar quality score (alignment with `PoemSpec`).

- `feedback: str`  
  Free-form Arabic explanation and suggestions.

**Usage**

- **Produced by**: `yehia_client /feedback`.  
- **Consumed by**:
  - Enhancer agent: together with `BaytMeterEval` to decide if the bayt
    should be kept or re-generated.
  - UI: show feedback to the user in “teacher mode” / “critique mode”.

---

## 4. RAG Models

These are used between the orchestrator / Yehia and the `rag_service`.

### 4.1 `RagSearchRequest`

Input to `rag_service.search`.

- `query_text: str` – natural-language query.  
- `top_k: int` – number of hits requested.

### 4.2 `RagSearchHit`

One candidate poem from semantic search:

- `poem_id: str` – matches dataset/Neo4j IDs.  
- `poem_title: Optional[str]`  
- `poet_name: Optional[str]`  
- `poem_description: str` – cleaned summary.  
- `poem_meter: Optional[str]`  
- `poem_era: Optional[str]`  
- `poem_theme: Optional[str]`  
- `has_bad_description: bool` – flagged low quality if true.

**Usage**

- `rag_service /search` returns `RagSearchResponse` with a list of these.  
- `yehia_client /build-spec` takes some hits as exemplars/hints when
  inferring `PoemSpec`.

### 4.3 `RagSearchResponse`

Just wraps:

- `hits: list[RagSearchHit]`

### 4.4 `RagPoemRecord`

Full poem record fetched from Neo4j / corpus:

- `poem_id: str`  
- `poem_title: Optional[str]`  
- `poet_name: Optional[str]`  
- `poem_meter: Optional[str]`  
- `poem_era: Optional[str]`  
- `poem_theme: Optional[str]`  
- `num_verses: int`  
- `verses: list[str]` – each element is a bayt as stored in the corpus.  
- `source_url: Optional[str]`

**Usage**

- Returned by `rag_service /poem/{id}`.  
- Can be converted into a `Poem` when needed by other parts of the system.

---

## 5. LLM Messaging

### 5.1 `LLMMessage`

Minimal representation of a chat-style message:

- `role: str` – `"system" | "user" | "assistant"`.  
- `content: str` – text payload.

**Usage**

- Internal to `yehia_client` and `shaer_client`:
  - `prompt.py` in each service returns `list[LLMMessage]`.  
  - `client.py` converts these into whatever shape the RunPod endpoint
    expects (e.g. `messages` with roles + content).

We keep this tiny and generic so it doesn’t depend on any specific
library’s `ChatCompletionMessage` type.

---

## 6. How Services Use These Schemas

- **yehia_client**
  - Inputs/outputs:
    - Uses `PoemSpec`, `YehiaFeedback`, `RagSearchHit`, `LLMMessage`.
  - Public endpoints wrap these models in service-specific
    request/response schemas.

- **shaer_client**
  - Uses `PoemSpec` and `Verse` as the main types for generation.
  - Its `prompt.py` converts `(PoemSpec, list[Verse], sequence_number)`
    into `list[LLMMessage]`.

- **rag_service**
  - Uses `RagSearchRequest`, `RagSearchResponse`, `RagPoemRecord`.

- **meter_service**
- **ashaar_meter_service** (reports Ashaar structural similarity scores without target meter).
  - Uses `BaytMeterEval` as the result type for `/eval-bayt`.

- **Orchestrator / Agents**
  - Glue everything together using:
    - `PoemSpec` to describe target poems.
    - `Poem` to represent full outputs.
    - `BaytMeterEval` + `YehiaFeedback` to drive enhancement loops.
    - `RagSearchHit` / `RagPoemRecord` for retrieval-heavy flows.

---

## 7. Design Philosophy

- Keep **common_schemas small and stable**: only put types here that
  truly need to be shared across multiple services.

- Service-specific request/response shapes (e.g.
  `GenerateBaytRequest`, `EvalBaytRequest`) should live in each
  service’s own `app/schemas.py` and **compose** these core models,
  instead of redefining fields manually.

- If you find yourself copy-pasting a data structure across services,
  it probably belongs in `common_schemas.schemas`.
