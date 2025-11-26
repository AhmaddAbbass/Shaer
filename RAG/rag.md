## 1. Dataset overview

We use a curated poem-level metadata dataset hosted on Hugging Face:

> **Dataset:** `Shaer-AI/ashaar-full-desc-preprocessed`
> **Link:** [https://huggingface.co/datasets/Shaer-AI/ashaar-full-desc-preprocessed](https://huggingface.co/datasets/Shaer-AI/ashaar-full-desc-preprocessed) ([Hugging Face][1])

Each row corresponds to **one Arabic poem (قصيدة)** and collects:

* structured metadata (meter, era, poet, URLs, etc.),
* the full list of verses,
* and a generated **Arabic prose description** of the poem.

This dataset is our **canonical source of truth** for everything RAG-related in the Shaer system:

* It feeds the **graph layer** (Neo4j via nano-graphrag) so we can reason over poets, eras, meters, topics, etc.
* It feeds the **vector store** (Chroma) via short textual descriptions, which we use for semantic retrieval when the orchestrator needs to “find poems about X”.

We **don’t** feed full poems into the embedding model; instead we rely on compact descriptions plus structured metadata, and only pull full verses when we want to display or analyze a poem.

---

## 2. What the dataset actually contains

On HF, the dataset has a single `train` split with ~**118k poems** in the `preprocessed` version. ([Hugging Face][1])
(Your earlier local EDA over a 143k-row version is the “rawer” dump; this HF dataset is a cleaned/trimmed subset.)

### 2.1 Columns and their meaning

From the HF schema and your EDA, each row has at least these fields: ([Hugging Face][1])

* **`poem_title`** (`string | null`): Optional title of the poem.
* **`poem_meter`** (`string`):

  * One of the classical meters: الطويل، الكامل، البسيط، الوافر، الخفيف… (15 total values).
* **`poem_verses`** (`list[string]`):

  * Full poem content, verse by verse.
  * Each entry is a **single hemistich line** annotated with `<s>` (صدر) and `<a>` (عجز) markers from the scraping/normalization stage.
  * List length ranges from 2 up to ~200 lines (i.e., ~1–100 bayts).
* **`poem_theme`** (`string | null`):

  * Coarse tag like: قصيدة قصيره، قصيدة عامه، قصيدة مدح، قصيدة حزينه، قصيدة هجاء، …
  * Many entries are `None` or very generic.
* **`poem_url`** (`string`):

  * Source page for the full poem text, e.g. aldiwan.net or DCT Abu Dhabi poetry portal.
* **`poet_name`** (`string`):

  * The poet’s name (e.g. ابن الرومي، معروف الرصافي).
* **`poet_description`** (`string | null`):

  * Short biographical note or Wikipedia-style description of the poet (often from the source site).
* **`poet_url`** (`string`):

  * Link to the poet’s page on the source site.
* **`poet_era`** (`string | null`):

  * Literary/historical era: العصر العباسي، العصر الأموي، العصر الحديث، العصر المملوكي، قبل الإسلام، … (≈14 values).
* **`poet_location`** (`string | null`):

  * Coarse geography (مصر، العراق، الشام، …) when available.
* **`poem_language_type`** (`string | null`):

  * For this project we can assume everything we care about is **Arabic**.
  * In practice, values are mostly فصيح or `null`.
* **`num_verses`** (`int`):

  * Number of bayts / verse pairs (derived from `poem_verses`).
* **`poem_id`** (`int`):

  * Stable numeric ID you assigned; we’ll treat this as our **primary key** across systems (graph + vector store).
* **`poem_description`** (`string`):

  * Yehia-generated synopsis of the poem in Arabic prose.
  * Length ranges roughly 10–950 characters. ([Hugging Face][1])
  * This is the **main text we embed** in Chroma.
* **`second_pass_candidate`** (`bool` in your local version; may or may not be exposed in the preprocessed HF subset):

  * Marks poems whose initial description was weak (“لا أعلم”, hallucinated metadata, repetition, etc.) and that were scheduled for a second summarization pass.

### 2.2 Quality quirks we know about

From your EDA dump and pipeline history, we know that `poem_description` is not always clean:

1. **“لا أعلم” / “لا أعرف” placeholders**

   * Around **15k poems** in the raw dump have descriptions equal to:

     * `لا أعلم.`, `لا اعلم`, `لا أعرف`, etc.
   * These are **explicit “I don’t know”** outputs from Yehia when it couldn’t summarize / was mis-prompted.
   * For RAG, these are **content-less** and must be treated as **missing descriptions**.

2. **English or mostly English descriptions (“english means trash”)**

   * A small subset have descriptions that are largely Latin characters (half-broken logs, errors, or leftover debugging text).
   * For our poetry RAG, any description that is primarily non-Arabic is **garbage** and we’ll drop it from the embedding index.

3. **Description ≈ poem text**

   * Some rows have `poem_description` that is effectively a **copy of the poem** (or large chunks of it) instead of a prose summary.
   * We can detect this by, e.g.,:

     * Normalizing both text fields (strip diacritics, punctuation, whitespace),
     * Concatenating `poem_verses`,
     * Measuring a character-level similarity (e.g. ratio ≥ 0.9).
   * For RAG this is also undesirable:

     * It inflates token count.
     * It defeats the purpose of “short semantic pointer”.

4. **Generic boilerplate descriptions**

   * Patterns like:

     * “هذه القصيدة من نظم الشاعر فلان…”
     * “في هذه القصيدة يتحدث الشاعر عن…” followed by very vague content.
   * We don’t necessarily drop these, but we may down-weight them later (e.g. mark as `generic_description=True`), because they add little semantic discrimination.

For the **graph**, all poems remain part of the graph (we still care about their meter, era, poet, etc.).
For the **vector store**, any poem whose description is “لا أعلم” / mostly English / just repeated poem text will be **excluded from embedding** and treated as “no semantic summary yet”.

Later, we can regenerate better descriptions and backfill those entries.

---

## 3. How this dataset becomes a graph (Neo4j + nano-graphrag)

Now, how we go from this HF table to a **graph + vector** setup that supports the orchestrator’s tasks.

High-level goals:

* Make it easy to answer:

  * “Give me a poem about longing in the Umayyad era, in بحر الطويل”
  * “Show me poems similar in theme/style to this description”
  * “Show me a poem by المتنبي that talks about courage”
* While **not** embedding huge poem texts, and keeping the graph interpretable.

We’ll have two tightly-coupled layers:

1. **Neo4j graph** – stores structured relationships (poem ↔ poet ↔ era ↔ meter, plus semantic concepts via nano-graphrag).
2. **Chroma vector index** – stores short Arabic **description snippets** per poem (and maybe per verse later).

### 3.1 Node types (what becomes a node)

From `ashaar-full-desc-preprocessed`, we derive the following core node types:

1. **`Poem` nodes** (central)

   * One node per row (per `poem_id`).
   * Properties (examples):

     * `poem_id` (int, **primary key**)
     * `title` (string | null)
     * `meter` (string)
     * `theme_raw` (string | null, original coarse theme)
     * `url` (string)
     * `num_verses` (int)
     * `description_raw` (string, original description)
     * `description_clean` (string | null, cleaned/shortened summary we actually embed)
     * `has_bad_description` (bool; “لا أعلم”, English, or “description == poem”)
     * `source` (enum: `"aldiwan"`, `"dctabudhabi"`, etc., inferred from URL)
     * `second_pass_candidate` (bool, if present)
     * optional quality flags, e.g. `is_generic_description` (bool)

2. **`Poet` nodes**

   * One node per distinct poet (`poet_url` or `(poet_name, poet_era)` as key).
   * Properties:

     * `name`
     * `url`
     * `description` (bio)
     * `location` (country/region)
     * maybe `num_poems` cached
   * Connected to all their poems.

3. **`Era` nodes**

   * One node per `poet_era` (العصر العباسي، العصر الحديث، …).
   * Properties:

     * `name`
     * maybe `period_start`, `period_end` if we add that later manually.

4. **`Meter` nodes**

   * One node per `poem_meter` (الطويل, البسيط, الكامل, …).
   * Properties:

     * `name`
     * maybe `foot_pattern` (تفعيلة pattern) later if we want.

5. **`Theme` nodes (optional / predicted)**

   * We don’t rely on the current `poem_theme` much (as you said), but:

     * We can **still store it as raw metadata** on the Poem node.
     * Later, when we run Ashaar’s theme classifier, we can connect Poem → Theme nodes:

       * `(:Poem)-[:HAS_THEME]->(:Theme {name: "غزل"})`
   * For now, `Theme` nodes may be partial or missing. That’s okay; we don’t block on it.

6. **`SourceSite` nodes (optional)**

   * One node per origin site (`aldiwan`, `poetry.dctabudhabi.ae`, …).
   * Allows queries like:

     * “Show me all poems scraped from Diwan vs DCT”.
   * Relationship: `(:Poem)-[:FROM_SITE]->(:SourceSite)`.

7. **`Concept` / `Entity` nodes from nano-graphrag**

   * We’ll feed poem descriptions (and optionally some verse text) into **nano-graphrag**, which extracts:

     * entities (أسماء، أماكن، مشاعر, etc.),
     * relations (“poem X mentions city Y”, “poem Y expresses sadness”, …).
   * nano-graphrag outputs a small semantic graph per batch of texts; we then import that into Neo4j as:

     * `(:Concept {label: "...", type: "emotion"|"topic"|"place"|...})`

       * e.g. `Concept("الحنين")`, `Concept("الشجاعة")`, `Concept("الحب")`.
     * Relationships like:

       * `(:Poem)-[:MENTIONS_CONCEPT]->(:Concept)`
       * `(:Concept)-[:RELATED_TO]->(:Concept)`

   This gives us a **semantic layer** on top of the clean metadata graph, and it’s entirely derived from `poem_description_clean`.

### 3.2 Edge types (how nodes connect)

Some key relationships:

* `(:Poem)-[:WRITTEN_BY]->(:Poet)`
* `(:Poem)-[:IN_ERA]->(:Era)`
* `(:Poem)-[:IN_METER]->(:Meter)`
* `(:Poem)-[:FROM_SITE]->(:SourceSite)`
* `(:Poem)-[:HAS_THEME]->(:Theme)` (later, once we trust theme classification)
* `(:Poem)-[:MENTIONS_CONCEPT]->(:Concept)` (from nano-graphrag)
* `(:Concept)-[:RELATED_TO]->(:Concept)` (semantic neighbors)
* Optionally:

  * `(:Poem)-[:SIMILAR_STYLE_TO {score: ...}]->(:Poem)`

    * We can create this later by running a batch similarity job over embeddings.

This structure means:

* Simple, structured queries (meter/era/poet) stay in the **graph**.
* Fuzzy “about X” queries hit the **vector index** and then hop into the graph via the `poem_id`.

---

## 4. Vector store: how we use descriptions (and what we exclude)

### 4.1 Why we embed only short descriptions

You explicitly don’t want to embed big blobs of text:

> “embedding large text is bad … the only important thing is some sentence that says the description, and we retrieve accordingly, and then we can filter.”

We follow that religiously:

* We **do not embed**:

  * full `poem_verses`,
  * full `poet_description`.
* We **only embed**:

  * a relatively short, cleaned **Arabic description** of each poem.

Concretely for each poem row:

1. Take `poem_description`.
2. Clean it:

   * strip whitespace, normalize Arabic letters,
   * remove obvious boilerplate patterns (“هذه القصيدة من نظم الشاعر…”).
3. If it is too long (e.g. > 2–3 sentences), we may later re-summarize it with Yehia/Shaer into **one concise sentence** (planned, optional).
4. Store the result as `description_clean`.

If **any** of the following holds:

* description is `None` or empty after cleaning,
* description is in the “لا أعلم” family,
* description is mostly Latin (English “trash”),
* description is almost identical to concatenated `poem_verses`,

then we set `has_bad_description=True` and **skip this poem** when building the Chroma index.

The poem still exists in Neo4j; it just won’t be hit by semantic description search until we regenerate a decent summary.

### 4.2 What goes into Chroma

For each “good” poem description:

* **document text**:

  * `doc = description_clean`
* **metadata**:

  ```jsonc
  {
    "poem_id": 12345,
    "poet_name": "ابن الرومي",
    "poem_meter": "الطويل",
    "poet_era": "العصر العباسي",
    "source": "aldiwan",
    "num_verses": 19
  }
  ```
* **embedding**:

  * computed using an Arabic-friendly embedding model (to be chosen later).

We then store this in **Chroma**, with a **document ID exactly equal to `poem_id`**.
This 1-to-1 mapping is what lets us go from “top-k results in Chroma” → “Poem nodes in Neo4j” trivially.

---

## 5. Implementation plan (step-by-step, talking to the reader)

Here is how all of this will be implemented, concretely, from the viewpoint of the RAG pipeline engineer.

### Phase 0 – Load dataset from HF

* Use the Hugging Face `datasets` library with the **Parquet** backend: ([Hugging Face][1])

  * `load_dataset("Shaer-AI/ashaar-full-desc-preprocessed", split="train")`
* Validate schema matches what we expect:

  * columns listed above,
  * `num_rows ≈ 118k`.

### Phase 1 – Clean descriptions & classify quality

For each row:

1. Normalize `poem_description`:

   * trim,
   * unify Arabic characters (أ/إ/ا…),
   * remove repeated whitespace.

2. Detect **“لا أعلم” / “لا أعرف”** style:

   * Lowercase & strip punctuation.
   * If the text is in a small whitelist set:

     * mark `has_bad_description=True`.

3. Detect **English / non-Arabic**:

   * Count Arabic vs non-Arabic characters.
   * If ratio of Arabic letters is below a threshold (e.g. < 0.6):

     * `has_bad_description=True`.

4. Detect **description ≈ poem**:

   * Concatenate `poem_verses` into one string, normalize like above.
   * Compute a simple similarity ratio with the description (e.g. token overlap, Levenshtein, etc.).
   * If ratio ≥ threshold:

     * `has_bad_description=True`.

5. Remove boilerplate patterns:

   * If text starts with something like:

     * “تتحدث القصيدة عن”، “في هذه القصيدة”، “هذه القصيدة من نظم…”
   * Strip only the **boring preamble** and keep the “meat”.

6. If after cleaning the description is too long:

   * Optionally mark `needs_resummarization=True` for a later pass (not blocking now).

7. Set:

   * `description_clean = cleaned_text or None`.
   * `has_bad_description` as computed.

We now have a **cleaned view** that the rest of the pipeline can trust.

### Phase 2 – Build and populate the Neo4j graph

We spin up Neo4j (Dockerized) and then run an ingestion job:

1. For each distinct **Poet**:

   * Create (or merge) a `(:Poet)` node with:

     * `name`, `url`, `description`, `location`.

2. For each distinct **Era**, **Meter**, and **SourceSite**:

   * Create (or merge) `(:Era)`, `(:Meter)`, `(:SourceSite)` nodes.

3. For each **Poem row**:

   * Create `(:Poem {poem_id, title, meter, theme_raw, url, num_verses, description_raw, description_clean, has_bad_description, source})`.
   * Link it:

     * `[:WRITTEN_BY]` → Poet
     * `[:IN_ERA]` → Era (if known)
     * `[:IN_METER]` → Meter
     * `[:FROM_SITE]` → SourceSite
   * Optionally:

     * Cache the first few verses as `preview_verse` property for UI.

Now we have a **clean metadata graph** independent of embeddings.

### Phase 3 – Build Chroma index

In a separate step (but reading the same HF dataset / cleaned frame):

1. Iterate over all poems with `has_bad_description == False` and `description_clean != None`.
2. For each poem:

   * Build the document string = `description_clean`.
   * Compute embedding and insert into Chroma with:

     * `id = poem_id`
     * `metadata` as above.
3. Persist Chroma index (also Dockerized) and expose it via HTTP (or directly via Python in the backend).

### Phase 4 – Add semantic concepts via nano-graphrag

To give the graph more **semantic richness**:

1. Take batches of `description_clean` (for poems with good descriptions).
2. Feed them to **nano-graphrag**, which:

   * extracts entities,
   * proposes relations between concepts and documents.
3. Convert nano-graphrag’s output into CSV/JSON and load into Neo4j:

   * `(:Concept)` nodes with a `label` and `type`,
   * `(:Poem)-[:MENTIONS_CONCEPT]->(:Concept)`,
   * `(:Concept)-[:RELATED_TO]->(:Concept)`.

Now the graph can answer queries like:

* “Show me poems linked to concepts {الحنين, الفراق} in العصر الأموي and بحر الطويل”
* “Cluster poems by concept graph neighborhoods”.

### Phase 5 – Orchestrator integration

Once this graph + vector store are in place, the agentic orchestrator (ChatGPT / other LLM with MCP tools) can:

* Hit **Chroma** to find semantically relevant poems based on a user’s Arabic description.
* Use **graph filters** (via a Neo4j tool):

  * filter by meter, era, poet, theme.
* Retrieve the **full poem_verses** and source URLs for display.
* Use the same graph to find “neighbors” (similar meters, similar concepts) when it wants to suggest related poems.

All of this is backed by **one dataset**:

> `Shaer-AI/ashaar-full-desc-preprocessed` ([Hugging Face][1])

…and we’re very explicit that:

* Bad descriptions (“لا أعلم”, English, or duplicated poem text) **do not** go into the embedding index.
* Those poems still live in the graph and can be retrieved structurally (e.g., “all poems by ابن زيدون in الكامل”) even before we fix their descriptions.

