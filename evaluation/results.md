# Results

## Scope

This note records the current evaluation setup and the results available so far for the advisor draft.

We evaluate generated poems along two groups of metrics:

1. **Formal structural metrics**, computed separately:
   - `meter`
   - `count_adherence`

2. **Semantic and literary judge metrics**, scored by an LLM judge:
   - `description_adherence`
   - `meaning`
   - `fluency`
   - `coherence`
   - `poeticness`

The LLM-judge results below use the current frozen evaluation pipeline and should be treated as the current draft numbers. `Fanar` is still pending a rerun after a fail-fast interruption, so its judge table is still partial at the time of writing.

## Evaluated systems

- `Shaer`: `Shaer-AI/shaer-sft-test`
- `Ashaar`: `Shaer-AI/shaer-eval-ashaar-native-controls`
- `Yehia`: `Shaer-AI/shaer-eval-instruction-yehia-base-sft-chat-template`
- `Fanar`: `Shaer-AI/fanar-eval-native-prompt`

## Judge setup

### Judge model

The final judge model is:

- `qwen/qwen3-235b-a22b-2507`

This choice was made from the 50-reference-poem calibration study documented in [judge_choice.md](./judge_choice.md).

### Scoring scale

All judge metrics use a strict integer scale:

- `1` = very poor
- `2` = poor
- `3` = acceptable / mixed
- `4` = good
- `5` = excellent

### Evaluation mode

- One metric is scored at a time.
- The judge returns only a structured integer score.
- All runs use the same frozen prompt pack in [judge_prompts.yaml](./judge_prompts.yaml).
- Output format is enforced through a function/tool call with the schema:

```json
{"score": 1}
```

### Metric applicability by model

`description_adherence` is only evaluated for models that are actually given the description-conditioned task:

- `Shaer`: yes
- `Yehia`: yes
- `Ashaar`: no
- `Fanar`: no

This is intentional. `Ashaar` and `Fanar` are not evaluated on `description_adherence` because their native prompting interfaces are not the same description-driven task used by `Shaer` and `Yehia`.

## Frozen judge prompts

### Shared system prompt

```text
You are an expert evaluator of classical-style Arabic poetry.

Evaluate only the requested metric.
Return valid JSON only in this exact format:
{"score": <integer from 1 to 5>}
```

### Description Adherence

```text
Input description:
{description}

Generated poem:
{poem}

Metric: Description Adherence

Definition:
Description Adherence measures how well the poem follows the requested topic and any requested scene, emotion, speaker stance, imagery, or semantic constraints, if present.

Important:
Judge only against the input description.

Scoring:
1 = The poem does not follow the description or contradicts it.
2 = The poem has only a weak or generic relation to the description.
3 = The poem partially follows the description but misses important requested elements.
4 = The poem mostly follows the description with minor omissions or mild genericness.
5 = The poem strongly and naturally realizes the description.

Return:
{"score": <integer from 1 to 5>}
```

### Meaning

```text
Generated poem:
{poem}

Metric: Meaning

Definition:
Meaning measures whether the poem conveys clear, non-trivial, internally valid poetic meaning rather than empty, repetitive, or contradictory phrasing.

Scoring:
1 = The poem has no clear meaning, is mostly nonsensical, or is severely contradictory.
2 = The poem has very shallow, repetitive, or weak meaning.
3 = The poem has understandable meaning, but it is generic, thin, or only partly developed.
4 = The poem has clear and worthwhile meaning with some semantic or emotional depth.
5 = The poem has rich, clear, non-trivial poetic meaning with strong semantic or emotional depth.

Return:
{"score": <integer from 1 to 5>}
```

### Fluency

```text
Generated poem:
{poem}

Metric: Fluency

Definition:
Fluency measures the grammatical, morphological, syntactic, and lexical well-formedness of the Arabic.

Scoring:
1 = The Arabic is largely broken, ungrammatical, or difficult to parse.
2 = The Arabic has many errors that significantly harm readability.
3 = The Arabic is understandable but has noticeable awkwardness or repeated errors.
4 = The Arabic is mostly fluent with only minor awkwardness or small errors.
5 = The Arabic is smooth, correct, and natural for classical-style Arabic poetry.

Return:
{"score": <integer from 1 to 5>}
```

### Coherence

```text
Generated poem:
{poem}

Metric: Coherence

Definition:
Coherence measures whether the poem forms a unified whole across lines through semantic, emotional, and rhetorical continuity, not mere topic repetition.

Scoring:
1 = The poem is fragmented, contradictory, or lacks connection between lines.
2 = The poem has frequent abrupt shifts and weak overall unity.
3 = The poem has some unity, but several lines feel loosely connected or stitched together.
4 = The poem is mostly unified with minor looseness or weak transitions.
5 = The poem has strong continuity and unity across its lines.

Return:
{"score": <integer from 1 to 5>}
```

### Poeticness

```text
Generated poem:
{poem}

Metric: Poeticness

Definition:
Poeticness measures the non-prosodic literary quality of the poem, including imagery, metaphor, rhetorical elegance, emotional resonance, and poetic diction.

Important:
Do not judge meter, rhyme consistency, or line-count adherence.

Scoring:
1 = The poem is not meaningfully poetic; it is flat, mechanical, or prose-like.
2 = The poem is weakly poetic, mostly cliche, artificial, or emotionally flat.
3 = The poem has some poetic qualities but is inconsistent or generic.
4 = The poem is clearly poetic, with good imagery, diction, or emotional force.
5 = The poem is strongly poetic, vivid, elegant, emotionally resonant, and rhetorically rich.

Return:
{"score": <integer from 1 to 5>}
```

## Structural metric context

The main formal metric of interest is meter adherence. The current comparison set gives the following average meter scores:

| Model | Meter Mean | Evaluation setup |
|---|---:|---|
| `Shaer` | `0.9064` | selected best generation per held-out prompt |
| `Ashaar` | `0.8535` | native controls |
| `Fanar` | `0.6488` | native prompt |
| `Yehia` | `0.1494` | instruction baseline |

These structural results are important context for reading the judge metrics below: the judge metrics measure semantic and literary quality, not formal metrical control.

## Judge results so far

### Current judge table

| Model | Rows scored | Description Adherence | Meaning | Fluency | Coherence | Poeticness |
|---|---:|---:|---:|---:|---:|---:|
| `Shaer` | `3481 / 3481` | `4.83` | `4.04` | `4.69` | `4.28` | `4.28` |
| `Ashaar` | `3481 / 3481` | `n/a` | `3.32` | `4.04` | `3.70` | `3.68` |
| `Yehia` | `3481 / 3481` | `4.56` | `4.07` | `4.69` | `4.39` | `4.16` |
| `Fanar` | `1156 / 3481` | `n/a` | `4.06` | `4.68` | `4.28` | `4.29` |

### Status note on Fanar

`Fanar` is not yet final in this table.

The first Fanar pass was interrupted by the worker fail-fast policy after four isolated row-level failures. The partial averages above reflect the rows completed before the interruption. Fanar is being rerun separately so the final Fanar values in the table should be updated after the rerun completes.

## Current interpretation

### 1. Shaer vs Ashaar

The current results strongly separate `Shaer` from `Ashaar` on the non-formal judge metrics:

- `Shaer` is clearly higher on `meaning`
- `Shaer` is clearly higher on `coherence`
- `Shaer` is clearly higher on `poeticness`
- `Ashaar` remains reasonably fluent, but its semantic and literary scores are consistently lower

Taken together with the structural results, this supports the view that `Ashaar` can produce metrically plausible and often readable verse, but is materially weaker than `Shaer` on controlled semantic and literary quality under this benchmark.

### 2. Shaer vs Yehia

The more interesting result is the comparison with `Yehia`.

`Yehia` remains strong on the judge metrics and stays broadly competitive with `Shaer` on `meaning`, `fluency`, `coherence`, and `poeticness`. This is not a negative result for `Shaer`. Instead, it suggests a cleaner and more defensible story:

- `Yehia` already provides strong underlying Arabic poetic quality
- `Shaer` preserves that quality
- `Shaer` substantially improves **formal control**, especially meter

In other words, the current evidence suggests that the `Shaer` SFT adapter adds strong prosodic and structural control without sacrificing the base model’s broader poetic quality.

### 3. Fanar

The partial Fanar numbers are currently high enough that they should not be interpreted casually. Two things can both be true:

- `Fanar` may be strong on literary surface quality
- `Shaer` may still be the more useful system for controllable generation and formal metrical adherence

The final interpretation for `Fanar` should wait until the rerun finishes.

## Paper-ready takeaway sentence

The cleanest current sentence for the paper is:

> `Shaer` preserves the strong semantic and literary quality of the `Yehia` base model while substantially improving formal poetic control, especially metrical adherence.

If a slightly stronger comparative sentence is needed:

> Relative to `Ashaar`, `Shaer` achieves materially better semantic meaning, coherence, and poeticness, while also delivering much stronger controllable metrical generation.

## Recommended framing in the draft

The results section should avoid claiming that `Shaer` dominates every model on every axis. The stronger and more defensible framing is:

1. `Yehia` is already a strong poetic base model.
2. `Shaer` retains comparable literary quality.
3. `Shaer` adds much stronger formal control, especially in meter.
4. `Ashaar` is weaker on semantic and literary quality in this evaluation.
5. `Fanar` is still pending a finalized judge rerun.
