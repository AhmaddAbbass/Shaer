# Meaning Judge Benchmarking: Thinking vs No-Thinking on the Fixed 3-Group Arabic Poetry Meaning Benchmark

## What this benchmark was trying to answer

This benchmark was built to answer one very specific operational question inside the Arabic poetry GRPO workflow:

**When we use Qwen through OpenRouter as the meaning judge, should we run it in a reasoning-enabled mode ("thinking") or in a true non-reasoning mode ("no_thinking")?**

This was not a full GRPO experiment.  
It was not a generation-quality experiment on the poetry model itself.  
It was a **judge-mode benchmark** on a fixed benchmark bank, with the goal of deciding which judge mode should be used in the next prompt-tuning loop.

The broader project goal is to make the **meaning reward** reliable enough for phase-1 GRPO, where the active rewards are:

- meter
- meaning
- count adherence

Inside that phase, the meaning judge is supposed to do **two things at once**:

1. measure **semantic faithfulness** to the requested description
2. measure **semantic substance** of the poem itself

That is why the judge outputs two scores instead of one:

- `fit_score`
- `substance_score`

and the notebook aggregates them as:

\[
\text{final\_score} = 0.70 \cdot \text{fit\_score} + 0.30 \cdot \text{substance\_score}
\]

---

## What was fixed before this final benchmark

A major issue was discovered during the earlier mode-comparison attempts:

the old request path for `"no_thinking"` was **not actually disabling reasoning**.

The notebook had been sending a mode flag through `chat_template_kwargs`, but the returned responses still contained:

- reasoning text
- reasoning details
- reasoning tokens

So although the request *looked* like a non-thinking call, the backend response clearly showed that reasoning was still happening.

This was fixed by switching to the correct OpenRouter reasoning control:

- **thinking** → `reasoning={"enabled": True}`
- **no_thinking** → `reasoning={"effort": "none"}`

and by adding provider constraints so unsupported parameters would not be silently ignored:

- `provider.require_parameters = True`
- `provider.allow_fallbacks = False`

After that fix, the no-thinking calls became genuinely non-thinking, confirmed by:

- `reasoning = None`
- no `reasoning_details`
- `reasoning_tokens = 0`

That correction is the reason this final benchmark matters: it is the first clean benchmark where the no-thinking path is actually real.

---

## The benchmark bank used here

The benchmark used a fixed **30-row bank**, with **10 source poems** arranged into **3 groups**:

### Group 1 — gold / correct pair
- original poem
- original description

This checks whether good real poems still receive strong meaning scores.

Expected behavior:
- high `fit_score`
- high `substance_score`
- high `final_score`

### Group 2 — wrong description
- original poem
- swapped-in wrong description

This checks whether the judge can reject semantic mismatch while still recognizing that the poem itself may remain meaningful.

Expected behavior:
- low `fit_score`
- moderate `substance_score`
- clearly lower `final_score` than group1

### Group 3 — bad-substance same-topic corruption
- original description
- intentionally weakened poem

This checks whether the judge can detect **semantic hollowness / weakened content** without requiring total nonsense or full topic drift.

Expected behavior:
- moderate `fit_score`
- lower `substance_score`
- `final_score` below group1

The benchmark bank was designed specifically so that the meaning judge could be tested on the three cases that matter most:

- genuine success
- topic mismatch
- same-topic but weak content

---

## How the judge works in this benchmark

The meaning judge prompt asks for two separate outputs:

### `fit_score`
How well the poem matches the **central semantic frame** of the description.

This is not meant to be literal paraphrase matching.  
The judge is supposed to look for the main topic, main relation, main situation, and rhetorical direction.

### `substance_score`
How much real meaning the poem itself carries.

This is meant to punish:
- empty filler
- shallow topical wording
- weak paraphrase-like output
- vague repetition
- semantically flat poetry

while still allowing a poem to be simple if it genuinely says something.

The final notebook kept the meaning prompt fixed and only changed the **judge mode**.

---

## What exactly was run in the final notebook

The final clean notebook did the following:

1. Loaded the fixed benchmark bank (`30` rows total).
2. Reused the already-saved **thinking** results from the artifacts folder.
3. Ran a fresh pass of **corrected no-thinking** using the fixed request format.
4. Verified that the no-thinking outputs actually respected the no-thinking mode.
5. Recomputed:
   - merged results
   - by-group summaries
   - by-mode summaries
   - pairwise comparisons
   - mode scores
   - winner JSON
6. Packaged the clean artifacts into a downloadable zip.

The final saved run metadata says:

- `n_benchmark_rows = 30`
- `n_thinking_rows = 30`
- `n_no_thinking_rows = 30`
- `n_merged_rows = 60`

The corrected no-thinking run had:

- `invalid_no_thinking_count = 0`

So every saved no-thinking row in the final artifact is a real non-thinking call.

---

## How the winner was decided

The notebook does not pick a winner using only latency or only average score.

It uses a composite rule that values four things:

1. **Group1 vs Group2 separation**
   - a good judge should score correct pairs high and wrong-description pairs low

2. **Group1 vs Group3 separation**
   - a good judge should still distinguish genuinely strong poems from weaker same-topic poems

3. **Group3 substance sensitivity**
   - a good judge should not over-reward semantically weak group3 corruptions

4. **Operational quality**
   - lower latency
   - fewer failures
   - valid mode behavior

The saved `mode_scores.csv` contains the ingredients of that comparison.

---

## Final benchmark results

## 1) By-group summary

### No-Thinking

- **Group1**
  - fit mean: `0.910`
  - substance mean: `0.795`
  - final mean: `0.8755`

- **Group2**
  - fit mean: `0.165`
  - substance mean: `0.470`
  - final mean: `0.2565`

- **Group3**
  - fit mean: `0.615`
  - substance mean: `0.310`
  - final mean: `0.5235`

### Thinking

- **Group1**
  - fit mean: `0.770`
  - substance mean: `0.650`
  - final mean: `0.7340`

- **Group2**
  - fit mean: `0.155`
  - substance mean: `0.580`
  - final mean: `0.2825`

- **Group3**
  - fit mean: `0.635`
  - substance mean: `0.315`
  - final mean: `0.5390`

---

## 2) By-mode overall summary

### No-Thinking
- fit mean: `0.5633`
- substance mean: `0.5250`
- final mean: `0.5518`
- latency mean: `2.1637s`
- latency median: `2.1614s`
- latency p90: `2.5373s`
- error count: `0`
- invalid mode count: `0`

### Thinking
- fit mean: `0.5200`
- substance mean: `0.5150`
- final mean: `0.5185`
- latency mean: `72.8524s`
- latency median: `28.0325s`
- latency p90: `64.9126s`
- error count: `1`
- invalid mode count: `0`

---

## 3) Separation quality

The two most important separation gaps are:

### Group1 − Group2
This is the main “correct pair vs wrong-description pair” gap.

- **no_thinking:** `0.6190`
- **thinking:** `0.4515`

This is a major win for no-thinking.

### Group1 − Group3
This is the main “true strong poem vs weaker same-topic poem” gap.

- **no_thinking:** `0.3520`
- **thinking:** `0.1950`

Again, no-thinking separates the groups more clearly.

### Group3 substance mean
Lower is better here, because group3 is supposed to be weaker.

- **no_thinking:** `0.310`
- **thinking:** `0.315`

This is effectively a tie, with a tiny edge to no-thinking.

---

## What happened inside the results

## A) No-thinking was dramatically faster

This is the clearest operational result in the whole benchmark.

- no-thinking median latency: **~2.16s**
- thinking median latency: **~28.03s**

And the tail was even more dramatic:

- no-thinking p90: **~2.54s**
- thinking p90: **~64.91s**

So even before discussing scores, no-thinking is far easier to use for real iterative prompt tuning.

---

## B) No-thinking also produced cleaner benchmark behavior

No-thinking gave:

- much stronger group1 scores
- lower group2 final scores
- slightly lower group3 substance

So this is not just a “speed winner.”  
It also behaved better on the benchmark’s intended structure.

The strongest evidence is the separation:

- bigger group1/group2 gap
- bigger group1/group3 gap

That is exactly what we want from the meaning judge at this stage.

---

## C) Thinking had one catastrophic failure that damaged its results

The worst example in the entire saved run is a **group1 thinking** row with:

- `fit_score = 0.0`
- `substance_score = 0.0`
- `final_score = 0.0`
- `notes = json_not_found`
- `latency_sec = 766.997`

That is a correct gold example which should never have collapsed to zero like this.

So the thinking mode not only ran slower, it also produced at least one major failure on a gold pair.

That alone is enough to seriously weaken confidence in it for repeated notebook iteration.

---

## D) Group2 is still not perfect

Although no-thinking won overall, group2 is not “solved.”

There are still some wrong-description rows that receive non-trivial fit.

For example, one no-thinking group2 row reached:

- `fit_score = 0.45`
- `substance_score = 0.65`
- `final_score = 0.51`

So the judge still has some cases where it gives too much semantic credit to a mismatched pair.

This is a real remaining problem, but it is a **prompt-quality problem**, not a mode-choice problem.

---

## E) Group3 is still not perfect either

Group3 is the hardest part of the benchmark, and the saved outputs show that it still needs work.

Some no-thinking group3 rows remain too healthy:

- fit up to `0.95`
- substance up to `0.45`
- final up to `0.77`

That means some of the selected “bad-substance” corruptions are still semantically too strong, or at least strong enough that the judge reasonably rewards them.

This matters because if group3 examples are too good, they stop functioning as sharp tests of semantic hollowness.

So the next real bottleneck after mode choice is still:

- improving the **group3 corruption / selection pipeline**
- then continuing meaning prompt tuning on top of that

---

## Final winner

The saved `winner.json` says:

```json
{
  "winner": "no_thinking",
  "thinking_score": -0.6339514858245856,
  "no_thinking_score": 1.2784734753608702,
  "fit_weight": 0.7,
  "judge_model": "qwen/qwen3.5-35b-a3b",
  "thinking_source": "/content/artifacts/thinking_mode_benchmark_v1_results.csv",
  "invalid_no_thinking_count": 0
}
```

So the final notebook winner is:

# **Winner: `no_thinking`**

---

## Why no-thinking wins

No-thinking wins for both **semantic** and **operational** reasons.

### Semantic reasons
- higher group1 mean
- lower group2 final mean
- slightly lower group3 substance
- much stronger group separation

### Operational reasons
- much faster
- no errors
- no invalid non-thinking rows
- no giant reasoning overhead
- no catastrophic JSON failure on a gold example

At this point, no-thinking is the right mode to carry forward into the next notebook for prompt tuning.

---

## What this benchmark established

This benchmarking pass settled the mode-choice question well enough to move on.

It established that:

1. the corrected OpenRouter no-thinking request path is working
2. the saved no-thinking outputs are genuinely non-thinking
3. no-thinking is the better mode for this benchmark
4. the next bottleneck is no longer mode choice

The next bottleneck is now:

- refining the meaning prompt further
- and improving group3 quality so that the benchmark becomes harsher and cleaner

