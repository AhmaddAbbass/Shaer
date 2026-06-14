# Description Regeneration Pipeline

Last updated: 2026-04-11 UTC.

This folder contains the current active pipeline for rebuilding the poetry SFT dataset with better conditioning text.

## 2026-04-11 Downstream Status

This pipeline is complete enough for the current repo direction.

Downstream outcomes now in use:

- rebuilt dataset:
  - `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`
- split SFT dataset:
  - `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`
- finished main SFT baseline trained on that split dataset:
  - `Shaer-AI/Shaer-adapters`
- derived GRPO preprocess dataset built from that split dataset:
  - `Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1`

So this folder now serves two purposes:

- preserve how `enhanced_description` was created
- document the upstream lineage for the finished SFT baseline and the later GRPO preprocess dataset

For paper-writing purposes, this folder is also the lineage anchor for why later SFT and GRPO experiments moved off the old `description` field and onto `enhanced_description`.

## Purpose

The old dataset field `description` is preserved, but it is no longer trusted as the conditioning source for the next SFT cycle.

The current goal is:

1. generate a better `enhanced_description` for each poem
2. rebuild the SFT prompt fields from `enhanced_description`
3. publish the rebuilt dataset
4. retrain SFT on the rebuilt dataset

That original goal is now achieved at repo level.

## Source And Target Datasets

Source dataset:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

Target dataset repo:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

Current downstream split repo:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Current downstream GRPO preprocess repo:

- `Shaer-AI/ashaar-enhanced-desc-baseform-final-sft-lte20-min500-splits-grpo-meter-count-v1`

## Filtering Rules

The preprocessing rules are:

1. validate `poem verses` as a non-empty even-length list of non-empty strings
2. compute:
   - `requested_bayts = len(poem verses) // 2`
3. drop malformed odd-shatr rows
4. keep only rows with:
   - `requested_bayts <= 20`
5. compute post-filter base-meter counts
6. drop base meters with count `< 500`

Current outcome on the source dataset:

- source rows:
  - `129610`
- after valid-row filter:
  - `127722`
- after `<=20` bayt filter:
  - `116167`
- final rows after meter cutoff:
  - `116032`
- dropped odd-shatr rows:
  - `1888`
- dropped `>20` rows:
  - `11555`
- dropped meter:
  - `المتدارك`

## Current Prompt Versions

Description prompt version:

- `pure_description_v8_poem_only_antidrift`

Final rebuilt SFT prompt version:

- `final_sft_meter_emphasis_v1`

Both are defined in:

- [prompt_contracts.py](prompt_contracts.py)

## Exact Description Prompt Sent To Qwen

System prompt:

```text
أنت تكتب وصفًا عربيًا لقصيدة.

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
- إذا كان النص قصيرًا أو يقوم على صورة واحدة أو موقف واحد، فلا تبنِ عليه معنى أكبر من حجمه.
- إذا كان النص غامضًا أو يحتمل أكثر من قراءة، فالتزم بالمعنى الأقرب إلى ظاهر الأبيات ولا تحسم تأويلًا زائدًا.
- لا تخترع مشاهد أو دوافع أو علاقات سببية غير ظاهرة.
- لا تجعل الوصف شرحًا بيتًا بيتًا.
- في النصوص القصيرة أو الغريبة، صف ما يظهر في النص أكثر مما تفسره.
- لا تذكر البحر أو القافية أو عدد الأبيات.
- لا تستخدم صيغ الطلب أو الأمر مثل: أريد، أكتب، اذكر، ركز، اجعل، يجب أن.
- لا تستبدل الغريب أو الخاص في النص بتعبير عام باهت.
- اكتب فقرة واحدة طبيعية مكتملة.
```

User prompt:

```text
الأبيات:
[كل الأشطار هنا]

اكتب وصفًا واحدًا يلتقط أفكار القصيدة ومعناها وصورها ونبرتها العامة.
```

## Final SFT Prompt Contract

The rebuilt SFT prompt uses `enhanced_description` instead of the old `description`.

System prompt:

```text
أنت شاعر عربي تكتب الشعر العمودي الكلاسيكي.
التزم بالبحر المحدد في كل شطر، واستلهم من الموضوع دون نقله حرفياً.
أخرج الأبيات فقط دون مقدمة أو تعليق.
التزم التزاماً صارماً بالبحر المطلوب، ولا تخرج عنه.
```

User template:

```text
البحر الأساسي: {base_meter}
الصيغة: {form}
اسم البحر المطلوب: {meter_label}
البحر المطلوب هو: {meter_label}
الموضوع: {enhanced_description}

اكتب {num_lines} شطراً ملتزماً بصيغة {form} من بحر {base_meter}، والبحر المطلوب هو {meter_label}، دون أي شرح إضافي.
```

`meter_label` rule:

- if `form == "تام"` then `meter_label = base_meter`
- otherwise `meter_label = f"{form} {base_meter}"`

## Scripts

Main scripts:

- [final_sft_dataset.py](final_sft_dataset.py)
  - prepare / verify / merge_publish
- [regenerate_descriptions.py](regenerate_descriptions.py)
  - sample-mode experiments and worker-mode regeneration
- [launch_final_sft_workers.sh](launch_final_sft_workers.sh)
  - detached launcher for many workers
- [worker_status.py](worker_status.py)
  - local progress summary and duplicate check
- [run_detached.sh](run_detached.sh)
  - detached single-process runner

## Current Prepared Run

Prepared run:

- run name:
  - `final_sft_regen_v1_20260407`
- local run root:
  - `/root/workspace/Shaer/description_generation/outputs/final_sft_regen_v1_20260407`
- manifest:
  - [workers_manifest.json](outputs/final_sft_regen_v1_20260407/workers_manifest.json)
- current worker count:
  - `10`

Smoke-test status:

- shard verification: passed
- worker independence: passed
- Hub progress syncing: passed
- smoke rows completed: `10`

Important:

- this prepared run is only for `10` workers
- if you want `20` workers, do not reuse this manifest
- instead, run `prepare` again with a new run name and `--num-workers 20`

## Standard Workflow

### 1. Prepare a run

Example with `20` workers:

```bash
/root/workspace/Shaer/sft/.venv/bin/python \
  /root/workspace/Shaer/description_generation/final_sft_dataset.py prepare \
  --run-name final_sft_regen_v1_20260407_w20 \
  --num-workers 20
```

### 2. Verify shard integrity

```bash
/root/workspace/Shaer/sft/.venv/bin/python \
  /root/workspace/Shaer/description_generation/final_sft_dataset.py verify \
  --workers-manifest /root/workspace/Shaer/description_generation/outputs/final_sft_regen_v1_20260407_w20/workers_manifest.json
```

### 3. Launch workers

```bash
WORKER_COUNT=20 \
/root/workspace/Shaer/description_generation/launch_final_sft_workers.sh \
final_sft_regen_v1_20260407_w20
```

### 4. Check progress

```bash
python3 /root/workspace/Shaer/description_generation/worker_status.py \
  --workers-manifest /root/workspace/Shaer/description_generation/outputs/final_sft_regen_v1_20260407_w20/workers_manifest.json
```

### 5. Merge and publish the final dataset

```bash
/root/workspace/Shaer/sft/.venv/bin/python \
  /root/workspace/Shaer/description_generation/final_sft_dataset.py merge_publish \
  --workers-manifest /root/workspace/Shaer/description_generation/outputs/final_sft_regen_v1_20260407_w20/workers_manifest.json
```

## Resume And Reliability Behavior

Worker behavior:

- one shared script is used for all workers
- each worker gets a disjoint shard from the manifest
- each worker writes only to its own local output directory
- progress is uploaded to the target HF dataset repo under:
  - `progress/<run_name>/workers/worker_XX/`
- resume mode skips already-completed `source_index` rows for that worker
- malformed JSON, empty descriptions, raw JSON leakage, and truncated outputs are retried

## What Happens At Merge Time

## Current Repo-Level Interpretation

This description-regeneration work is no longer the next execution task.

It is now completed upstream lineage for:

- the finished SFT paper baseline
- the pushed meter+count preprocess dataset
- the next intended GRPO restart

The merge step:

1. joins worker outputs back onto the staged filtered dataset by `source_index`
2. keeps the original `description`
3. adds `enhanced_description`
4. rebuilds:
   - `sft_prompt`
   - `sft_completion`
   - `sft_full_text`
   - `sft_num_lines`
   - `sft_total_tokens`
5. re-applies the `<= 2048` token cap
6. writes and uploads the final dataset split plus a new README card

## After This Folder's Work Finishes

The next work item is not GRPO.

The next work item is:

1. retrain SFT on the rebuilt dataset
2. evaluate the new SFT baseline
3. only then revisit GRPO
