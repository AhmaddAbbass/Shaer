# Shaer: Paper-Ready Project Dossier

Last prepared: 2026-05-27  
Target venue: ArabicNLP 2026  
Authors: Ahmad Abbas, Tamara Fakih, Nour Fakih

This document is not the final paper. It is the detailed technical and writing dossier that should let a paper-writing collaborator draft the ArabicNLP submission quickly and accurately. It describes Shaer as a complete project for advancing Arabic classical poetry generation: a curated poetry instruction dataset, a meter- and form-conditioned generation setup, and QLoRA adapters trained on top of Yehia-7B.

Important writing constraint: keep the paper story focused on the completed SFT project only.

## 1. One-Paragraph Project Summary

Shaer is a classical Arabic poetry generation project that adapts `Navid-AI/Yehia-7B-preview` to produce العمودي classical Arabic poetry conditioned on meter, form, requested length, and topic. The project starts from an Ashaar-derived Arabic poetry corpus and converts it into a supervised instruction-tuning dataset with explicit metrical controls, cleaned poem text, regenerated semantic topic descriptions, deterministic train/evaluation/test splits, and a weighted training sampler that reduces dominance by frequent meter-form-length groups. The final model artifact is a QLoRA adapter published as `Shaer-AI/Shaer-adapters`, trained for two epochs over the final split dataset. The paper should present Shaer as a dataset-plus-model contribution for controlled Arabic poetry generation, with planned evaluation along four axes: meter adherence, meaning/topic adherence, fluency/coherence, and poeticness.

## 2. Paper Claim To Support

The intended claim is:

> Shaer advances Arabic classical poetry generation by combining Arabic-specific model selection, poetry-specific dataset curation, meter/form/length-aware instruction formatting, and parameter-efficient fine-tuning of Yehia-7B, producing a controllable poetry generation system intended to achieve state-of-the-art performance on meter adherence, meaning preservation, fluency, coherence, and poeticness.

Use this carefully in the final paper:

- Before final human/automatic evaluation is complete, write “we aim to show” or “we evaluate whether”.
- After metrics are filled, the abstract can say “Shaer achieves state-of-the-art” only if the comparison table supports it.
- If no strong public baseline exists for the exact task, phrase the claim as “to our knowledge, Shaer is the first released Yehia-based meter-conditioned classical Arabic poetry generation system with public SFT data, adapters, and training artifacts.”

## 3. ArabicNLP Paper Structure

ArabicNLP 2026 follows the ACL-style paper culture. The CFP lists June 18, 2026 as the abstract submission due date and June 25, 2026 as the direct paper submission due date. The paper should therefore be written like a standard empirical NLP paper, with clear claims, reproducible artifacts, and a focused evaluation section.

Recommended structure:

1. Abstract
2. Introduction
3. Related Work
4. Task Definition
5. Dataset Construction
6. Model Selection
7. Method: Prompt Format and SFT Training
8. Experimental Setup
9. Results
10. Analysis and Error Analysis
11. Limitations
12. Ethics and Data Statement
13. Conclusion

ArabicNLP-specific information that should appear:

- What variety of Arabic is targeted: classical/literary Arabic poetry, not dialectal conversation.
- What poetic form is targeted: الشعر العمودي, generated as ordered شطر lines.
- How meter is represented: `base_meter` and `form`.
- Which meters/forms are included and which rare meters/forms are excluded.
- What preprocessing is Arabic-specific: tatweel cleanup, shatr/bayt validation, meter-form normalization.
- What evaluation is Arabic-poetry-specific: meter adherence, line-count adherence, topic/meaning fit, fluency/coherence, poeticness.
- What artifacts are released: final datasets, split dataset, adapter repo, plots, metrics, and model card.

## 4. Suggested Abstract Draft

Use this as a starting point, not final text:

> Classical Arabic poetry generation requires control over semantic content, poetic fluency, and strict prosodic form. We present Shaer, a dataset and model-adaptation project for meter-conditioned Arabic poetry generation. Shaer fine-tunes Yehia-7B, a strong Arabic model selected based on Arabic language benchmarking and internal poetry-generation comparison, using a curated Ashaar-derived instruction dataset. Starting from raw poetry records, we normalize meter and form labels, remove malformed or overly long examples, regenerate semantic conditioning descriptions from poem text, clean Arabic orthographic artifacts such as tatweel, and construct deterministic train/evaluation/test splits stratified by meter, form, and poem length. We train QLoRA adapters over all linear layers of Yehia-7B using completion-only supervised fine-tuning. The resulting system generates classical Arabic poem lines conditioned on target meter, form, requested length, and topic. We release the curated datasets and adapters, and evaluate Shaer on meter adherence, meaning fit, fluency/coherence, and poeticness.

After results are complete, add one sentence with the strongest metric:

> Shaer achieves X on meter adherence, Y on meaning fit, Z on fluency/coherence, and outperforms [baselines] by [margin].

## 5. Contributions

The paper should state the contributions as:

1. A curated Arabic classical poetry SFT dataset derived from Ashaar, with explicit controls for meter, form, requested length, and semantic topic.
2. A reproducible dataset construction pipeline that filters malformed poems, controls poem length, removes undersupported meter groups, cleans tatweel artifacts, and builds deterministic stratified splits.
3. A Yehia-7B QLoRA adapter for controlled Arabic poetry generation, trained with all-linear LoRA and completion-only SFT.
4. A public artifact release: final unsplit dataset, final split dataset, adapter weights, model card, training metrics, and plots.
5. A planned evaluation protocol for Arabic poetry generation covering meter, meaning, fluency/coherence, and poeticness.

Avoid claiming “first” unless the related-work review is made exhaustive. A safer claim is “one of the first released systems combining public meter-conditioned Arabic poetry SFT data with Yehia-based adapters and detailed training artifacts.”

## 6. Related Work

### 6.1 Arabic Poetry Analysis and Generation

The closest poetry-specific reference is Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches. Ashaar introduces a framework for Arabic poetry analysis and generation, including datasets and models for meter, theme, era classification, diacritization, Arudi-style extraction, and conditional poetry generation using a character-based GPT model. Shaer differs from Ashaar in the generation model and conditioning setup: instead of training a character-level generator, Shaer adapts a modern Arabic instruction model using QLoRA and conditions generation through a structured instruction prompt containing meter, form, length, and topic.

AraPoemBERT is relevant as an Arabic poetry understanding model rather than a generation model. It pretrains a BERT-style model exclusively on Arabic poetry and reports strong results on downstream poetry analysis tasks such as meter classification, sub-meter classification, rhyme classification, sentiment, and poet gender classification. This is useful in the related work because it shows that poetry-specific pretraining helps analysis tasks, but Shaer targets controllable generation. AraPoemBERT may also be mentioned as a potential evaluator or auxiliary analysis model, not as a direct generation baseline.

The small public Hugging Face models `akhooli/gpt2-small-arabic-poetry` and `elgeish/gpt2-medium-arabic-poetry` appear to be GPT-2-style Arabic poetry generators, but no clear direct paper was found for these specific checkpoints. If used as baselines, cite the model cards and cite AraGPT2 as the related Arabic GPT-2 generation paper. AraGPT2 introduced Arabic GPT-2 language models trained from scratch for Arabic generation and evaluated them on Arabic text generation tasks.

The model `arbml/Ashaar_model` is directly tied to the Ashaar project and should be treated as a poetry-generation or continuation baseline if it can be run reliably. If included experimentally, the paper should clearly state whether it supports the same controls as Shaer. If it does not support meter/form/topic/length in the same format, compare it qualitatively or under a simplified prompt setup.

### 6.2 Arabic-Centric Large Language Models

Fanar is an Arabic-centric multimodal generative AI platform from QCRI. The Fanar paper describes Arabic-focused language, speech, and image systems, including Fanar Star and Fanar Prime. Fanar is relevant for two reasons: it is part of the broader Arabic LLM ecosystem, and Hala-9B is based on `QCRI/Fanar-1-9B-Instruct`.

Hala is an Arabic-centric instruction and translation model family. The Hala technical report describes training models at 350M, 700M, 1.2B, and 9B parameters, using translated instruction data and model merging to balance Arabic specialization with base-model capabilities. The public `hammh0a/Hala-9B` model is a Gemma2-style model based on `QCRI/Fanar-1-9B-Instruct`, with a non-commercial license. In Shaer, Hala is relevant as a strong Arabic model candidate and a planned comparison point. The paper should include an empty subsection for the internal Yehia-vs-Hala poetry comparison until the actual examples and judgments are recovered.

Yehia-7B is the chosen base model for Shaer. The public model card describes `Navid-AI/Yehia-7B-preview` as a 7B Arabic/English conversational model based on `humain-ai/ALLaM-7B-Instruct-preview`, with Apache-2.0 licensing. It was selected because it is a strong Arabic model according to Arabic language benchmarks and because internal poetry-generation testing favored Yehia over Hala for the desired classical poetry behavior. The final paper should cite AraLingBench for the benchmark motivation and include the internal Hala-vs-Yehia comparison once it is reconstructed.

### 6.3 Arabic Benchmarks and Model Selection

AraLingBench is a human-annotated benchmark for evaluating Arabic linguistic capabilities of LLMs. It evaluates Arabic linguistic competence across several categories and includes Arabic-specific models such as Yehia, Hala, Fanar, JAIS, and ALLaM. For Shaer, AraLingBench motivates choosing Yehia as a strong Arabic base model before domain adaptation.

Do not overstate AraLingBench as a poetry benchmark. It is a linguistic Arabic benchmark, not a classical-poetry generation benchmark. The correct paper wording is:

> We selected Yehia-7B as the base model because it showed strong Arabic linguistic competence in AraLingBench and because our internal poetry-generation comparison indicated better suitability for classical Arabic verse than Hala-9B.

### 6.4 Multilingual and Qwen-Based Models

Qwen3 is relevant only indirectly. The Qwen3 technical report describes the Qwen family’s multilingual capabilities and broad benchmark performance. In Shaer, a Qwen-family model was used during data construction to generate semantic topic descriptions from poem text. This use should be described as dataset curation, not as the final generator. The final generator is Yehia-7B plus Shaer adapters.

The model `Cyb3RQ/arabic-poetry-qwen3-8b-GGUF` appears to be a Qwen3-8B Arabic poetry fine-tune, but no formal direct paper was identified. It can be mentioned as an online public poetry-oriented model if used as a baseline, but the final paper should avoid making unsupported claims about its training or evaluation.

## 7. Task Definition

Shaer addresses controlled classical Arabic poetry generation.

Input:

- `base_meter`: the main Arabic meter family.
- `form`: the meter form, such as تام, مجزوء, مخلع, or أحذ.
- requested number of شطر lines.
- a semantic topic description.

Output:

- poem text only.
- one generated شطر per line.
- no prose preface.
- no explanation.
- no metadata.

The task is stricter than generic Arabic text generation because a valid output must satisfy both semantic and formal constraints:

- It should match the requested topic.
- It should be fluent literary Arabic.
- It should be coherent across lines.
- It should sound poetic.
- It should obey the requested meter/form.
- It should produce the requested number of lines.

## 8. Why Yehia-7B

The base model is `Navid-AI/Yehia-7B-preview`.

Paper-facing rationale:

1. Yehia is an Arabic/English 7B instruction model based on ALLaM-7B.
2. It has an Apache-2.0 license, which is compatible with releasing adapters.
3. AraLingBench indicates that Yehia is one of the strong Arabic models in the tested model set.
4. Internal poetry-oriented testing favored Yehia over Hala for classical Arabic generation.
5. Yehia is small enough for efficient QLoRA adaptation while still having strong Arabic linguistic priors.

Empty section to fill later:

### Internal Yehia-vs-Hala Poetry Comparison

To be filled by the authors.

Minimum information needed:

- prompt set used for comparison.
- number of generated poems per model.
- decoding settings.
- human evaluation criteria.
- examples where Yehia outperformed Hala.
- whether Hala underperformed more on meter, coherence, diction, instruction following, or poeticness.

Suggested final-paper wording after filling:

> In addition to benchmark-based model selection, we conducted an internal comparison between Yehia-7B and Hala-9B on classical Arabic poetry prompts. Human inspection favored Yehia for [meter adherence / diction / coherence / poeticness], so we selected Yehia as the base model for domain adaptation.

## 9. Dataset Lineage

The clean lineage for the paper is:

```text
Original Ashaar corpus
  -> Shaer-AI/ashaar-v1-base-form
  -> Shaer-AI/ashaar-v1-base-form-with-descriptions
  -> Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed
  -> Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500
  -> Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits
  -> Shaer-AI/Shaer-adapters
```

Use the public dataset IDs exactly because the final dataset names include “enhanced-descriptions”. In prose, however, describe the step as “regenerating richer semantic descriptions” or “semantic description regeneration” rather than building the whole paper title around that phrase.

There is also an older visible prehistory branch:

```text
Original Ashaar corpus
  -> Shaer-AI/ashaar-preprocessed
  -> Shaer-AI/ashaar-preprocessed-desc
  -> Shaer-AI/ashaar-preprocessed-postworkers
  -> Shaer-AI/ashaar-full-desc
  -> Shaer-AI/ashaar-full-desc-preprocessed
```

This older branch is useful as repository history, but the paper should focus on the direct final chain unless reviewers ask for every intermediate artifact.

## 10. Dataset Construction, Step By Step

### 10.1 Original Ashaar-Derived Corpus

The project begins from an Ashaar-derived corpus of Arabic poems. The raw/preprocessed records contain poem text, poem metadata, poet metadata, meter labels, themes, URLs, and poem descriptions. The direct raw scraping code is not part of the current paper story; the paper should simply state that the dataset is derived from Ashaar and then describe the reproducible transformations used for Shaer.

### 10.2 Base Meter and Form Normalization

Dataset:

- `Shaer-AI/ashaar-v1-base-form`

This stage converts raw poem records into a controlled meter/form dataset. It introduces:

- `base_meter`
- `form`

Supported base meters:

- الطويل
- الكامل
- البسيط
- الوافر
- الخفيف
- السريع
- الرجز
- الرمل
- المتقارب
- المجتث
- المنسرح
- المديد
- الهزج
- المتدارك

Excluded rare meters at this stage:

- المضارع
- المقتضب

Supported forms:

- تام
- مجزوء
- مخلع only for البسيط
- أحذ only for الكامل

Rare or inconsistent forms were removed so that the model sees stable control labels. The dataset card reports a validation split rule of:

```text
take = min(50, ceil(5% of each (meter, form) group))
```

Paper explanation:

> We normalize heterogeneous meter labels into a canonical control representation consisting of base meter and form. Rare meters and unstable form labels are removed to avoid conditioning the model on categories with insufficient support.

### 10.3 Description-Joined Base-Form Dataset

Dataset:

- `Shaer-AI/ashaar-v1-base-form-with-descriptions`

Rows:

- 143,022

Main columns:

- `id`
- `poem verses`
- `base_meter`
- `form`
- `poem theme`
- `poem meter`
- `poem url`
- `description`

This stage joins the controlled meter/form representation with a natural-language description field. The description gives the topic/meaning conditioning signal that later becomes part of the SFT prompt.

### 10.4 Initial SFT Formatting and Token Trimming

Dataset:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

Source:

- `Shaer-AI/ashaar-v1-base-form-with-descriptions`

Row counts:

| Stage | Rows |
|---|---:|
| source rows | 143,022 |
| after row processing | 129,688 |
| removed by token filter | 78 |
| final rows | 129,610 |

Added SFT columns:

- `sft_prompt`
- `sft_completion`
- `sft_full_text`
- `sft_num_lines`
- `sft_total_tokens`

Important behavior:

- invalid or empty poem rows were removed.
- poem verses were converted into newline-separated completions.
- token length was computed using the Yehia tokenizer.
- rows with `sft_total_tokens > 2048` were removed.

Paper explanation:

> We first convert poem records into an instruction-tuning format by pairing a structured Arabic prompt with the original poem text as the target completion. This step removes malformed records and examples that exceed the token budget.

### 10.5 Semantic Description Regeneration

Dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

Source:

- `Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed`

This is the final unsplit SFT-ready dataset. The key change is that the old description field is preserved, but a new topic description is generated from the poem text itself. The SFT prompt is rebuilt using the regenerated semantic description.

Use this wording in the paper:

> To improve the semantic conditioning signal, we regenerated poem-level topic descriptions from the poem text. The regenerated description is intended to summarize the poem’s theme, imagery, and tone without copying lines, inventing unsupported meanings, mentioning meter/rhyme, or turning the description into an instruction.

Avoid making the title of the paper about “enhanced descriptions”, but do include this step in the dataset section because it is central to the curation pipeline.

Generation details:

- provider: OpenRouter-compatible API.
- model configured in the script: `qwen/qwen3.5-35b-a3b`.
- temperature: 0.0.
- output format: JSON.
- maximum output tokens: 300.
- reasoning disabled.
- fallbacks disabled.
- expected key: `new_description`.
- malformed, empty, raw-JSON-leaking, or nonconforming outputs were retried.

The prompt instructed the model to write a single Arabic paragraph that:

- begins with “القصيدة تتحدث عن...”
- captures the poem’s ideas, meaning, imagery, and tone.
- does not copy the poem.
- does not invent unsupported meanings.
- does not mention meter, rhyme, or number of lines.
- does not use request/instruction phrases.
- does not explain the poem line by line.
- remains conservative when the poem is short, ambiguous, or image-based.

Worker setup:

- final run: `final_sft_regen_v1_20260407_w24`
- workers: 24
- assigned rows: 116,032
- original worker successes: 115,992
- original worker errors: 40
- retry artifacts used:
  - `final_sft_regen_v1_20260407_w24_retry40_w4`
  - `final_sft_regen_v1_20260407_w24_retry15_plaintext`
- final merged rows: 116,032
- rows removed by final token cap: 0

### 10.6 Final Filtering Rules

The final unsplit dataset is built using:

1. valid poem verses only.
2. even number of shatr lines.
3. `requested_bayts = len(poem verses) // 2`.
4. keep only `requested_bayts <= 20`.
5. compute base-meter counts after the length/even-shatr filters.
6. keep meters with at least 500 examples.
7. rebuild SFT fields from the regenerated semantic description.

Row-count transition:

| Step | Rows | Drop |
|---|---:|---:|
| old trimmed source | 129,610 | - |
| after valid even-shatr filter | 127,722 | 1,888 |
| after `requested_bayts <= 20` | 116,167 | 11,555 |
| after meter support `>= 500` | 116,032 | 135 |

The 135-row final drop corresponds to المتدارك after the earlier filters, because it fell below the 500-example threshold.

### 10.7 Tatweel Cleanup

Arabic tatweel/kashida artifacts were cleaned after final row construction.

Rules:

- remove internal tatweel from poem lines.
- preserve only edge tatweel at shatr boundaries when present.
- remove tatweel from old descriptions.
- remove tatweel from regenerated semantic descriptions.
- rebuild prompt fields after cleanup.

Cleanup summary:

| Metric | Value |
|---|---:|
| rows processed | 116,032 |
| tatweel characters removed | 5,461,645 |
| rows with poem tatweel before | 66,361 |
| rows with internal poem tatweel before | 65,965 |
| rows changed | 65,965 |
| rows with poem tatweel after | 514 |
| validation failures | 0 |

Paper explanation:

> Tatweel was treated as an orthographic artifact that can distort tokenization and encourage spurious copying. We therefore removed internal tatweel while preserving boundary cases that survived validation.

## 11. Final Prompt Format

The final SFT prompt is an Arabic instruction prompt. It conditions the model on:

- base meter.
- form.
- requested number of shatr lines.
- semantic topic description.

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
عدد الأشطر المطلوب: {num_lines}
الموضوع: {description}

اكتب {num_lines} أشطارًا ملتزمة بصيغة {form} من بحر {base_meter} دون أي شرح إضافي.
```

Training target:

```text
{poem_shatr_1}
{poem_shatr_2}
...
```

Paper explanation:

> Before SFT, we performed iterative prompt engineering to choose an instruction format that made the base Yehia model produce more structurally suitable poems. This matters because SFT starts from the base model’s response distribution: a prompt that already elicits closer behavior gives the optimizer a better starting point than a prompt that elicits prose, explanations, or weak control following.

Do not describe discarded prompt variants. The paper only needs the selected final prompt and the rationale.

## 12. Final Train/Eval/Test Split

Dataset used for training:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Source:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500`

Split:

- train: 94%
- eval: 3%
- test: 3%
- seed: 42

Counts:

| Split | Rows |
|---|---:|
| train | 109,070 |
| eval | 3,481 |
| test | 3,481 |
| total | 116,032 |

Stratification key:

```text
base_meter || form || length_bucket
```

Length buckets:

- `1-3`
- `4-6`
- `7-10`
- `11-20`

All final rows stayed at the finest stratification level:

| Split group level | Rows |
|---|---:|
| `base_meter_form_length_bucket` | 116,032 |

Paper explanation:

> We split the dataset deterministically with stratification over the joint meter, form, and length-bucket label. This avoids evaluation splits that accidentally omit rare form-length combinations and makes the held-out sets reflect the same control structure as training.

## 13. Split Distributions

Base-meter counts:

| Meter | Train | Eval | Test |
|---|---:|---:|---:|
| البسيط | 16,563 | 529 | 529 |
| الخفيف | 9,263 | 295 | 295 |
| الرجز | 4,311 | 137 | 137 |
| الرمل | 4,316 | 138 | 138 |
| السريع | 6,758 | 216 | 216 |
| الطويل | 26,346 | 841 | 841 |
| الكامل | 20,375 | 649 | 649 |
| المتقارب | 4,877 | 156 | 156 |
| المجتث | 1,938 | 61 | 61 |
| المديد | 504 | 17 | 17 |
| المنسرح | 2,579 | 84 | 84 |
| الهزج | 573 | 18 | 18 |
| الوافر | 10,667 | 340 | 340 |

Form counts:

| Form | Train | Eval | Test |
|---|---:|---:|---:|
| أحذ | 351 | 11 | 11 |
| تام | 104,224 | 3,327 | 3,327 |
| مجزوء | 3,882 | 124 | 124 |
| مخلع | 613 | 19 | 19 |

Length-bucket counts:

| Bucket | Train | Eval | Test |
|---|---:|---:|---:|
| `1-3` | 52,202 | 1,668 | 1,668 |
| `4-6` | 25,266 | 805 | 805 |
| `7-10` | 14,824 | 473 | 473 |
| `11-20` | 16,778 | 535 | 535 |

## 14. Training Sampling Method

Training used a weighted sampler on the train split only. Evaluation and test remained unweighted.

Group key:

```text
base_meter || form || length_bucket
```

Weight formula:

```text
raw_weight(g) = (1 / count(g)) ** 0.6
normalized_weight(g) = raw_weight(g) / mean(raw_weight over groups)
```

Rationale:

> Classical Arabic meters are highly imbalanced in the source data. A fully natural sampler would overrepresent common groups such as تام الطويل and تام الكامل. A fully balanced sampler would over-amplify rare groups and risk instability. The exponent 0.6 is a middle ground: it softens dominance by frequent groups while preserving the empirical distribution.

Paper wording:

> We apply train-only inverse-frequency smoothing over the joint meter-form-length group. The sampler uses replacement and assigns each training example a weight proportional to the smoothed inverse count of its group. Held-out evaluation and test splits are kept natural to measure real distribution performance.

## 15. SFT Training Configuration

Final adapter repo:

- `Shaer-AI/Shaer-adapters`

Run:

- `train_20260407_231929`

Base model:

- `Navid-AI/Yehia-7B-preview`

Dataset:

- `Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits`

Training runtime:

- 45,102.4652 seconds.
- approximately 12.53 hours.

Training objective:

- causal language modeling.
- completion-only loss.
- prompt tokens masked with `-100`.
- only poem completion tokens contribute to loss.

Sequence setup:

- max sequence length: 1,024.
- train packing: enabled.
- truncation mode: `keep_end`.

Quantization:

- QLoRA.
- 4-bit loading.
- NF4 quantization.
- double quantization: true.
- compute dtype: bfloat16.
- model dtype: bfloat16.

LoRA:

| Parameter | Value |
|---|---|
| PEFT type | LoRA |
| task type | causal LM |
| target modules | all linear layers |
| materialized modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| rank `r` | 64 |
| alpha | 128 |
| dropout | 0.05 |
| bias | none |
| RS-LoRA | true |

Optimization:

| Parameter | Value |
|---|---|
| optimizer | `paged_adamw_8bit` |
| learning rate | `8e-5` |
| scheduler | cosine |
| warmup ratio | 0.03 |
| weight decay | 0.0 |
| max grad norm | 1.0 |
| train batch size per device | 1 |
| eval batch size per device | 1 |
| gradient accumulation | 16 |
| epochs | 2 |
| global steps | 4,236 |
| eval interval | 200 steps |
| save interval | 200 steps |
| logging interval | 10 steps |
| save total limit | 3 |
| seed | 42 |

Paper explanation:

> We use QLoRA to adapt Yehia-7B efficiently while preserving the base model. Unlike attention-only LoRA setups, we target all major linear projections in the attention and feed-forward blocks, allowing the adapter to modify both token interaction and lexical/style transformation behavior.

## 16. Training Outcomes Available Now

Available metrics:

| Metric | Value |
|---|---:|
| initial eval loss | 3.080312 |
| best eval loss | 2.207448 |
| best eval step | 3,000 |
| final eval loss | 2.207448 |
| final test loss | 2.193206 |
| train loss | 1.620038 |
| global steps | 4,236 |
| training runtime | 12.53 hours |

Probe metrics:

| Metric | Step | Value |
|---|---:|---:|
| initial probe meter mean | 0 | 0.022648 |
| initial count adherence | 0 | 0.692308 |
| best probe meter mean | 2,800 | 0.604216 |
| best count adherence | 1,400 | 1.000000 |
| final probe meter mean | 4,236 | 0.508705 |
| final count adherence | 4,236 | 1.000000 |

Interpretation:

- Cross-entropy improved strongly from the initial evaluation loss to the best evaluation loss.
- Generation-side line-count adherence reached 1.0 on the probe set.
- Meter adherence improved substantially but varied strongly by meter.
- The best loss checkpoint and the best meter-probe checkpoint were not identical.

Best checkpoints:

- best eval checkpoint: step 3,000.
- best eval loss: 2.207448.
- best meter-probe checkpoint: step 2,800.
- final/latest checkpoint: step 4,236.

HF adapter paths:

- best adapter: `adapters/fresh_sft/train/best`
- latest adapter: `adapters/fresh_sft/train/latest`

Current retained HF checkpoints:

- checkpoint 4,000.
- checkpoint 4,200.
- checkpoint 4,236.

## 17. Per-Meter Probe Snapshot

Final probe meter means:

| Meter | Score |
|---|---:|
| الطويل | 0.990793 |
| الكامل | 0.963871 |
| المنسرح | 0.824677 |
| الوافر | 0.778932 |
| الخفيف | 0.578323 |
| البسيط | 0.519473 |
| السريع | 0.510735 |
| المجتث | 0.509197 |
| المتقارب | 0.501623 |
| الرمل | 0.350412 |
| الرجز | 0.055653 |
| الهزج | 0.023443 |
| المديد | 0.006034 |

Paper interpretation:

> The probe results suggest that SFT strongly improves structural control overall, but meter adherence remains uneven across meters. High-frequency or rhythmically easier meters such as الطويل and الكامل show strong probe performance, while some lower-resource or difficult meters remain weak. This motivates reporting per-meter results rather than only aggregate scores.

Do not use the probe alone as the final SOTA evidence. It is a useful training monitor, but the paper should include a stronger final evaluation.

## 18. Plots and Artifacts

The adapter repo includes training reports and plots:

- `reports/fresh_sft/train_20260407_231929/summary.md`
- `reports/fresh_sft/train_20260407_231929/key_metrics.json`
- `reports/fresh_sft/train_20260407_231929/run_summary.json`
- `reports/fresh_sft/train_20260407_231929/config_snapshot.json`
- `reports/fresh_sft/train_20260407_231929/effective_runtime_config.json`
- `reports/fresh_sft/train_20260407_231929/dataset_summary.json`
- `reports/fresh_sft/train_20260407_231929/split_summary.json`
- `reports/fresh_sft/train_20260407_231929/metrics.csv`
- `reports/fresh_sft/train_20260407_231929/probe_metrics.csv`
- `reports/fresh_sft/train_20260407_231929/final_probe_per_meter.csv`
- `reports/fresh_sft/train_20260407_231929/plots/loss_panels.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_panels.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_meter_heatmap.png`
- `reports/fresh_sft/train_20260407_231929/plots/probe_meter_small_multiples.png`

Use the plots as:

- Figure 1: dataset construction pipeline.
- Figure 2: training/eval loss curve.
- Figure 3: probe meter and count adherence over training.
- Figure 4: per-meter probe heatmap or small multiples.

## 19. Final Evaluation Plan

The final paper should evaluate:

1. Meter adherence.
2. Meaning/topic adherence.
3. Fluency and coherence.
4. Poeticness.
5. Length adherence.

### 19.1 Suggested Baselines

Strongest useful baselines:

- base Yehia-7B with the final prompt.
- Hala-9B with the final prompt.
- `arbml/Ashaar_model`, if runnable.
- GPT-2 Arabic poetry models, if runnable.
- a Qwen-based Arabic poetry model, if runnable.

If a baseline does not support exact meter/form/topic/length conditioning, state the limitation clearly.

### 19.2 Suggested Automatic Metrics

Meter:

- use the project’s meter scorer/classifier.
- report aggregate and per-meter scores.
- report exact line-count adherence separately.

Meaning/topic:

- human evaluation is preferred.
- model-assisted judging can be used if the prompt and validation are reported.
- include examples of high and low meaning fit.

Fluency/coherence:

- human Likert ratings.
- judge prompt rating if human budget is limited.

Poeticness:

- human expert or Arabic-fluent annotator rating.
- criteria should include diction, imagery, rhythm impression, non-prosaic quality, and avoidance of templatic output.

### 19.3 Suggested Human Evaluation Table

Use a 1-5 scale:

| Model | Meter | Meaning | Fluency/Coherence | Poeticness | Overall |
|---|---:|---:|---:|---:|---:|
| Yehia-7B base | TBD | TBD | TBD | TBD | TBD |
| Hala-9B | TBD | TBD | TBD | TBD | TBD |
| Ashaar baseline | TBD | TBD | TBD | TBD | TBD |
| Shaer | TBD | TBD | TBD | TBD | TBD |

Annotator instructions:

- Meter: Does the poem follow the requested meter?
- Meaning: Does the poem match the topic without merely copying keywords?
- Fluency/coherence: Is the Arabic grammatical, natural, and coherent across lines?
- Poeticness: Does it read like poetry rather than plain prose or templatic text?
- Overall: Would this be acceptable as generated classical Arabic verse?

### 19.4 Suggested Test Prompt Set

Build a balanced prompt set over:

- common meters: الطويل, الكامل, البسيط, الوافر.
- mid-resource meters: الخفيف, السريع, المتقارب, الرمل.
- difficult/low-resource meters: الرجز, الهزج, المديد.
- lengths: 2, 4, 6, 8, 10 bayts.
- topics: praise, longing, wisdom, elegy, nature, courage, separation, nostalgia.

Avoid testing only easy short poems. Include both seen-like and challenging prompt combinations.

## 20. Tables To Include In The Paper

Table 1: Related models.

| Model/System | Type | Relevance to Shaer |
|---|---|---|
| Ashaar | Arabic poetry analysis/generation framework | closest poetry-specific prior work |
| AraPoemBERT | poetry analysis encoder | useful for analysis context, not generation |
| AraGPT2 | Arabic generation LM | basis for GPT-2 Arabic generation baselines |
| Fanar | Arabic-centric LLM platform | broader Arabic LLM ecosystem |
| Hala-9B | Arabic instruction model | candidate/base comparison |
| Yehia-7B | Arabic instruction model | Shaer base model |

Table 2: Dataset lineage.

| Stage | Dataset | Rows | Main change |
|---|---|---:|---|
| base-form | `ashaar-v1-base-form` | 143,025 train + 882 validation | canonical meter/form labels |
| descriptions | `ashaar-v1-base-form-with-descriptions` | 143,022 | adds topic descriptions |
| SFT trimmed | `ashaar-with-descriptions-baseform-final-trimmed` | 129,610 | SFT format and token filtering |
| final unsplit | `...lte20-min500` | 116,032 | semantic description regeneration, length and meter-support filtering |
| final split | `...lte20-min500-splits` | 109,070 / 3,481 / 3,481 | deterministic stratified train/eval/test |

Table 3: Training configuration.

Include base model, dataset, QLoRA, LoRA rank/alpha/dropout, target modules, optimizer, LR, batch, grad accumulation, epochs, max length, sampling method.

Table 4: Final evaluation.

Fill after human/automatic evaluation.

Table 5: Per-meter results.

Report each meter separately. This is important because average meter adherence can hide weak meters.

## 21. Error Analysis To Add Later

The paper should analyze:

- meters with weak adherence, especially المديد, الهزج, and الرجز in the current probe.
- whether failures are due to low data, difficult rhythm, noisy labels, or generation length.
- cases where the model follows meter but misses meaning.
- cases where the model follows meaning but breaks meter.
- repetitions, templatic openings, and generic poetic language.
- whether long requested outputs degrade more than short outputs.

Suggested categories:

| Error Type | Description | Example Needed |
|---|---|---|
| meter drift | starts correct then changes meter | TBD |
| topic drift | poem becomes generic or unrelated | TBD |
| prose-like output | fluent Arabic but not poetic | TBD |
| repetition | repeated line openings or phrases | TBD |
| length failure | wrong number of shatrs | TBD |
| weak diction | modern/plain wording instead of poetic register | TBD |

## 22. Limitations

Suggested limitations section:

- The dataset is derived from existing poetry corpora and inherits their coverage biases.
- Some meters and forms are much more frequent than others.
- The current adapter improves control but per-meter probe results show uneven adherence.
- The system targets classical vertical Arabic poetry and does not cover free verse, dialect poetry, or modern prose poetry.
- Evaluation of poeticness and meaning requires human judgment; automatic metrics alone are insufficient.
- Semantic descriptions are generated automatically and may still contain occasional abstraction or interpretation errors.
- The model may reproduce stylistic patterns from the training corpus.

## 23. Ethics and Data Statement

Keep this section factual and modest:

- The project generates poetic text, not factual advice.
- The model can produce culturally sensitive or historically stylized content depending on prompts.
- The system should not be presented as replacing human poets.
- If examples are included, avoid offensive, political, or religiously sensitive prompts unless intentionally analyzed.
- State the public artifact links and licenses as shown in the model/dataset cards.

Do not include unsupported claims about the license of the original Ashaar source unless verified.

## 24. Reproducibility Checklist

Public artifacts:

- adapters: https://huggingface.co/Shaer-AI/Shaer-adapters
- final unsplit dataset: https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500
- final split dataset: https://huggingface.co/datasets/Shaer-AI/ashaar-with-enhanced-descriptions-baseform-final-sft-lte20-min500-splits
- base model: https://huggingface.co/Navid-AI/Yehia-7B-preview

Local reproducibility anchors:

- `description_generation/final_sft_dataset.py`
- `description_generation/regenerate_descriptions.py`
- `description_generation/tatweel_cleanup_publish.py`
- `description_generation/prompt_contracts.py`
- `sft/publish_sft_stratified_splits.py`
- `sft/split_utils.py`
- `sft/train_sft.py`
- `train_sft_qlora.py`
- `sft/outputs/train/train_20260407_231929/run_summary.json`
- `sft/outputs/train/train_20260407_231929/config_snapshot.json`
- `sft/outputs/train/train_20260407_231929/split_summary.json`
- `artifacts/sft/train_20260407_231929_key_metrics.json`

## 25. Citation and Source Notes

Use these references in the paper draft:

- ArabicNLP 2026 CFP: https://www.aclweb.org/portal/node/14460
- Ashaar paper: https://arxiv.org/abs/2307.06218
- Ashaar HF paper page: https://huggingface.co/papers/2307.06218
- AraPoemBERT: https://arxiv.org/abs/2403.12392
- AraGPT2 ACL Anthology: https://aclanthology.org/2021.wanlp-1.21
- Fanar: https://arxiv.org/abs/2501.13944
- Hala technical report: https://arxiv.org/abs/2509.14008
- Hala ACL/PDF entry: https://aclanthology.org/2026.abjadnlp-1.32.pdf
- AraLingBench: https://arxiv.org/abs/2511.14295
- AraLingBench KAUST PDF: https://repository.kaust.edu.sa/bitstreams/5a7fa661-7e91-401c-b1ab-1d9e3085385f/download
- Qwen3 technical report: https://arxiv.org/abs/2505.09388
- Yehia-7B model card: https://huggingface.co/Navid-AI/Yehia-7B-preview
- Hala-9B model card: https://huggingface.co/hammh0a/Hala-9B

## 26. What The Friend Writing The Paper Should Do Next

1. Convert Sections 1-5 into Abstract and Introduction.
2. Convert Section 6 into Related Work.
3. Convert Sections 7-14 into Dataset and Method.
4. Convert Section 15 into Experimental Setup.
5. Convert Sections 16-18 into preliminary Results and Training Analysis.
6. Fill Section 19 with the real final evaluation.
7. Fill Section 8’s Yehia-vs-Hala subsection with the internal comparison.
8. Add qualitative examples: one strong Shaer poem, one weak base-Yehia output, one Hala comparison, and one difficult-meter weak output.
9. Add a clean dataset pipeline figure.
10. Add a final table comparing Shaer to baselines on meter, meaning, fluency/coherence, and poeticness.

## 27. Recommended Final Paper Framing

Do:

- frame Shaer as a dataset-plus-adapter project.
- emphasize controlled Arabic poetry generation.
- emphasize Yehia selection.
- emphasize meter/form/length/topic conditioning.
- emphasize the clean SFT pipeline.
- include exact dataset counts and training hyperparameters.
- report per-meter evaluation.

Do not:

- discuss discarded experiments.
- make the paper about prompt-search history.
- make claims about Hala without the internal comparison data.
- claim state of the art before the evaluation table supports it.
- rely only on loss as evidence of poetry quality.

Best final paper title candidates:

1. `Shaer: Advancing Controlled Classical Arabic Poetry Generation with Yehia-7B`
2. `Shaer: Meter-Conditioned Classical Arabic Poetry Generation with Yehia-7B`
3. `Shaer: Dataset Curation and QLoRA Adaptation for Arabic Classical Poetry Generation`
4. `Advancing Arabic Classical Poetry Generation with Meter-Aware SFT of Yehia-7B`

The strongest title is probably:

> Shaer: Advancing Controlled Classical Arabic Poetry Generation with Yehia-7B
