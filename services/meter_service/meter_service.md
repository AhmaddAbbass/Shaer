Aha perfect, this changes the game in a *nice* way 😌
Now meter_service doesn’t need some abstract “mystery scansion model” – it can **wrap this exact BiLSTM classifier**.

Let me rewrite the **meter_service spec** assuming:

* You already have this BiLSTM code living somewhere like `Models/Ashaar_runtime/...`.
* We will *use* `classifier_meter_scores(text)` as the core engine.
* `BaytMeterEval` stays simple:
  `target_meter, meter_score (0–100), on_meter (bool), notes`.

No code, just super concrete behavior/spec.

---

## 1. What this BiLSTM gives us

From your snippet we have:

* `classifier_meter_scores(text) -> Dict[meter_label, prob]`

  * Lazy-loads:

    * a Keras BiLSTM model (`poem_meter_bilstm.keras`),
    * a label encoder (`meter_label_encoder.joblib`),
    * a char vocab config JSON (with `stoi` and `max_len`).
  * Returns a dict like:

    ```python
    {
      "البسيط": 0.87,
      "الكامل": 0.05,
      "الطويل": 0.03,
      ...
    }
    ```

* `meter_reward(completions, poem_meter=...) -> list[float]`

  * This is GRPO-oriented: it:

    * extracts text from completions,
    * calls `classifier_meter_scores`,
    * if `poem_meter` given: returns probability of that label (clamped 0–1),
    * otherwise returns `max(probabilities)`.

For **meter_service**, we don’t need GRPO batching; we want **one bayt at a time**:

> Take `verse_text` + `target_meter` → compute probability that bayt belongs to that meter, plus some explanation.

So internally we will:

* Call `classifier_meter_scores(verse_text)`.
* Use the probability of the requested `target_meter` as the **score basis**.
* Also look at the best meter predicted by the model (argmax).

---

## 2. Schema recap (unchanged)

We keep the earlier API contract; we just refine what happens inside.

### Common schema (from `common_schemas.schemas`)

* `BaytMeterEval`:

  * `target_meter: str`
  * `meter_score: int` (0–100)
  * `on_meter: bool`
  * `notes: str`

### meter_service HTTP models (`services/meter_service/app/schemas.py`)

* `EvalBaytRequest`:

  * `verse_text: str`
  * `target_meter: str`
  * Optional: `return_details: bool = False` (if you later want debug info).

* `EvalBaytResponse`:

  * `result: BaytMeterEval`

We’re not changing this shape; just how `BaytMeterEval` is computed.

---

## 3. `config.py` for meter_service (tuned to BiLSTM)

Before, we had generic `METER_BACKEND`. Now we can be more concrete:

### Env variables to support

In `services/meter_service/.env`:

* Backend / model:

  * `METER_BACKEND=bilstm`
    (for now we only support this; later you can add others.)

  * Optional overrides (if you want to override defaults in your GRPO module):

    * `BILSTM_MODEL_PATH`
    * `BILSTM_LABEL_ENCODER_PATH`
    * `BILSTM_VOCAB_CONFIG_PATH`

    If not set, we fall back to the defaults in your snippet:
    `"Models/Ashaar_runtime/bilstm_only/bilstm_model/training/..."`.

* Scoring logic:

  * `METER_SCORE_THRESHOLD=80`
    → if score ≥ 80 → `on_meter = True`.
  * Optional fine-grain:

    * `METER_HIGH_CONF_THRESHOLD=90`
    * `METER_LOW_CONF_THRESHOLD=50`
      (useful for shaping notes; see below).

* Runtime:

  * `METER_LOG_LEVEL=INFO`
  * `METER_TIMEOUT_SECONDS=10` (only relevant if you later add remote calls).

### config.py responsibilities

* Load env vars and expose:

  * `BACKEND_MODE = "bilstm"`
  * `BILSTM_MODEL_PATH`, `BILSTM_LABEL_ENCODER_PATH`, `BILSTM_VOCAB_CONFIG_PATH`
  * `SCORE_THRESHOLD` (int)
  * `HIGH_CONF_THRESHOLD`, `LOW_CONF_THRESHOLD`
  * `LOG_LEVEL`

* Optionally: a mapping of **user-facing meter names → label encoder names** if they ever differ (more on this next).

---

## 4. Meter label mapping logic

Very important question:

> *Do the label encoder’s class names match your external `poem_meter` names?*
> e.g. exactly `"البسيط"`, `"الطويل"`, `"الكامل"` etc.

There are two possibilities:

### Case A – They **already match**

If your GRPO `poem_meter` column used the exact same strings from the dataset, and that’s what you’ll send in `target_meter`, then you’re golden:

* `classifier_meter_scores` returns keys like `"البسيط"`.
* `target_meter` from `PoemSpec` is also `"البسيط"`.
* So `dist[target_meter]` works directly.

### Case B – They *don’t* match 1:1

Maybe the label encoder uses:

* `"basit"`, `"tawil"`, `"kamil"` etc.
  While your external API expects Arabic names.

Then `meter_service` needs a **mapping** in `config.py`, something like:

* `METER_NAME_MAP = { "البسيط": "basit", "الطويل": "tawil", ... }`

And in the logic you’ll:

* Map `target_meter` → `encoder_label`.
* Use that key when reading `dist`.

For the spec, we just say:

> `scansion.py` must ensure `target_meter` is mapped to the label space of `classifier_meter_scores`.

---

## 5. `scansion.py` – wrapping the BiLSTM classifier

Now let’s re-spec **how `scansion.evaluate_bayt` works** given your BiLSTM utilities.

### 5.1. Initialization

* On first import / first call, `scansion.py`:

  * Imports `classifier_meter_scores` from your existing module, e.g.:

    > (`Models.Ashaar_runtime.bilstm_only.bilstm_model.meter_utils` or wherever you put this snippet).

  * Relies on that module’s global lazy loading:

    * `_load_bilstm_assets` uses `PROJECT_ROOT / "Models" / "Ashaar_runtime" / ...`.

  * Optionally, if you want, you can:

    * Override `model_path`, `label_encoder_path`, `vocab_config_path` using env values from `config.py`.

  * Since your snippet already forces TensorFlow to use CPU and hide GPU (`set_visible_devices([], "GPU")`), this is nice: no GPU conflicts with PyTorch.

* On startup, you can log something like:

  > “Meter backend: BiLSTM (Ashaar_runtime), model path: ...”

### 5.2. Preprocessing

Your BiLSTM encoder ` _encode_text_to_ints` already does:

* `text.replace("[sep]", " ")`,
* char-level encoding with `_STOI` and `_MAX_LEN`.

For meter_service, we can keep preprocessing **minimal**:

* `text = verse_text.strip()`
* (optional) unify `[sep]` if your verses might contain it.

So we don’t need to reinvent normalization here; we trust the classifier.

### 5.3. Core evaluation algorithm

Define the conceptual function:

> `evaluate_bayt(verse_text: str, target_meter: str) -> BaytMeterEval`

Steps:

1. **Empty / too short text handling**

   * If `verse_text` is empty or extremely short (e.g. < 3 chars):

     * return `meter_score = 0`,
     * `on_meter = False`,
     * `notes = "النص قصير جدًا ولا يمكن تقييم وزنه."`.

2. **Get probability distribution**

   * Call:

     ```python
     dist = classifier_meter_scores(verse_text)
     ```

   * Now you have a dict `{label: prob}`.

   * If `dist` is empty or model returns nonsense:

     * treat as failure; return score 0, `on_meter=False`, notes “تعذّر تقييم الوزن.”.

3. **Resolve target label**

   * Use `config` to map `target_meter` into label space:

     * If Case A (matching labels): `target_label = target_meter`.
     * If Case B: `target_label = METER_NAME_MAP.get(target_meter)`.

   * If `target_label` is `None` or not in `dist`:

     * This means **unsupported meter**.
     * Option 1 (strict): return HTTP 400 from `api.py`.
     * Option 2 (graceful):

       * set `meter_score = 0`,
       * `on_meter = False`,
       * `notes = "هذا البحر غير مدعوم من نموذج الوزن."`.

4. **Extract target probability + best meter**

   * `prob_target = dist.get(target_label, 0.0)`
   * `best_label = argmax(dist)`
   * `best_prob = dist[best_label]`

5. **Convert probability to score 0–100**

   * `meter_score = int(round(prob_target * 100))`

   This matches your GRPO reward: probability in [0, 1] → scalar in [0, 1]; we just rescale to 0–100.

6. **Determine `on_meter`**

   * `SCORE_THRESHOLD = METER_SCORE_THRESHOLD` from env (e.g. 80).

   * `on_meter = meter_score >= SCORE_THRESHOLD`

7. **Craft human notes**

   Use `HIGH_CONF_THRESHOLD` and `LOW_CONF_THRESHOLD` to give better notes. Example behavior:

   * If `prob_target` is very high (≥ high threshold, say 0.9):

     > `notes = f"البيت مضبوط على بحر {target_meter} بدرجة ثقة عالية (≈{meter_score}%)."`

   * Else if `on_meter == True` but not super high:

     > `notes = f"البيت مناسب عمومًا لبحر {target_meter} (احتمال ≈{meter_score}%). قد توجد بعض الاختلافات الجزئية في التفعيلات."`

   * Else if `on_meter == False` and `best_prob` is moderately high:

     > `notes = f"النموذج يرجّح أن البيت من بحر {best_label} (≈{int(best_prob * 100)}%) أكثر من بحر {target_meter}."`

   * Else (both target and best probs are low):

     > `notes = "البيت بعيد عن أوزان البحور المعروفة في النموذج؛ ربما يحتاج إلى إعادة صياغة شاملة من ناحية الوزن."`

8. **Return BaytMeterEval**

   * `BaytMeterEval(target_meter=target_meter, meter_score=meter_score, on_meter=on_meter, notes=notes)`

This is the whole “meter brain” of the service.

---

## 6. `api.py` – now with BiLSTM semantics

The HTTP logic doesn’t change much, but we can be explicit:

### `POST /eval-bayt` flow

1. Parse JSON → `EvalBaytRequest`:

   * `verse_text` (required),
   * `target_meter` (required).

2. Basic validation:

   * If `verse_text` is empty:

     * return HTTP 400 with message `"verse_text must not be empty"`.
   * If `target_meter` is empty:

     * HTTP 400 `"target_meter must not be empty"`.

3. Call:

   ```python
   result = scansion.evaluate_bayt(request.verse_text, request.target_meter)
   ```

   * Inside, we use BiLSTM + mapping logic above.

4. Wrap in `EvalBaytResponse` and return.

5. If `evaluate_bayt` raises an exception (e.g. model file missing):

   * Log error,
   * Return a 500 with a friendly message (or a fallback `BaytMeterEval` with score 0 and a “service unavailable” note, depending on your philosophy).

---

## 7. Testing with the BiLSTM backbone

### `test_eval_bayt.py` (unit-ish, no TF required if mocked)

* For tests, you don’t want to load TensorFlow every time. So:

  * Mock `scansion.classifier_meter_scores` to return a fixed dict, e.g.:

    ```python
    {"البسيط": 0.93, "الكامل": 0.02}
    ```

  * Then:

    * Case 1:

      * target_meter="البسيط"
      * Expect:

        * `meter_score ≈ 93`, `on_meter=True`,
        * notes mention بحر البسيط.

    * Case 2:

      * target_meter="الكامل"
      * Expect:

        * `meter_score ≈ 2`, `on_meter=False`,
        * notes mention that best meter is البسيط.

    * Case 3:

      * unsupported meter name:

        * Expect a clean behavior: either HTTP 400 or `on_meter=False` with note.

### (Optional) integration test (real TF)

* In a separate “slow” test, you can:

  * Actually load the BiLSTM assets (no mocking).
  * Feed a known textbook bayt in a known meter.
  * Check that score is high and `on_meter=True`.

---

## 8. Docker implications (with TF)

Because your BiLSTM is Keras/TF-based:

* `meter_service` Docker image will need:

  * `tensorflow` (matching version used to save the model; your snippet already handles some Keras 3 quirks).
  * `joblib` and `scikit-learn` (or at least joblib + `sklearn.exceptions`).
  * `numpy`, `json`, etc.

* The snippet already:

  * Forces `TF_CPP_MIN_LOG_LEVEL="2"` (warn+ only),
  * Tries to hide GPU devices (`tf.config.set_visible_devices([], "GPU")`),
  * Forces dtype policy float32.

This is actually ideal: **meter_service stays CPU-only**, not fighting with PyTorch GPU usage for Shaer.

---

## 9. How orchestrator uses this new BiLSTM-based meter_service

Nothing changes in how orchestrator sees it:

* It calls `POST /eval-bayt` with:

  ```json
  {
    "verse_text": "والورد تَخجله أناملُ سوسنٍ",
    "target_meter": "البسيط"
  }
  ```

* It receives:

  ```json
  {
    "result": {
      "target_meter": "البسيط",
      "meter_score": 92,
      "on_meter": true,
      "notes": "البيت مضبوط على بحر البسيط بدرجة ثقة عالية (≈92%)."
    }
  }
  ```

Then enhancer logic is:

* If `on_meter == False` or `meter_score < some_strict_threshold` (maybe 85 for generation), trigger regeneration via `shaer_client`.

---

So to summarize the **reimplementation**:

* `meter_service` stays structurally identical (same endpoints).
* `scansion.py` becomes a **thin wrapper** around your existing `classifier_meter_scores`.
* All the heavy TF loading / compatibility hacks stay **inside** that Ashaar_runtime module.
* The service simply:

  * calls that function,
  * converts probabilities to 0–100,
  * compares to threshold,
  * crafts neat Arabic notes.
