# Meter Service Load Failure (BiLSTM meter model)

The meter service starts but cannot load the legacy BiLSTM meter model. This prevents real meter scoring and forces the fallback result (score=0, on_meter=false).

## Symptom
Container logs (meter_service) show:
```
Failed to load meter assets: Layer 'embedding' expected 1 variables, but received 0 variables during loading. Expected: ['embedding/embeddings:0']
```
Earlier attempts also hit:
```
Unknown dtype policy: 'DTypePolicy'
```

## What we tried
- Pinned TensorFlow CPU to 2.12.0 (lighter, legacy-friendly).
- Set `TF_USE_LEGACY_KERAS=1`.
- Registered `DTypePolicy` via custom_object_scope and `tf.keras.utils.get_custom_objects`.
- Added custom objects when loading the model.
- Tried multiple TF pins (2.15.1 too heavy to finish cleanly in container).
- Tests pass because they mock the scansion layer; the real model still fails to deserialize.

## Likely cause
- The saved model at `models/Ashaar_runtime/bilstm_model/training/poem_meter_bilstm.keras` was serialized with a newer/older Keras stack and uses a custom dtype/policy. Current TF builds can parse the config but the embedding layer weights are not being restored (0 variables found).

## How to unblock
1) Re-export the meter model with compatible serialization (recommended):
   - On a box with the original training environment, load the model and re-save as:
     ```python
     import tensorflow as tf
     model = tf.keras.models.load_model("poem_meter_bilstm.keras", compile=False)
     model.save("poem_meter_bilstm_savedmodel", include_optimizer=False, save_format="tf")
     # optionally: model.save("poem_meter_bilstm.h5", include_optimizer=False)
     ```
   - Ship the new SavedModel/H5 into `models/Ashaar_runtime/bilstm_model/` and update loader paths.

2) Try a more compatible runtime (heavier):
   - Install `tensorflow-cpu==2.15.1` plus `tf-keras==2.15.0` in the Dockerfile.
   - Risk: large image and still may fail if weights missing.

3) Temporary alternative:
   - Swap to a different meter check (rule-based/classifier) that loads cleanly; adjust `scansion.py` to use it.

## Current state
- Meter service container builds and runs on port 8004 but reports `assets_loaded: false` and returns fallback meter responses (no real scoring).

## Reproduction
```
docker build -t meter-service -f services/meter_service/Dockerfile .
docker run --rm -p 8004:8004 meter-service
# logs show embedding variables missing
```

## Suggested next step
Re-export the BiLSTM model with its weights using a matching Keras/TF and replace the on-disk artifact. If that’s not possible, try the `tensorflow-cpu==2.15.1` + `tf-keras==2.15.0` combo and verify if weights load. Otherwise, replace the meter evaluator with a simpler classifier that loads.***
