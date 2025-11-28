from __future__ import annotations

import os
import sys


def main() -> None:
    # Lazy import so the script fails fast if TF is missing.
    import tensorflow as tf

    print("TF version:", tf.__version__)

    src = "models/Ashaar_runtime/bilstm_model/training/poem_meter_bilstm.keras"
    if not os.path.exists(src):
        print(f"Source model not found at: {src}")
        sys.exit(1)

    try:
        model = tf.keras.models.load_model(src, compile=False)
        print("Loaded model via tf.keras")
    except Exception as e:
        print("Failed to load via tf.keras:", e)
        print("If this is a Keras 3 artifact, install `keras` and try `keras.saving.load_model` to convert.")
        sys.exit(1)

    # H5 export first (most portable)
    h5_path = "models/Ashaar_runtime/bilstm_model/poem_meter_bilstm_v2.h5"
    try:
        model.save(h5_path, include_optimizer=False)
        print(f"Saved H5 model to: {h5_path}")
    except Exception as e:
        print("Failed to save H5:", e)

    # SavedModel export (optional; may fail on some TF versions)
    savedmodel_path = "models/Ashaar_runtime/bilstm_model/poem_meter_bilstm_savedmodel"
    try:
        model.save(savedmodel_path, include_optimizer=False, save_format="tf", signatures=None, options=None)
        print(f"Saved SavedModel to: {savedmodel_path}")
    except Exception as e:
        print("SavedModel export failed (continuing, H5 may be enough):", e)


if __name__ == "__main__":
    main()
