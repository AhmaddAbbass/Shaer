from __future__ import annotations

import os
import sys


def main():
    # Use standalone Keras 3, NOT tf.keras
    import keras
    from keras import saving

    print("Keras version:", keras.__version__)

    # --------------------------------------------------------
    # Dummy DTypePolicy so Keras can deserialize the old config
    # {"class_name": "DTypePolicy", "config": {"name": "float32"}, ...}
    # --------------------------------------------------------
    class DummyDTypePolicy:
        def __init__(self, name="float32", **kwargs):
            self.name = name

        def get_config(self):
            return {"name": self.name}

    # Register under the exact name that appears in the saved config
    saving.register_keras_serializable(package="keras", name="DTypePolicy")(
        DummyDTypePolicy
    )

    src = "Models/Ashaar_runtime/bilstm_only/bilstm_model/training/poem_meter_bilstm.keras"
    if not os.path.exists(src):
        print(f"Source model not found at: {src}")
        sys.exit(1)

    # 1) Load old model
    try:
        model = saving.load_model(src, compile=False)
        print("Loaded model via keras.saving.load_model")
    except Exception as e:
        print("Failed to load via keras.saving.load_model:", repr(e))
        sys.exit(1)

    out_dir = "Models/Ashaar_runtime/bilstm_only/bilstm_model"
    os.makedirs(out_dir, exist_ok=True)

    # 2) Save to H5
    h5_path = os.path.join(out_dir, "poem_meter_bilstm_v2.h5")
    try:
        saving.save_model(model, h5_path, include_optimizer=False)
        print(f"Saved H5 model to: {h5_path}")
    except Exception as e:
        print("Failed to save H5:", repr(e))

    # 3) Save to new .keras
    keras_path = os.path.join(out_dir, "poem_meter_bilstm_v2.keras")
    try:
        saving.save_model(model, keras_path, include_optimizer=False)
        print(f"Saved Keras model to: {keras_path}")
    except Exception as e:
        print("Failed to save .keras:", repr(e))


if __name__ == "__main__":
    main()
