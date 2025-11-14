# test_wrapper_savedtf.py
import os, numpy as np, tensorflow as tf
from tensorflow import keras

WINDOW = 60
models_dir = "models"
saved_tf_path = os.path.join(models_dir, "lstm_tradewise_savedtf")
legacy_saved = os.path.join(models_dir, "saved_model")

def try_wrap(path):
    print("Attempting to wrap SavedModel at:", path)
    layer = keras.layers.TFSMLayer(path, call_endpoint='serving_default')
    inputs = keras.Input(shape=(WINDOW,1), dtype=tf.float32)
    model = keras.Model(inputs=inputs, outputs=layer(inputs))
    return model

# Prefer re-saved TF folder, else legacy saved_model
target = None
if os.path.exists(saved_tf_path):
    target = saved_tf_path
elif os.path.exists(legacy_saved):
    target = legacy_saved
else:
    raise SystemExit("No SavedModel folder found in models/")

try:
    wrapper = try_wrap(target)
    print("Wrapper model created:", type(wrapper))
    X = np.random.rand(1, WINDOW, 1).astype("float32")
    out = wrapper.predict(X)
    print("Predict OK. Output shape:", np.array(out).shape)
    print("Sample output:", np.array(out).flatten()[:5])
except Exception as e:
    print("Wrapping/predict failed:", repr(e))
    # Try raw TF saved_model inference as fallback
    try:
        raw = tf.saved_model.load(target)
        signatures = list(raw.signatures.keys())
        print("Raw saved_model loaded. signatures:", signatures)
    except Exception as e2:
        print("Also failed to load raw saved_model:", repr(e2))
