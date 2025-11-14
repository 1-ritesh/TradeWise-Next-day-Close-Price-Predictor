# test_wrapper.py
import os, numpy as np, tensorflow as tf
from tensorflow import keras

WINDOW = 60
models_dir = "models"
keras_path = os.path.join(models_dir, "lstm_tradewise.keras")
h5_path = os.path.join(models_dir, "lstm_tradewise.h5")
saved_tf_path = os.path.join(models_dir, "lstm_tradewise_savedtf")
legacy_saved = os.path.join(models_dir, "saved_model")

wrapper = None
# try .keras
if os.path.exists(keras_path):
    print("Trying .keras")
    wrapper = tf.keras.models.load_model(keras_path, compile=False)
# fallback h5
elif os.path.exists(h5_path):
    print("Trying .h5")
    try:
        wrapper = tf.keras.models.load_model(h5_path, compile=False)
    except Exception as e:
        print("h5 load failed:", e)
# fallback saved_tf_path
elif os.path.exists(saved_tf_path):
    print("Trying saved_tf_path")
    try:
        wrapper = tf.keras.models.load_model(saved_tf_path, compile=False)
    except Exception as e:
        print("saved_tf_path load failed:", e)
# final: wrap legacy saved_model with TFSMLayer
elif os.path.exists(legacy_saved):
    print("Trying legacy saved_model (wrap with TFSMLayer)")
    layer = keras.layers.TFSMLayer(legacy_saved, call_endpoint='serving_default')
    inputs = keras.Input(shape=(WINDOW,1), dtype=tf.float32)
    wrapper = keras.Model(inputs=inputs, outputs=layer(inputs))

print("Wrapper type:", type(wrapper))
# run a dummy predict
X = np.random.rand(1, WINDOW, 1).astype("float32")
out = wrapper.predict(X)
print("Output shape:", np.array(out).shape)
print("Sample output (first 5):", np.array(out).flatten()[:5])
