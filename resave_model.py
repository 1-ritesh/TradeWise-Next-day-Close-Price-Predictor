# resave_model.py
import os
import tensorflow as tf

models_dir = "models"

h5_path = os.path.join(models_dir, "lstm_tradewise.h5")
keras_path = os.path.join(models_dir, "lstm_tradewise_fixed.keras")

print("Exists .h5:", os.path.exists(h5_path))

# 1. Load using the same TF/Keras version you trained with
model = tf.keras.models.load_model(h5_path, compile=False)
print("Loaded .h5 successfully.")

# 2. Force a re-trace before saving  (.keras format needs this!)
model.build(input_shape=(None, 60, 1))  # <--- IMPORTANT
print("Model rebuilt: input shape set to (None,60,1)")

# 3. Save clean .keras
model.save(keras_path, include_optimizer=False)
print("Saved clean .keras to:", keras_path)
