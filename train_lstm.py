# train_lstm.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Config ---
PROCESSED_DIR = "processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
UNITS = 64        # LSTM units for first layer
DROPOUT = 0.2
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_tradewise.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "history.npy")
PLOT_PATH = os.path.join(MODEL_DIR, "pred_vs_true.png")

# --- Load data + scaler ---
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

print("Shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# --- Build model ---
tf.keras.backend.clear_session()
model = Sequential([
    LSTM(UNITS, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(DROPOUT),
    BatchNormalization(),
    LSTM(UNITS//2, return_sequences=False),
    Dropout(DROPOUT),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
model.summary()

# --- Callbacks ---
es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[es, mc],
    verbose=2
)

# Save history (loss arrays)
np.save(HISTORY_PATH, history.history)
print("Saved history to", HISTORY_PATH)
print("Best epoch (approx):", np.argmin(history.history['val_loss'])+1)

# --- Load best model (safety) ---
model.load_weights(MODEL_PATH)
print("Loaded best model weights from", MODEL_PATH)

# --- Predict on test set ---
y_pred_scaled = model.predict(X_test)
# inverse scale (scaler expects shape (-1,1))
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

# --- Metrics on original price scale ---
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)

print(f"Test R2: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test MSE: {mse:.4f}")

# --- Save plot ---
plt.figure(figsize=(12,6))
plt.plot(y_true, label='Actual (test)', linewidth=1)
plt.plot(y_pred, label='Predicted (test)', linewidth=1)
plt.legend()
plt.title(f"Test Predictions vs Actuals — R2={r2:.3f}, MAE={mae:.2f}")
plt.xlabel("Test sample index")
plt.ylabel("Price")
plt.grid(True)
plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=150)
print("Saved prediction plot to", PLOT_PATH)

# --- Save model in TF SavedModel format (optional) ---
SAVED_MODEL_DIR = os.path.join(MODEL_DIR, "saved_model")
tf.saved_model.save(model, SAVED_MODEL_DIR)
print("Saved full model to", SAVED_MODEL_DIR)

# quick diagnostics: view a few samples
print("\nSample comparisons (first 20 test samples):")
for i in range(20):
    print(f"{i:02d} true={y_true[i]:.2f}  pred={y_pred[i]:.2f}  diff={y_pred[i]-y_true[i]:+.2f}")


# Print locations of saved artifacts
print("\nArtifacts saved:")
print(" - Keras .h5 weights:", MODEL_PATH)
print(" - SavedModel folder:", SAVED_MODEL_DIR)
print(" - History .npy:", HISTORY_PATH)
print(" - Plot:", PLOT_PATH)
