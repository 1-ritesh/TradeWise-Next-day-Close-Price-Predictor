# debug_scaler_and_stats.py
import os, pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

PROCESSED = "processed"
MODELS = "models"

# load arrays and scaler
X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED, "X_test.npy"))
y_train = np.load(os.path.join(PROCESSED, "y_train.npy"))  # scaled (1-col)
y_test  = np.load(os.path.join(PROCESSED, "y_test.npy"))

with open(os.path.join(PROCESSED, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# Also load unscaled original arrays by reconstructing from CSV if present
import pandas as pd
csv_path = os.path.join("data", "AAPL_6y.csv")
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
series = df['Close'].dropna().values.reshape(-1,1)

WINDOW = X_train.shape[1] if len(X_train.shape)>1 else 60
# Rebuild raw windows to get original y_train/y_test in original scale
X_raw, y_raw = [], []
for i in range(WINDOW, len(series)):
    X_raw.append(series[i-WINDOW:i, 0])
    y_raw.append(series[i, 0])
X_raw = np.array(X_raw)
y_raw = np.array(y_raw).reshape(-1,1)

split = X_train.shape[0]
y_train_orig = y_raw[:split].reshape(-1,1)
y_test_orig  = y_raw[split:].reshape(-1,1)

print("Samples -> train:", len(y_train_orig), " test:", len(y_test_orig))
print()
print("Original TRAIN y stats: min, max, mean: ", float(y_train_orig.min()), float(y_train_orig.max()), float(y_train_orig.mean()))
print("Original TEST  y stats: min, max, mean: ", float(y_test_orig.min()),  float(y_test_orig.max()),  float(y_test_orig.mean()))
print()
# Scaler internals (if available)
try:
    print("Scaler data_min_, data_max_ (scaler):", scaler.data_min_, scaler.data_max_)
    print("Scaler scale_:", scaler.scale_)
except Exception as e:
    print("Could not read scaler internals:", e)

# Load model predictions from previous run if saved numpy preds exist, else compute by loading model
preds_file = os.path.join(MODELS, "last_y_pred.npy")
if os.path.exists(preds_file):
    y_pred = np.load(preds_file)
    print("\nLoaded saved y_pred from", preds_file)
else:
    # load model and predict
    import tensorflow as tf
    model = tf.keras.models.load_model(os.path.join(MODELS, "saved_model"))
    y_pred_scaled = model.predict(X_test)
    # inverse scale predicted and true
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).reshape(-1,1)
    print("\nPredictions computed using models/saved_model")

y_true = y_test_orig

print("\nPred mean/min/max:", float(y_pred.mean()), float(y_pred.min()), float(y_pred.max()))
print("True mean/min/max:", float(y_true.mean()), float(y_true.min()), float(y_true.max()))
print("\nFirst 10 pairs (true, pred):")
for i in range(10):
    print(f"{i:02d} true={y_true[i,0]:.2f}  pred={y_pred[i,0]:.2f}  diff={y_pred[i,0]-y_true[i,0]:+.2f}")

# metrics (original scale)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2 = r2_score(y_true.flatten(), y_pred.flatten())
mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
print(f"\nMetrics on original scale -> R2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")

# save the pred for inspection if needed
np.save(os.path.join(MODELS, "debug_y_pred.npy"), y_pred)
print("\nSaved debug_y_pred.npy in models/")
