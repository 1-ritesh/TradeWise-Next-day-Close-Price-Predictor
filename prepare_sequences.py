# prepare_sequences.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

CSV_PATH = os.path.join("data", "AAPL_6y.csv")  # change if needed
WINDOW_SIZE = 60
TEST_RATIO = 0.1
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
print("Loaded:", CSV_PATH, "rows:", len(df))

series = df['Close'].dropna().values.reshape(-1, 1)
print("Close series shape:", series.shape)

# Create raw (unscaled) sliding windows first
X_raw, y_raw = [], []
for i in range(WINDOW_SIZE, len(series)):
    X_raw.append(series[i-WINDOW_SIZE:i, 0])  # (WINDOW_SIZE,)
    y_raw.append(series[i, 0])                # scalar

X_raw = np.array(X_raw)  # (n_samples, WINDOW_SIZE)
y_raw = np.array(y_raw)  # (n_samples,)

# Chronological train/test split
split_index = int((1 - TEST_RATIO) * X_raw.shape[0])
X_train_raw = X_raw[:split_index]
y_train_raw = y_raw[:split_index]
X_test_raw = X_raw[split_index:]
y_test_raw = y_raw[split_index:]

print("Raw train samples:", X_train_raw.shape[0], "Raw test samples:", X_test_raw.shape[0])

# Fit scaler only on training CLOSE values (reshape to (-1,1))
scaler = MinMaxScaler(feature_range=(0, 1))
# fit scaler on the flattened training values (all values used for stable scaling)
scaler.fit(y_train_raw.reshape(-1, 1))

# apply the scaler to whole data (train+test)
def scale_windows(X_raw, scaler):
    # X_raw shape: (n_samples, WINDOW_SIZE)
    n, w = X_raw.shape
    X_flat = X_raw.reshape(-1, 1)  # (n*w, 1)
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(n, w)
    return X_scaled

X_train_scaled = scale_windows(X_train_raw, scaler)
X_test_scaled = scale_windows(X_test_raw, scaler)
y_train_scaled = scaler.transform(y_train_raw.reshape(-1, 1)).reshape(-1, 1)
y_test_scaled = scaler.transform(y_test_raw.reshape(-1, 1)).reshape(-1, 1)

# reshape for LSTM: (samples, timesteps, features)
X_train = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
y_train = y_train_scaled
y_test = y_test_scaled

print("X shape (samples, timesteps, features):", X_train.shape, X_test.shape)
print("y shape:", y_train.shape, y_test.shape)

# Save scaler and arrays
with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("Saved scaler to", os.path.join(OUT_DIR, "scaler.pkl"))

np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)
print("Saved arrays to", OUT_DIR)
