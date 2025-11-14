# streamlit_app.py
"""
TradeWise — simple Streamlit UI to fetch history, show chart, and predict next-day Close
Assumptions:
 - Trained model weights exist at models/lstm_tradewise.h5 (preferred) OR a SavedModel at models/saved_model
 - Scaler exists at processed/scaler.pkl (fitted previously)
 - Model expects input shape: (1, 60, 1) and was trained on scaled Close values
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# plotting
import matplotlib.pyplot as plt

# tensorflow lazy import (load model in cached function)
import tensorflow as tf

# -------------------------
# Config / paths
# -------------------------
MODEL_H5_PATH = os.path.join("models", "lstm_tradewise.h5")
MODEL_SAVED_DIR = os.path.join("models", "saved_model")
SCALER_PATH = os.path.join("processed", "scaler.pkl")

WINDOW = 60  # must match training window

# -------------------------
# Caches for speed
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_history(ticker: str, period_days: int = 365*3):
    """Fetch daily history for ticker (period_days back)."""
    end = datetime.now()
    start = end - timedelta(days=period_days)
    df = yf.download(ticker, start=start.date(), end=end.date(), interval="1d", progress=False, auto_adjust=True)
    df = df.dropna()
    return df

@st.cache_resource(show_spinner=False)
def load_scaler(path=SCALER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Attempt to load a model and ALWAYS return a callable Keras-style object with .predict().
    Preference order:
    1) models/lstm_tradewise_savedtf (wrapped with TFSMLayer)
    2) models/saved_model (legacy) wrapped with TFSMLayer
    3) models/lstm_tradewise.h5 (legacy HDF5) loaded with compile=False (fallback)
    """
    import tensorflow as tf
    from tensorflow import keras
    models_dir = "models"
    saved_tf_path = os.path.join(models_dir, "lstm_tradewise_savedtf")
    legacy_saved = os.path.join(models_dir, "saved_model")
    h5_path = os.path.join(models_dir, "lstm_tradewise.h5")

    # Helper to wrap a SavedModel dir into a Keras model using TFSMLayer
    def wrap_savedmodel_dir(path):
        layer = keras.layers.TFSMLayer(path, call_endpoint='serving_default')
        inputs = keras.Input(shape=(WINDOW, 1), dtype=tf.float32)
        model_wrapper = keras.Model(inputs=inputs, outputs=layer(inputs))
        return model_wrapper

    # 1) re-saved TF directory (preferred)
    if os.path.exists(saved_tf_path):
        try:
            return wrap_savedmodel_dir(saved_tf_path)
        except Exception as e:
            st.write("Failed to wrap saved_tf_path:", e)

    # 2) legacy saved_model folder
    if os.path.exists(legacy_saved):
        try:
            return wrap_savedmodel_dir(legacy_saved)
        except Exception as e:
            st.write("Failed to wrap legacy saved_model:", e)

    # 3) fallback to .h5 (may still raise LSTM mismatch, but try)
    if os.path.exists(h5_path):
        try:
            return tf.keras.models.load_model(h5_path, compile=False)
        except Exception as e:
            st.write("Failed to load .h5 fallback:", e)

    raise FileNotFoundError("No compatible model found in models/. Try re-saving in training env.")



# -------------------------
# Utility: prepare last-60 days and scale
# -------------------------
def prepare_last_window(df: pd.DataFrame, scaler, window=WINDOW):
    """
    Input:
      df: dataframe with Close column indexed by date, at least `window` rows
      scaler: fitted MinMaxScaler that can inverse_transform/transform
    Returns:
      X_input: np.array shape (1, window, 1) scaled
      last_dates: list of dates used
      last_closes: original close prices for those dates (unscaled)
    """
    if "Close" not in df.columns:
        raise ValueError("Dataframe must contain 'Close' column")
    closes = df["Close"].values.astype(float)
    if len(closes) < window:
        raise ValueError(f"Need at least {window} historical Close values, got {len(closes)}")
    last_window = closes[-window:]
    # scale with scaler; scaler expects 2D array
    last_window_scaled = scaler.transform(last_window.reshape(-1, 1)).reshape(1, window, 1)
    return last_window_scaled, df.index[-window:], last_window

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="TradeWise — Next-day Price Predictor", layout="wide")
st.title("TradeWise — Next-day Close Price Predictor")

with st.sidebar:
    st.header("Prediction Settings")
    ticker = st.text_input("Ticker (Yahoo Finance)", value="AAPL").upper().strip()
    history_years = st.slider("History length (years)", 1, 6, 3)
    refresh = st.button("Refresh / Predict")
    st.markdown("---")
    st.markdown("**Model & Scaler**")
    st.markdown(f"- Model H5 found: `{os.path.exists(MODEL_H5_PATH)}`")
    st.markdown(f"- SavedModel folder found: `{os.path.exists(MODEL_SAVED_DIR)}`")
    st.markdown(f"- Scaler found: `{os.path.exists(SCALER_PATH)}`")
    st.markdown("---")
    st.markdown("Notes:")
    st.caption("Model expects past 60 days (window). The app uses the scaler saved during training. For best results, retrain periodically with recent data.")

# main area
col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f"{ticker} — Historical Close price")
    try:
        df = fetch_history(ticker, period_days=365 * history_years)
        if df.empty:
            st.error("No historical data returned. Check ticker or network.")
        else:
            st.line_chart(df["Close"])
            st.write(f"Showing {len(df)} rows (last: {df.index[-1].date()})")
    except Exception as e:
        st.exception(e)
        st.stop()

with col2:
    st.subheader("Model Status")
    try:
        scaler = load_scaler()
        model = load_model()
        st.success("Scaler and model loaded.")
        st.write(f"Model type: `{type(model)}`")
    except Exception as e:
        st.error(f"Could not load model/scaler: {e}")
        st.stop()

# Predict button logic
if refresh:
    try:
        # prepare input (last 60 days), last_dates and last_closes
        X_input, last_dates, last_closes = prepare_last_window(df, scaler, window=WINDOW)

        # ---- Robust prediction handling for many model output types ----
        import numpy as np
        import tensorflow as tf

        # Obtain model raw output (could be np.array, tf.Tensor, dict, list, etc.)
        if hasattr(model, "predict"):
            raw_out = model.predict(X_input)
        else:
            # raw TF saved_model: try to call the signature if present
            infer = None
            if hasattr(model, "signatures"):
                infer = model.signatures.get("serving_default")
            if infer is None:
                # try to call model directly
                try:
                    raw_out = model(tf.convert_to_tensor(X_input.astype("float32")))
                except Exception as e_call:
                    raise RuntimeError(f"Cannot invoke saved_model: {e_call}")
            else:
                # call signature - find first input name and call
                input_name = list(infer.structured_input_signature[1].keys())[0]
                inp = tf.convert_to_tensor(X_input, dtype=tf.float32)
                raw_out = infer(**{input_name: inp})

        # Normalize raw_out to numpy array y_pred_scaled shape (n_samples, 1)
        y_pred_scaled = None
        # dict -> take first value
        if isinstance(raw_out, dict):
            first_val = list(raw_out.values())[0]
            if isinstance(first_val, tf.Tensor):
                y_pred_scaled = first_val.numpy()
            else:
                y_pred_scaled = np.array(first_val)
        elif isinstance(raw_out, tf.Tensor):
            y_pred_scaled = raw_out.numpy()
        elif isinstance(raw_out, (list, tuple)):
            y_pred_scaled = np.array(raw_out)
        elif isinstance(raw_out, np.ndarray):
            y_pred_scaled = raw_out
        else:
            try:
                y_pred_scaled = raw_out.numpy()
            except Exception:
                raise RuntimeError(f"Unsupported model output type: {type(raw_out)}")

        # Ensure shape (n_samples, 1)
        y_pred_scaled = np.asarray(y_pred_scaled)
        if y_pred_scaled.ndim == 0:
            y_pred_scaled = y_pred_scaled.reshape(1, 1)
        elif y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        elif y_pred_scaled.ndim >= 2 and y_pred_scaled.shape[-1] != 1:
            y_pred_scaled = y_pred_scaled.reshape(y_pred_scaled.shape[0], -1)[:, 0].reshape(-1, 1)

        # Inverse scale to original price
        try:
            y_pred_unscaled = scaler.inverse_transform(y_pred_scaled).flatten()
        except Exception as e:
            raise RuntimeError(f"Scaler inverse_transform failed. Error: {e}")

        # predicted scalar (first sample)
        predicted_price = float(y_pred_unscaled[0])

        # Show the metric
        st.metric(label="Predicted next-day Close", value=f"{predicted_price:.2f}")

        # Plot last WINDOW closes and predicted next-day point
        fig, ax = plt.subplots(figsize=(9,4))
        ax.plot(last_dates, last_closes, marker="o", label="Close (last 60 days)")
        next_day = last_dates[-1] + pd.Timedelta(days=1)
        ax.scatter([next_day], [predicted_price], color="red", label="Predicted next-day")
        ax.set_title(f"{ticker} Close — last {WINDOW} days + next-day prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Simple confidence estimate (heuristic based on recent return volatility)
        returns = pd.Series(last_closes).pct_change().dropna()
        if len(returns) > 0:
            sigma = returns.std()
            ci_low = predicted_price * (1 - 1.645 * sigma)
            ci_high = predicted_price * (1 + 1.645 * sigma)
            st.write(f"Approx. range (heuristic, 90%): {ci_low:.2f} — {ci_high:.2f} (std={sigma:.4f})")

    except Exception as e:
        st.exception(e)


# footer / instructions
st.markdown("---")
st.caption("TradeWise UI — uses model trained on 60-day windows. Retrain model & scaler after significant market moves for best accuracy.")
