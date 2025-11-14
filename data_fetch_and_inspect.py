# data_fetch_and_inspect.py
import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

# 1) PARAMETERS — change ticker if you want
TICKER = "AAPL"          # start with one stock (Apple)
YEARS = 6                # at least 5 years; use 6 to be safe
CSV_DIR = "data"
os.makedirs(CSV_DIR, exist_ok=True)

# 2) DOWNLOAD
ticker = yf.Ticker(TICKER)
df = ticker.history(period=f"{YEARS}y", interval="1d", auto_adjust=True)
# keep only business days and drop NaNs
df = df.dropna()

# 3) SAVE CSV
csv_path = os.path.join(CSV_DIR, f"{TICKER}_{YEARS}y.csv")
df.to_csv(csv_path)
print(f"Saved {csv_path} — rows: {len(df)}")

# 4) QUICK INSPECTION
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head().to_string())

# 5) PLOT Close price (save figure)
plt.figure(figsize=(10,5))
plt.plot(df.index, df['Close'])
plt.title(f"{TICKER} Close Price — last {YEARS} years")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plot_path = os.path.join(CSV_DIR, f"{TICKER}_close_plot.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=150)
print(f"Saved plot to {plot_path}")
