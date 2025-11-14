
# TradeWise – Stock Price Prediction

TradeWise is a lightweight LSTM-based system that predicts the next-day closing price of any stock using the previous 60 days of historical data.  
The project includes data preparation, model training, and a Streamlit UI for real-time predictions.

## Features
- LSTM model trained on 5+ years of stock data
- Uses a 60-day window for forecasting
- Real-time data via Yahoo Finance API
- Simple Streamlit interface for predictions

## Run the App
```bash
streamlit run streamlit_app.py
````

## Project Structure

```powershell
processed/      # scaler + prepared sequences
models/         # saved models (.h5 / saved_model / tf)
data/           # downloaded stock data
streamlit_app.py
train_lstm.py
prepare_sequences.py
resave_model.py
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Author

Ritesh Patil

