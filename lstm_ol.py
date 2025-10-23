import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping

# -------- Config --------
SEQ_LEN = 60          # lookback window (days)
EPOCHS = 20
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8     # 80/20 split
METRICS_CSV = "metrics.csv"

# (Optional) reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_output_folder(ticker: str):
    out_dir = f"PDAOutput/PDA_{ticker}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def get_ticker(file_path: str) -> str:
    return Path(file_path).stem

def make_sequences(arr: np.ndarray, seq_len: int):
    """Build (X, y) sequences from a scaled 1D series shaped (n, 1)."""
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len:i])
        y.append(arr[i])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_and_evaluate(file_path: str):
    # ----- Load & prep -----
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    close = df["Close"].values.reshape(-1, 1)

    split_idx = int(len(close) * TRAIN_SPLIT)
    train_raw, test_raw = close[:split_idx], close[split_idx:]

    # scale on train only (avoid leakage)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    # Build sequences
    X_train, y_train = make_sequences(train_scaled, SEQ_LEN)

    # For test, create sequences over the *full* scaled array, then slice
    all_scaled = np.vstack([train_scaled, test_scaled])
    X_all, y_all = make_sequences(all_scaled, SEQ_LEN)

    # Test sequences start SEQ_LEN before the split boundary
    test_start = split_idx - SEQ_LEN
    X_test, y_test = X_all[test_start:], y_all[test_start:]

    # ----- Model -----
    model = build_lstm(input_shape=(SEQ_LEN, 1))
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Small validation split from train sequences
    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es]
    )

    # Predict (scaled)
    y_pred_scaled = model.predict(X_test, verbose=0)

    # Inverse transform to price space
    y_true = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Metrics in price units
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)

    return df, split_idx, y_true.squeeze(), y_pred.squeeze(), mae, mape, rmse, r2

def plot_results(df, split_idx, y_pred, out_png_path: str):
    """Plot true close vs predictions aligned to test dates."""
    # True prices
    close = df["Close"].values
    dates = df["Date"]

    # Test window dates
    test_dates = dates.iloc[split_idx:].reset_index(drop=True)
    # Predictions begin SEQ_LEN steps into the test window
    pred_dates = test_dates.iloc[SEQ_LEN:].reset_index(drop=True)

    plt.figure(figsize=(18, 9))
    plt.title("LSTM Forecast")
    plt.plot(dates, close, label="True (Close)")
    plt.plot(pred_dates, y_pred, label="Prediction")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=70)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_path)
    plt.clf()

def main():
    datasets = Path("datasets").rglob("*.csv")

    for csv_path in datasets:
        ticker = get_ticker(str(csv_path))
        out_dir = create_output_folder(ticker)

        try:
            df, split_idx, y_true, y_pred, mae, mape, rmse, r2 = train_and_evaluate(str(csv_path))
        except Exception as e:
            print(f"[{ticker}] Error: {e}")
            continue

        # Save metrics row (append)
        row = pd.DataFrame([{
            "Ticker": ticker,
            "Model": "LSTM",
            "MAE": mae,
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2
        }])
        row.to_csv(METRICS_CSV, mode="a", header=not os.path.exists(METRICS_CSV), index=False)

        # Plot
        plot_path = f"{out_dir}/{ticker}_LSTM.png"
        plot_results(df, split_idx, y_pred, plot_path)

        print(f"{ticker} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape*100:.2f}%, R2: {r2:.3f} -> {plot_path}")

if __name__ == "__main__":
    main()