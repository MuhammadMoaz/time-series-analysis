import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    r2_score, root_mean_squared_error
)

# -------------------
# Feature Engineering
# -------------------
def make_features(df, max_lag=5, roll_windows=(5, 10)):
    out = df.copy()
    # Lagged closes
    for lag in range(1, max_lag + 1):
        out[f'Close_lag{lag}'] = out['Close'].shift(lag)
    # Lagged OHLCV
    for col in ["Open", "High", "Low", "Volume"]:
        out[f"{col}_lag1"] = out[col].shift(1)
    # Rolling averages of Close
    for w in roll_windows:
        out[f'Close_roll{w}'] = out['Close'].shift(1).rolling(w).mean()
    # Keep Date and Close explicitly
    return out[['Date','Close'] + [c for c in out.columns if 'lag' in c or 'roll' in c]].dropna().copy()

# -------------------
# Main
# -------------------
def main():
    datasets = Path("datasets").rglob("*.csv")

    for data in datasets:
        file_name = str(data)
        ticker = file_name.replace("datasets\\","").replace("datasets/","").split('.')[0]

        # Load
        df = pd.read_csv(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Build features
        feat_df = make_features(df, max_lag=5, roll_windows=(5,10))

        # Split features/target/dates
        feature_cols = [c for c in feat_df.columns if c not in ['Date','Close']]
        X = feat_df[feature_cols].to_numpy()
        y = feat_df['Close'].to_numpy()
        dates = feat_df['Date'].to_numpy()

        # Chronological split
        train_size = int(len(feat_df)*0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = dates[train_size:]

        # Train RF
        rf = RandomForestRegressor(
            n_estimators=300, max_depth=None,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Predict
        y_pred = rf.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        print(f"\n=== {ticker} Random Forest Results ===")
        print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape:.2f}%, R²: {r2:.3f}")

        # Plot
        plt.figure(figsize=(14,6))
        plt.plot(dates, y, label="True")
        plt.plot(dates_test, y_pred, label="RF Forecast", linestyle="--")
        plt.title(f"{ticker} Random Forest Forecast")
        plt.legend()

        # Force yearly ticks
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=45)

        os.makedirs(f"PDAOutput/PDA_{ticker}", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_RF.png")
        plt.close()

if __name__ == "__main__":
    main()
