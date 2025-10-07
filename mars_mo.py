import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from pyearth import Earth

def create_output_folder(file_name, ticker):
    dir_name = f"PDAOutput/PDA_{ticker}"
    os.makedirs(dir_name, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def main():
    datasets = Path("datasets").rglob("*.csv")

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)
        ticker = get_ticker(file_name)
        create_output_folder(file_name, ticker)

        # Sort dates
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date

        df['t'] = np.arange(len(df)) / len(df)

        # Train/test splitting
        train_size = int(len(df)*0.8)

        df['Open_lag1'] = df['Open'].shift(1)
        df['High_lag1'] = df['High'].shift(1)
        df['Low_lag1']  = df['Low'].shift(1)
        df['Volume_lag1'] = df['Volume'].shift(1)

        feat_df = df.dropna().copy()

        X = feat_df[['t','Open_lag1','High_lag1','Low_lag1','Volume_lag1']].to_numpy()
        y = feat_df['Close'].to_numpy()
        dates = feat_df['Date'].to_numpy()

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = dates[train_size:]

        # Fit model
        mars = Earth(max_degree=2)
        mars.fit(X_train, y_train)

        # Predict
        y_pred = mars.predict(X_test)

        # Model Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"{ticker} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape*100:.2f}%, R2: {r2:.3f}")

        plt.figure(figsize=(12, 6))
        plt.plot(dates, y, label="True")
        plt.plot(dates_test, y_pred, label="MARS Forecast", linestyle="--")
        plt.title(f"{ticker} MARS forecast")
        plt.legend()
        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_MARS.png")
        plt.clf()

main()