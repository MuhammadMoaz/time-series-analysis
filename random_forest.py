import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from pathlib import Path

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

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        df['t'] = np.arange(len(df)) / len(df)

        df['Open_lag1'] = df['Open'].shift(1)
        df['High_lag1'] = df['High'].shift(1)
        df['Low_lag1'] = df['Low'].shift(1)
        df['Volume_lag1'] = df['Volume'].shift(1)

        feat_df = df[['Date','Close','t','Open_lag1','High_lag1','Low_lag1','Volume_lag1']].dropna().copy()

        X = feat_df[['t','Open_lag1','High_lag1','Low_lag1','Volume_lag1']].to_numpy()
        y = feat_df['Close'].to_numpy()
        dates = feat_df['Date'].to_numpy()

        train_size = int(len(feat_df)*0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = dates[train_size:]

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        # Model Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        print(f"{ticker} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape*100:.2f}%, R2: {r2:.3f}")

        plt.figure(figsize=(18, 9))
        plt.plot(dates, y, label="True")
        plt.plot(dates_test, y_pred, label="RandomForest Forecast", linestyle='--')
        plt.title(f"{ticker} Random Forest Forecast")
        plt.legend()

        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=70)

        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_RF.png")
        plt.clf()

main()