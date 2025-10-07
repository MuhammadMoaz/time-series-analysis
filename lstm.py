import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error

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

        # # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date

        close_prices = df.loc[:,'Close'].to_numpy()

        train_size = int(len(close_prices) * 0.8)
        train_data = close_prices[:train_size].reshape(-1, 1)
        test_data = close_prices[train_size:].reshape(-1, 1)

        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data).reshape(-1)
        test_data = scaler.transform(test_data).reshape(-1)

        EMA = 0.0
        gamma = 0.1
        for ti in range(len(train_data)):
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
            train_data[ti] = EMA

        all_close_data = np.concatenate([train_data, test_data], axis=0)

        # EMA Averaging
        N = all_close_data.size
        train_size = int(len(all_close_data) * 0.8)

        run_avg_predictions = []
        mse_errors = []

        running_mean = 0.0
        decay = 0.5

        # Predict only on test set
        for pred_idx in range(train_size, N):
            running_mean = running_mean*decay + (1.0-decay)*all_close_data[pred_idx-1]
            run_avg_predictions.append(running_mean)
            mse_errors.append((run_avg_predictions[-1] - all_close_data[pred_idx])**2)

        mse = np.mean(mse_errors)
        rmse = np.sqrt(mse)

        # Convert predictions and true values back into numpy arrays for metrics
        y_true = all_close_data[train_size:]
        y_pred = np.array(run_avg_predictions)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f'{ticker} MSE error for EMA averaging (test only): %.5f' % (0.5*np.mean(mse_errors)))
        print(f'{ticker} RMSE error for EMA averaging (test only): {rmse:.5f}')
        print(f'{ticker} MAE error for EMA averaging: {mae}')
        print(f'{ticker} MAPE error for EMA averaging: {mape}')
        print(f'{ticker} R^2 score for EMA averaging: {r2}')

        plt.figure(figsize=(18,9))
        plt.title(f"{ticker} EMA Averaging Forecast")
        plt.plot(range(df.shape[0]), all_close_data, color='b', label='True')
        plt.plot(range(train_size, N), run_avg_predictions, color='orange', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_LSTM.png")
        plt.clf()

main()