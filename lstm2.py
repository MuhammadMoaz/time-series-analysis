import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

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

        # Standard Averaging

        window_size = 100
        train_size = int(len(all_close_data) * 0.8)
        N = all_close_data.size

        std_avg_predictions = []
        mse_errors = []

        # Start predicting only from the test set
        for pred_idx in range(train_size, N):
            std_avg_predictions.append(np.mean(all_close_data[pred_idx-window_size:pred_idx]))
            mse_errors.append((std_avg_predictions[-1] - all_close_data[pred_idx])**2)

        mse = np.mean(mse_errors)
        rmse = np.sqrt(mse)

        print(f'{ticker} MSE error for standard averaging (test only): %.5f' % (0.5*np.mean(mse_errors)))
        print(f'{ticker} RMSE error for standard averaging (test only): {rmse:.5f}')

        plt.figure(figsize=(18,9))
        plt.title(f"{ticker} Standard Averaging Forecast")
        plt.plot(range(df.shape[0]), all_close_data, color='b', label='True')
        plt.plot(range(train_size, N), std_avg_predictions, color='orange', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.show()

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

        print(f'{ticker} MSE error for EMA averaging (test only): %.5f' % (0.5*np.mean(mse_errors)))
        print(f'{ticker} RMSE error for EMA averaging (test only): {rmse:.5f}')

        plt.figure(figsize=(18,9))
        plt.title(f"{ticker} EMA Averaging Forecast")
        plt.plot(range(df.shape[0]), all_close_data, color='b', label='True')
        plt.plot(range(train_size, N), run_avg_predictions, color='orange', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Mid Price')
        plt.legend(fontsize=18)
        plt.show()

        plt.figure(figsize = (18,9))
        plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
        plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Mid Price',fontsize=18)
        plt.show()

main()