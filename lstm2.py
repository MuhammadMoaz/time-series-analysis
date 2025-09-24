import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
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

        close_prices = df.loc[:,'Close'].values

        train_size = int(len(close_prices) * 0.8)
        train_data = close_prices[:train_size]
        test_data = close_prices[train_size:]

        scaler = MinMaxScaler()
        train_data = train_data.reshape(-1, 1)
        test_data - test_data.reshape(-1, 1)

        smoothing_window_size = 1000
        for di in range(0, 10000, smoothing_window_size):
            scaler.fit(train_data[di:di+smoothing_window_size, :])
            train_data[di:di+smoothing_window_size, :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

        scaler.fit(train_data[di+smoothing_window_size:, :])
        train_data[di+smoothing_window_size:, :] = scaler.transform(train_data[di+smoothing_window_size:, :])

        train_data = train_data.reshape(-1)
        test_data = scaler.transform(test_data).reshape(-1)

        EMA = 0.0
        gamma = 0.1

        for ti in range(6102):
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
            train_data[ti] = EMA

        all_close_data = np.concatenate([train_data, test_data], axis=0)

        window_size = 100
        N = train_data.size
        std_avg_predictions = []
        std_avg_x = []
        mse_errors = []

        for pred_idx in range(window_size,N):

            if pred_idx >= N:
                date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
            else:
                date = df.loc[pred_idx,'Date']

            std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
            mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
            std_avg_x.append(date)

        print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))
        
        plt.figure(figsize = (18,9))
        plt.plot(range(df.shape[0]),all_close_data,color='b',label='True')
        plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')
        #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
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