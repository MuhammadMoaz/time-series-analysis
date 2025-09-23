import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
from pathlib import Path

def create_output_folder(file_name, ticker):
    dir_name = f"PDAOutput/PDA_{ticker}"
    os.makedirs(dir_name, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

# Create training dataset
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def main():
    datasets = Path("datasets").rglob("*.csv")

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)
        ticker = get_ticker(file_name)
        create_output_folder(file_name, ticker)

        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Use only Close prices for forecasting
        data = df[['Close']].values

        # normalised data between range of 0 and 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        time_step = 60  # 60 days lookback
        X, y = create_dataset(scaled_data, time_step)

        # Reshape input to [samples, time_steps, features] for LSTM
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build the LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            Dropout(0.2), # Randomly drops 20% of neurons, prevents overfitting
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25), # hideen fully connected layer for non-linear transformation.
            Dense(1) # output layer
        ])

        model.compile(optimizer='adam', loss='mean_squared_error') # adam = adaptive learning
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

        # Make predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Visualization
        train_dates = df.index[time_step:train_size+time_step]
        test_dates = df.index[train_size+time_step+1:]

        plt.figure(figsize=(12,6))
        plt.plot(df.index, df['Close'], label="Actual Close Price", color="blue")
        plt.plot(test_dates, test_predict, label="Test Predictions", color="red", linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"{ticker} Stock Close Price Forecast with LSTM")
        plt.legend()
        plt.grid()
        plt.show()
        
main()