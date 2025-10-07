import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
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

        # Feature engineering (example: lagged feature)
        for lag in range(1, 11):
            df[f"Lag_{lag}"] = df['Close'].shift(lag)
        df.dropna(inplace=True)

        # Define features (X) and target (y)
        X = df[[f'Lag_{lag}' for lag in range(1, 11)] + ['High', 'Low', 'Open', 'Volume']] 
        y = df['Close']

        # Split into training and testing (example: last 250 for testing)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Initialize and train the Random Forest Regressor
        rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Make predictions
        predictions = rf_regressor.predict(X_test)

        # Model evaluation
        rmse = root_mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f'{ticker} RMSE error for EMA averaging: {rmse:.5f}')
        print(f'{ticker} MAE error for EMA averaging: {mae}')
        print(f'{ticker} MAPE error for EMA averaging: {mape}')
        print(f'{ticker} R^2 score for EMA averaging: {r2}')

        plt.figure(figsize=(14, 6))
        plt.plot(df['Date'][:split_index], y_train, label='Actual', color='blue')
        plt.plot(df['Date'][split_index:], y_test, label='Actual Test', color='green') # Actual Test Data to compare with prediction
        plt.plot(df['Date'][split_index:], predictions, label='Predicted', color='red', linestyle='--')
        plt.title(f'Random Forest Regressor for {ticker} Time Series Forecasting')
        plt.xlabel('Date')
        plt.ylabel('Close')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_Random_Forest.png")
        plt.clf()

main()