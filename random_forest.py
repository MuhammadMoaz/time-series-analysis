import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
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
        df = df.set_index('Date').sort_index()

        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Close_lag3'] = df['Close'].shift(3)

        # Set features (X) and target (y)
        X = df[['Close_lag1', 'Close_lag2', 'Close_lag3', 'High', 'Low', 'Open', 'Volume']]
        y = df['Close'].loc[X.index]

        # Split Data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # Initialise RandomForest Model
        rfr = RandomForestRegressor(n_estimators=100, random_state=1)

        # Fit model to training data
        rfr.fit(X_train, y_train)

        # Recursively predicting on test data
        rfr_preds = []
        last_known = X_test.iloc[0].copy()

        for i in range(len(X_test)):
            rfr_pred = rfr.predict(last_known.to_frame().T)[0]
            rfr_preds.append(rfr_pred)

            # Shift lag values
            last_known['Close_lag3'] = last_known['Close_lag2']
            last_known['Close_lag2'] = last_known['Close_lag1']
            last_known['Close_lag1'] = rfr_pred

            # updating non lag features with actual test values
            if i+1 < len(X_test):
                next_row = X_test.iloc[i+1]
                last_known['High'] = next_row['High']
                last_known['Low'] = next_row['Low']
                last_known['Open'] = next_row['Open']
                last_known['Volume'] = next_row['Volume']

        rfr_preds = pd.Series(rfr_preds, index=y_test.index)

        # Model Performance Evaluation
        rmse = root_mean_squared_error(y_test, rfr_preds)
        print(f"{ticker} RMSE: {rmse}")

        # FUTURE DATA FORECASTING (OPTIONAL) JUST MESSING AROUND
        future_steps = 365
        future_preds = []
        last_known = X.iloc[-1].copy()
        last_date = df.index[-1]

        for i in range(future_steps):
            pred = rfr.predict(last_known.to_frame().T)[0]
            future_preds.append(pred)

            # roll lag values
            last_known['Close_lag3'] = last_known['Close_lag2']
            last_known['Close_lag2'] = last_known['Close_lag1']
            last_known['Close_lag1'] = pred

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
        future_preds = pd.Series(future_preds, index=future_dates)
        # FUTURE DATA FORECASTING ENDS

        # Visualise
        plt.figure(figsize=(12, 6))
        
        plt.plot(y_test.index, y_test, label='Actual Data', color='blue')
        plt.plot(y_test.index, rfr_preds, label='Predicted Data', color='red', linestyle='--')
        plt.plot(future_preds.index, future_preds, label='Forecast (Future)', color='green', linestyle='--') # FUTURE DATA FORECASTING (OPTIONAL)
        
        plt.title('Random Forest Time Series Forecast vs. Actual')
        plt.legend()    
        plt.grid(True)
        plt.xlabel('Date')
        plt.ylabel('Value')
        
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_Random_Forest.png")
        plt.clf()

main()