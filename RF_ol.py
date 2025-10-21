import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


def create_output_folder(ticker):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    os.makedirs(dirName, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def series_transform(data, in_n = 0, out_n = 0):
    var_num = 1
    colmns = list()
    df = pd.DataFrame(data)

    for i in range(in_n, 0, -1):
        colmns.append(df.shift(i))

    for i in range(0, out_n):
        colmns.append(df.shift(-i))

    aggregated = pd.concat(colmns, axis = 1)
    aggregated.dropna()
    return aggregated.values

def rf_forecast(train_set, testX):
    train_set = np.asarray(train_set)
    trainX, trainy = train_set[:, :-1], train_set[:, -1]
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict([testX])
    return yhat[0]

def test_splitter(data, size_n):
    return data[:-size_n, :], data[-size_n:, :]

def walk_forward(data):
    pred = list()
    train_size = int(len(data) * 0.2)#80 / 20 train test split
    #train_size = 1220#80 / 20 train test split
    train_set, test_set = test_splitter(data, train_size)
    num = 0

    historical = [x for x in train_set]
    for i in range(len(test_set)):
        # I/O columns of a supervised Learning problem set
        testX, testy = test_set[i, :-1], test_set[i, -1]
        #predicted 
        yhat = rf_forecast(historical, testX)
        pred.append(yhat)
        historical.append(test_set[i])
        num += 1
        print(f"num: {num}, expected: {testy}, predicted: {yhat}")
    #metrics
    return test_set[:,-1], pred

def rf_model(data, ticker):
        df = pd.read_csv(data)
        series = df["Close"].copy()
        # act_set = series.iloc[len(series)-10:]
        act_set = series
        data = series_transform(act_set, in_n=1, out_n=1)
        y, yhat = walk_forward(data)

        rmse = root_mean_squared_error(y, yhat)
        mape = mean_absolute_percentage_error(y, yhat)
        mae = mean_absolute_error(y, yhat)
        r2 = r2_score(y, yhat)

        print(f"{ticker} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, MAPE: {mape*100:.2f}%, R2: {r2:.3f}")
 
        results = {
            "Ticker": ticker, 
            "Model": "RF", 
            "MAE": mae, 
            "MAPE": mape, 
            "RMSE": rmse, 
            "R2": r2
        }

        results_df = pd.DataFrame([results])
        results_df.to_csv("metrics.csv", mode='a+', header=not os.path.exists("metrics.csv"), index=False)

        fig, (ax1) = plt.subplots(1,1, figsize=(20,10))
        ax1.plot(df["Date"], df["Close"], label="Actual")
        ax1.plot(df["Date"].iloc[len(df) - len(yhat):], yhat, label="Predicted")
        
        ax1.xaxis.set_major_locator(mdates.YearLocator()) # xaxis ticks
        ax1.set_ylabel("Close Price")
        plt.xticks(rotation = 70)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_RFFinal.png")
        plt.clf()

def main():
    datasets = Path("datasets").rglob("*.csv")
    dataset_list = []
    ticker_list = []

    for data in datasets:
        filename = str(data)
        ticker = get_ticker(str(data))
        create_output_folder(ticker)

        dataset_list.append(filename)
        ticker_list.append(ticker)
       
    # rf_model(dataset_list[0], ticker_list[0])
    # rf_model(dataset_list[1], ticker_list[1])
    # rf_model(dataset_list[2], ticker_list[2])
    # rf_model(dataset_list[3], ticker_list[3])
    # rf_model(dataset_list[4], ticker_list[4])
    # rf_model(dataset_list[5], ticker_list[5])
    # rf_model(dataset_list[6], ticker_list[6])
    rf_model(dataset_list[7], ticker_list[7])
    # rf_model(dataset_list[8], ticker_list[8])
    
    
main()