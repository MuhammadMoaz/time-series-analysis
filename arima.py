import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import sarimax
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import os
import warnings # remove later

def set_df_size(df, days):
    if days == 0:
        return df
    else:
        return df.iloc[len(df)-days:].copy(deep = True)

def create_output_folder(ticker):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    os.makedirs(dirName, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def AICHelper(df, ticker):
    df = set_df_size(df, 0)
    df.dropna()

    train_size = int(len(df) * 0.8) #60 day of test
    # test train, dataframes 
    train_set = df.iloc[:train_size]

    lowest_AIC = 0
    best_p = 0
    best_q = 0

    for p in range(3):
        for d in range(3):
            for q in range(3):
                try:
                    model = ARIMA(train_set["Close"], order=(p, d, q))
                    result = model.fit()
                    print (ticker, p, d, q, ": ", result.aic)

                except:
                    print ("No AIC value")

    print (" ")
        
def CustomARIMAStats(df, ticker, p, d, q):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    image_file_path = f"{dirName}/{ticker}_model_metrics.png"
    file_path = f"{dirName}/{ticker}_model_summary.txt"
    df.dropna()
    df = set_df_size(df, 0)

    train_size = int(len(df) * 0.8)#80 / 20 train test split
    # test train, dataframes 
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]

    # creating model
    model = ARIMA(train_set["Close"], order=(p,d,q))
    res = model.fit()

    with open(file_path, 'w+') as f:
        f.write(str(res.summary()))

    # plotting residuals
    plt.title(f"{ticker} Residuals and Diagnostics")
    res.plot_diagnostics(figsize=(20,15), auto_ylims = True, lags = 25, 
                         acf_kwargs={'zero': False})
    plt.savefig(image_file_path)
    plt.clf()

    residuals = res.resid[1:]
    fig, (ax1, ax2) = plt.subplots(2,1)
    plot_acf(residuals, lags=25, auto_ylims=True, missing="drop", 
             zero=True, color="red", alpha=0.05, ax=ax1, title="ACF")
    plot_pacf(residuals, lags=25, alpha=0.05, zero=True, ax=ax2,
              auto_ylims=True, title="PACF")
    plt.savefig(f"{dirName}/{ticker}_Residual_ACF.png")
    plt.clf()

def ARIMAForcast(df, ticker, p,d,q):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    file_path = f"{dirName}/{ticker}_model_forecast.png"

    # change size / number of days to work with
    df = set_df_size(df, 0)

    train_size = int(len(df) * 0.8)
    # test train, dataframes 
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]

    model = ARIMA(train_set["Close"], order=(p,d,q))
    result = model.fit()

    forecast_ARIMA = result.get_forecast(len(test_set))
    pred_mean = forecast_ARIMA.predicted_mean
    forecast_CI = forecast_ARIMA.conf_int()

    df["forecasted"] = [None]*len(train_set) + list(pred_mean)
    df["Date"] = pd.to_datetime(df["Date"])
    test_set["Date"] = pd.to_datetime(test_set["Date"])

    # plotting forecast
    fig, (ax1) = plt.subplots(1,1, figsize=(20,10))
    ax1.plot(df["Date"], df["Close"], label="Actual")
    ax1.plot(df["Date"], df["forecasted"], label="Predicted")

    ax1.fill_between(test_set["Date"], forecast_CI['lower Close'], 
                 forecast_CI['upper Close'], alpha=0.5, label="95% CI")
    
    ax1.xaxis.set_major_locator(mdates.YearLocator()) # xaxis ticks
    ax1.set_ylabel("Close Price")
    plt.xticks(rotation = 70)
    plt.legend()

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

    # getting forecast metrics, adding them to txt file
    # print(list(pred_mean))
    rmse = np.sqrt(mean_squared_error(test_set["Close"], list(pred_mean))) 
    mape = mean_absolute_percentage_error(test_set["Close"], list(pred_mean))
    mae = mean_absolute_error(test_set["Close"], list(pred_mean))
    r2 = r2_score(test_set["Close"], list(pred_mean))

    with open(f"{dirName}/{ticker}_model_metrics.txt", 'w+') as f:
        f.write(f"The following are the metrics for {ticker}'s ARIMA model:\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAPE: {mape}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"R^2: {r2}\n")

def main():
    warnings.filterwarnings("ignore") # remove later
    datasets = Path("datasets").rglob("*.csv")
    dataset_list = []
    ticker_list = []

    for data in datasets:
        filename = str(data)
        ticker = get_ticker(str(data))
        create_output_folder(ticker)

        dataset_list.append(filename)
        ticker_list.append(ticker)

        df = pd.read_csv(filename)
        # AICHelper(df, ticker)


    # # # ARIMA 1 AMC 212
    CustomARIMAStats(pd.read_csv(dataset_list[0]),ticker_list[0],1,1,2)
    ARIMAForcast(pd.read_csv(dataset_list[0]),ticker_list[0],1,1,2)
    # ARIMA 2 BHP
    CustomARIMAStats(pd.read_csv(dataset_list[1]),ticker_list[1],2,1,2)
    ARIMAForcast(pd.read_csv(dataset_list[1]),ticker_list[1],2,1,2)
    # ARIMA 3 CBA
    CustomARIMAStats(pd.read_csv(dataset_list[2]),ticker_list[2],2,1,2)
    ARIMAForcast(pd.read_csv(dataset_list[2]),ticker_list[2],2,1,2)
    # ARIMA 4 CSL
    CustomARIMAStats(pd.read_csv(dataset_list[3]),ticker_list[3],1,2,2)
    ARIMAForcast(pd.read_csv(dataset_list[3]),ticker_list[3],1,2,2)
    # ARIMA 5 NAB
    CustomARIMAStats(pd.read_csv(dataset_list[4]),ticker_list[4],1,1,2)
    ARIMAForcast(pd.read_csv(dataset_list[4]),ticker_list[4],1,1,2)
    # ARIMA 6 PME
    CustomARIMAStats(pd.read_csv(dataset_list[5]),ticker_list[5],1,2,2)
    ARIMAForcast(pd.read_csv(dataset_list[5]),ticker_list[5],1,2,2)
    # ARIMA 7 RIO
    CustomARIMAStats(pd.read_csv(dataset_list[6]),ticker_list[6],2,1,2)
    ARIMAForcast(pd.read_csv(dataset_list[6]),ticker_list[6],0,1,1)
    # ARIMA 8 RMD
    CustomARIMAStats(pd.read_csv(dataset_list[7]),ticker_list[7],2,1,2)   
    ARIMAForcast(pd.read_csv(dataset_list[7]),ticker_list[7],2,1,2)
    # ARIMA 9 WBC
    CustomARIMAStats(pd.read_csv(dataset_list[8]),ticker_list[8],2,1,1)
    ARIMAForcast(pd.read_csv(dataset_list[8]),ticker_list[8],2,1,1)


main()