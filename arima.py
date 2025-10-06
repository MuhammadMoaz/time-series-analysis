import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import train_test_split
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error
import os


def create_output_folder(ticker):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    os.makedirs(dirName, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def AICHelper(df, ticker):
    df.dropna()

    train_size = int(len(df)*0.8) #80 / 20 train test split
    # test train, dataframes 
    train_set = df.iloc[:train_size]

    lowest_AIC = 0
    best_p = 0
    best_q = 0

    for p in range(6):
        for q in range(6):
            try:
                model = ARIMA(train_set["Close"], order=(p, 1, q))
                result = model.fit()

                if p == 0 and q == 0:
                    lowest_AIC = result.aic
                elif result.aic < lowest_AIC:
                    lowest_AIC = result.aic
                    best_p = p
                    best_q = q

            except:
                print ("No AIC value")

    print (ticker, best_p, 1, best_q, ": ", result.aic)
        
def CustomARIMA(df, ticker, p, d, q):
    dirName = f"ARIMAOutput/ARIMA_{ticker}"
    image_file_path = f"{dirName}/{ticker}_model_metrics.png"
    file_path = f"{dirName}/{ticker}_model_summary.txt"
    df.dropna()

    train_size = int(len(df)*0.8) #80 / 20 train test split
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
             zero=False, color="red", alpha=0.05, ax=ax1, title="ACF")
    plot_pacf(residuals, lags=25, alpha=0.05, zero=False, ax=ax2,
              auto_ylims=True, title="PACF")
    plt.savefig(f"{dirName}/{ticker}_Residual_ACF.png")
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

        df = pd.read_csv(filename)
        # AICHelper(df, ticker)


    # ARIMA 1 AMC
    CustomARIMA(pd.read_csv(dataset_list[0]),ticker_list[0],1,1,1)
    # ARIMA 2 BHP
    CustomARIMA(pd.read_csv(dataset_list[1]),ticker_list[1],1,1,0)
    # ARIMA 3 CBA
    CustomARIMA(pd.read_csv(dataset_list[2]),ticker_list[2],1,1,0)
    # ARIMA 4 CSL
    CustomARIMA(pd.read_csv(dataset_list[3]),ticker_list[3],1,1,0)
    # ARIMA 5 NAB
    CustomARIMA(pd.read_csv(dataset_list[4]),ticker_list[4],1,1,0)
    # ARIMA 6 PME
    CustomARIMA(pd.read_csv(dataset_list[5]),ticker_list[5],1,1,0)
    # ARIMA 7 RIO
    CustomARIMA(pd.read_csv(dataset_list[6]),ticker_list[6],1,1,0)
    # ARIMA 8 RMD
    CustomARIMA(pd.read_csv(dataset_list[7]),ticker_list[7],1,1,0)   
    # ARIMA 9 WBC
    CustomARIMA(pd.read_csv(dataset_list[8]),ticker_list[8],1,1,0)


main()