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

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]


def CustomARIMA(df, ticker, p, d, q):
    df.drop(index=0)
    X_train, X_test, y_train, y_test = train_test_split(df["Date"], df["Close"], train_size=len(df)-30, random_state=42)

    model = ARIMA(y_train, order=(p,d,q))
    res = model.fit()
    print(res.summary())



def main():
    datasets = Path("datasets").rglob("*.csv")
    dataset_list = []
    ticker_list = []
    for data in datasets:
        dataset_list.append(str(data))
        ticker_list.append(get_ticker(str(data)))


    df = pd.read_csv(dataset_list[0])

    # ARIMA 1
    CustomARIMA(pd.read_csv(dataset_list[0]),ticker_list[0],1,1,0)
    # ARIMA 2
    CustomARIMA(pd.read_csv(dataset_list[1]),ticker_list[1],1,1,0)
    # ARIMA 3
    CustomARIMA(pd.read_csv(dataset_list[2]),ticker_list[2],1,1,0)
    # ARIMA 4
    CustomARIMA(pd.read_csv(dataset_list[3]),ticker_list[3],1,1,0)
    # ARIMA 5
    CustomARIMA(pd.read_csv(dataset_list[4]),ticker_list[4],1,1,0)
    # ARIMA 6
    CustomARIMA(pd.read_csv(dataset_list[5]),ticker_list[5],1,1,0)
    # ARIMA 7
    CustomARIMA(pd.read_csv(dataset_list[6]),ticker_list[6],1,1,0)
    # ARIMA 8
    CustomARIMA(pd.read_csv(dataset_list[7]),ticker_list[7],1,1,0)   
    # ARIMA 9
    CustomARIMA(pd.read_csv(dataset_list[8]),ticker_list[8],1,1,0)


main()