import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import os
from pathlib import Path

'''
This File takes csv files in as input and performs an EDA on them, 
it then outputs data into folders with appropriate titling, these 
folders will contain graphs of different types to visualise data
'''

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def create_output_folder(file_name, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    os.makedirs(dirName, exist_ok=True)

# basic Statistics 
def summarise_data(data, file_path):
    ticker = get_ticker(file_path)
    data = pd.read_csv(file_path)

    # to see results for price ADF test, do data["Close"]
    resultClose = adfuller(data["Close"])
    resultLR = adfuller(data["Log_Returns"].dropna()) # drops first row with NaN val

    stationClose = ""
    stationLR = ""

    if (resultClose[1] <= 0.05) & (resultClose[4]['5%'] > resultClose[0]):
        stationClose = "Stationary"
    else:
        stationClose = "Non-stationary"

    if (resultLR[1] <= 0.05) & (resultLR[4]['5%'] > resultLR[0]):
        stationLR = "Stationary"
    else:
        stationLR = "Non-stationary"

    

    with open(f"EDAOutput/EDA_{ticker}/{ticker}.txt", 'w+') as f:
        f.write(f"{ticker} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}\n\n")

        # Augmented Dickey Fuller for both Close price and Close LR
        f.write(f"ADF Statistic: {resultClose[0]}. P-value: {resultClose[1]}\n")
        f.write(f"The Close Price data is {stationClose}\n\n")

        f.write(f"ADF Statistic: {resultLR[0]}. P-value: {resultLR[1]}\n")
        f.write(f"The Log Return Close Price data is {stationLR}\n\n")

def getAvgClose(data, ticker):
    avg_close = round(data['Close'].mean(), 2)

    with open(f"EDAOutput/EDA_{ticker}/{ticker}.txt", 'a') as f:
        f.write(f"Average Close Price: {avg_close}")

# visualisation 
def genLineSub(var_name, file_name):
    datasets = Path("datasets").rglob("*.csv")
    file_path = f"EDAOutput/{file_name}.png"

    plt.figure(figsize=(20,10))

    for i, data in enumerate(datasets, 1):
        file_name = str(data)
        ticker = get_ticker(file_name)
        df = pd.read_csv(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df.drop(index=0)

        ax = plt.subplot(3,3,i)
       
        ax.plot(df["Date"], df[var_name])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylabel(var_name)
        # ax.set_xlabel("Date")
        plt.xticks(rotation = 70)
        ax.set_title(f"{ticker} {var_name} Close Price")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()


# Visualisation 1 - Histogram
def genHistogram(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    
    columns = data.columns.tolist()
    columns.remove("Date")
    
    for var in columns:
        file_path = f"{dirName}/{ticker}_{var}_Hist.png"
        
        if os.path.exists(file_path):
            continue
        else:
            plt.hist(data[var], color='purple')
            plt.title(f'Distribution of {var} Price')
            plt.xlabel(f'{var} Price')
            plt.ylabel('Count')
            plt.savefig(file_path)
            plt.clf()

# Visualisation 2 - Correlation Matrix
def genCorrMatrix(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    file_path = f"{dirName}/{ticker}_corr.png"

    columns = data.columns.tolist()
    columns.remove('Date')
    col_data = data[columns]
    corr_matrix = col_data.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.savefig(file_path)
    plt.clf()
# Visualisation 3 - Line Graph
def genLineGraph(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"

    file_path = f"{dirName}/{ticker}_Line.png"

    x_val = data['Date']
    y_val = data['Close']

    plt.plot(x_val, y_val)

    plt.xlabel("X-Axis Label")
    plt.ylabel("Y-Axis Label")
    plt.title("Line Graph")

    plt.savefig(file_path)
    plt.clf()
# Visualisation X - gen line subplots


def autoCorrACF(data, ticker, var_name):
    dirName = f"EDAOutput/EDA_{ticker}"
    file_path = f"{dirName}/{ticker}_{var_name}_AutoCorr.png"

    inc_zero = True
    if var_name == "Log_Returns":
        inc_zero = False

    # plt.figure(figsize=(20,15))
    plot_acf(data[var_name], lags=50, auto_ylims=True, missing="drop", 
             zero=inc_zero, color="red", alpha=0.05)
    plt.xlabel("Lags")
    plt.ylabel("Auto Correlation")
    plt.title(f"{ticker} {var_name} Auto Correlation")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def autoCorrPACF(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    file_path = f"{dirName}/{ticker}_ReturnsPartialAutoCorr.png"

    plt.figure(figsize=(20,15))
    plot_pacf(data["Log_Returns"].dropna(), lags=50, alpha=0.05, zero=False)
    plt.ylim(-0.5,0.5)
    plt.xlabel("Lags")
    plt.ylabel("Partial Auto Correlation")
    plt.title(f"{ticker} PACF")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def autoCorrMPL():
    datasets = Path("datasets").rglob("*.csv")
    file_path = "EDAOutput/Grouped_AutoCorr.png"

    plt.figure(figsize=(20,10))

    for i, data in enumerate(datasets, 1):
        file_name = str(data)
        ticker = get_ticker(file_name)
        df = pd.read_csv(data)
        df["Date"] = pd.to_datetime(df["Date"])

        ax = plt.subplot(3,3,i)
        ax.acorr(df["Close"], usevlines=True, normed=True, maxlags=250, lw=2)
        ax.set_title(f"{ticker} AutoCorr")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()
    
# moving average, subplots (mean function for dataframes)
def movAveragePlot():
    datasets = Path("datasets").rglob("*.csv")
    file_path = "EDAOutput/MA_lineGraph.png"
    plt.figure(figsize=(20,10))

    for i, data in enumerate(datasets, 1):
        file_name = str(data)
        ticker = get_ticker(file_name)
        df = pd.read_csv(data)
        df["Date"] = pd.to_datetime(df["Date"])
        df['Moving_Average'] = df['Close'].rolling(window=100).mean() 
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df.drop(df.index[0])  
        
        ax = plt.subplot(3,3,i)

        ax.plot(df["Date"], df["Log_Returns"], color = "blue", lw=1, label="Close")
        # ax.plot(df["Date"], df["Moving_Average"], color = "red", linestyle="--", lw=2, label = "MA")
        
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylabel("MA Close Price")
        # ax.set_xlabel("Date")
        plt.xticks(rotation = 70)
        ax.legend()
        ax.set_title(f"{ticker} MA Close Price")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()


def main():
    datasets = Path("datasets").rglob("*.csv")

    # plot line graphs for Close and LR --
    genLineSub("Close", "Group_ClosePLineGraphs")
    genLineSub("Log_Returns", "Group_LogReturnsLineGraphs")

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)

        ticker = get_ticker(file_name)
        # create_output_folder(file_name, ticker)

        # get basic statistics of data --
        summarise_data(df, file_name)
        getAvgClose(df, ticker)

        # Get Auto Correlation Graphs of Close Price
        autoCorrACF(df, ticker, "Close")

        # Get Auto Correlation Graphs of Log Returns Close Price
        autoCorrACF(df, ticker, "Log_Returns")
        autoCorrPACF(df, ticker)

  
    # autoCorrMPL()
    # movAveragePlot()
main()