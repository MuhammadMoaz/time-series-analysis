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

def summarise_data(data, file_path): #fix!!!
    ticker = get_ticker(file_path)
    data = pd.read_csv(file_path)

    result = adfuller(data["Close"].diff().dropna())
    station = ""

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        station = "Stationary"
    else:
        station = "Non-stationary"

    with open(f"EDAOutput/EDA_{ticker}/{ticker}.txt", 'w+') as f:
        f.write(f"{ticker} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}\n")
        f.write(f"ADF Statistic: {result[0]}. P-value: {result[1]}\n")
        f.write(f"The close price data is {station}")


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
def genLineSub():
    datasets = Path("datasets").rglob("*.csv")
    file_path = "EDAOutput/Grouped_LineGraphs.png"

    plt.figure(figsize=(20,10))

    for i, data in enumerate(datasets, 1):
        file_name = str(data)
        ticker = get_ticker(file_name)
        df = pd.read_csv(data)
        df["Date"] = pd.to_datetime(df["Date"])

        ax = plt.subplot(3,3,i)
       
        ax.plot(df["Date"], df["Close"])
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.set_ylabel("Close Price")
        # ax.set_xlabel("Date")
        plt.xticks(rotation = 70)
        ax.set_title(f"{ticker} Close Price")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def autoCorrACF(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    file_path = f"{dirName}/{ticker}_AutoCorr.png"

    plt.figure(figsize=(20,15))
    plot_acf(data["Close"].diff().dropna(), lags=25)#two years is 730 days
    # plt.ylim(0,1)
    plt.xlabel("Lags")
    plt.ylabel("Corr")
    plt.title(f"{ticker} Auto Correlation")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

def autoCorrPACF(data, ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    file_path = f"{dirName}/{ticker}_PartialAutoCorr.png"

    plt.figure(figsize=(20,15))
    plot_pacf(data["Close"], lags=50)#two years is 730 days
    # plt.ylim(0,1)
    plt.xlabel("Lags")
    plt.ylabel("Corr")
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

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)

        ticker = get_ticker(file_name)
        create_output_folder(file_name, ticker)
        # summarise_data(df, file_name)

        # genHistogram(df, ticker)
        # genCorrMatrix(df, ticker)
        # genLineGraph(df, ticker)
        # autoCorrACF(df, ticker)
        # autoCorrPACF(df, ticker)

    # genLineSub()
    # autoCorrMPL()
    movAveragePlot()
main()