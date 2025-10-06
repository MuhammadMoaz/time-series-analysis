import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

def create_output_folder(ticker):
    dirName = f"EDAOutput/EDA_{ticker}"
    os.makedirs(dirName, exist_ok=True)

def set_df_size(df, days):
    if days == 0:
        return df
    else:
        return df.iloc[len(df)-days:].copy(deep = True)

# basic Statistics 
def summarise_data(file_path, train_split = False, datasize = 0):
    ticker = get_ticker(file_path)
    data = pd.read_csv(file_path)
    data = set_df_size(data, datasize)
    ticker_ext = ticker

    train_size = int(len(data)*0.8)
    if train_split:
        data = data.iloc[:train_size]
        ticker_ext = f"{ticker}TrainSplit"

    # to see results for price ADF test, do data["Close"]
    resultClose = adfuller(data["Close"])
    resultLR = adfuller(data["Log_Returns"].dropna()) # drops first row with NaN val
    resultDiff = adfuller(data["Differenced"].dropna()) # drops first row with NaN val

    stationClose = "Non-stationary"
    stationLR = "Non-stationary"
    stationDiff = "Non-stationary"

    if (resultClose[1] <= 0.05) & (resultClose[4]['5%'] > resultClose[0]):
        stationClose = "Stationary"

    if (resultLR[1] <= 0.05) & (resultLR[4]['5%'] > resultLR[0]):
        stationLR = "Stationary"

    if (resultDiff[1] <= 0.05) & (resultDiff[4]['5%'] > resultDiff[0]):
        stationDiff = "Stationary"

    avg_close = round(data['Close'].mean(), 2)

    with open(f"EDAOutput/EDA_{ticker}/{ticker_ext}.txt", 'w+') as f:
        f.write(f"{ticker} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}\n")
        f.write(f"Average Close Price: {avg_close}\n\n")

        # Augmented Dickey Fuller for both Close price and Close LR
        f.write(f"ADF Statistic: {resultClose[0]}. P-value: {resultClose[1]}\n")
        f.write(f"The Close Price data is {stationClose} with a critical value of 5%\n\n")

        f.write(f"ADF Statistic: {resultLR[0]}. P-value: {resultLR[1]}\n")
        f.write(f"The Log Return Close Price data is {stationLR} with a critical value of 5%\n\n")

        f.write(f"ADF Statistic: {resultDiff[0]}. P-value: {resultDiff[1]}\n")
        f.write(f"The Differened Close Price data is {stationDiff} with a critical value of 5%\n\n")

# visualisation 
def genLineSub(var_name, file_name, t_scale = 'y', datasize = 0):
    datasets = Path("datasets").rglob("*.csv")
    file_path = f"EDAOutput/{file_name}.png"

    plt.figure(figsize=(20,10))

    for i, data in enumerate(datasets, 1):
        file_name = str(data)
        ticker = get_ticker(file_name)
        df = pd.read_csv(data)
        sized_df = set_df_size(df, datasize)
        sized_df["Date"] = pd.to_datetime(sized_df["Date"])
        sized_df.dropna()

        ax = plt.subplot(3,3,i)
       
        ax.plot(sized_df["Date"], sized_df[var_name])
        if t_scale == 'm':
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            
        ax.set_ylabel(var_name)
        # ax.set_xlabel("Date")
        plt.xticks(rotation = 70)
        ax.set_title(f"{ticker} {var_name} Close Price")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()

# Visualisation 1 - Histogram
def genHistogram(var_name, file_name):
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
       
        ax.hist(df[var_name], color="blue")
        ax.set_title(f"{ticker} Histogram of {var_name} Close Price")

    plt.tight_layout()
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

def genACFGraphs(data, ticker, var_name, train_split = False):
    dirName = f"EDAOutput/EDA_{ticker}"
    data.dropna()

    if train_split:
        file_path = f"{dirName}/training_{ticker}_{var_name}_ACF.png"
        train_size = int(len(data)*0.8)
        data = data.iloc[:train_size]
    else:
        file_path = f"{dirName}/original_{ticker}_{var_name}_ACF.png"

    fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15, 15))

    plot_acf(data[var_name], lags=25, auto_ylims=True, missing="drop", 
             zero=False, color="red", alpha=0.05, ax=ax1, title="ACF")
    
    plot_pacf(data[var_name].dropna(), lags=25, alpha=0.05, zero=False, ax=ax2,
              auto_ylims=True, title="PACF")
    
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

    # plot line graphs for Close LR and Diff
    genLineSub("Close", "Group_ClosePLineGraphs", 'm', 150)
    genLineSub("Log_Returns", "Group_LogReturnsLineGraphs", 'm', 150)
    genLineSub("Differenced", "Group_DiffLineGraphs", 'm', 150)

    # to make individual plots for each ticker
    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)
        sized_df = set_df_size(df, 150)

        # create eda folders for each ticker
        ticker = get_ticker(file_name)
        create_output_folder(ticker)

        # get basic statistics of data 
        summarise_data(file_name, True, 150)

        # # Get Auto Correlation Graphs of Close Price
        genACFGraphs(sized_df,ticker,"Close", True)
        genACFGraphs(sized_df,ticker,"Differenced", True)
        genACFGraphs(sized_df,ticker,"Log_Returns", True)

        # basic visualisation of data
        #     # corr matrix thingy
  
main()