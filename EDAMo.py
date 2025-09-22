import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def summarise_data(data, file_path):
    ticker = get_ticker(file_path)
    data = pd.read_csv(file_path)

    with open(f"EDAOutput/EDA_{ticker}/{ticker}.txt", 'w+') as f:
        f.write(f"{ticker} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}")

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

    col_data = data[columns].diff().dropna()
    corr_matrix = col_data.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.title(f"{ticker} Correlation Matrix (Difference in Close)")
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
    plt.title(f"{ticker} Line Graph")

    plt.savefig(file_path)
    plt.clf()

# Visualisation X - Boxplots

# Visualisation X - Scatterplots

def main():
    datasets = Path("datasets").rglob("*.csv")

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)

        ticker = get_ticker(file_name)
        create_output_folder(file_name, ticker)
        summarise_data(df, file_name)

        genHistogram(df, ticker)
        genCorrMatrix(df, ticker)
        genLineGraph(df, ticker)

main()