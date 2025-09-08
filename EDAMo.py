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
# function calls all EDA making functions
def edaMaker():
    print("edaMaker")

def nameStripper(name):
    strName = name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

# output import check as txt file, and graphs as pngs, 
# store them in folders related by ticker codes

# Visualisation 1 - Histograms
def genHistogram(data, fn):
    strippedName = nameStripper(fn)
    dirName = f"EDAOutput/EDA_{strippedName}"
    
    columns = data.columns.tolist()
    columns.remove("Date")
    
    for var in columns:
        file_path = f"{dirName}/{strippedName}_{var}_Hist.png"
        
        if os.path.exists(file_path):
            continue
        else:
            plt.hist(data[var], color='purple')
            plt.title(f'Distribution of {var} Price')
            plt.xlabel(f'{var} Price')
            plt.ylabel('Count')
            # plt.show()
            plt.savefig(file_path)

def genCorrMatrix(data, fn):
    strippedName = nameStripper(fn)
    dirName = f"EDAOutput/EDA_{strippedName}"
    file_path = f"{dirName}/{strippedName}_corr.png"

    columns = data.columns.tolist()
    # columns.remove('Date')
    col_data = data[columns]
    

    corr_matrix = col_data.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.savefig(file_path)

# Visualisation 2 - Boxplots
# Visualisation 3 - Scatterplots

# eda function 2
# eda function 3
# eda function 4


def importCheckFile(data, fn):
    strippedName = nameStripper(fn)
    dirName = f"EDAOutput/EDA_{strippedName}"
    os.makedirs(dirName, exist_ok=True)

    with open(f"EDAOutput/EDA_{strippedName}/{strippedName}.txt", 'w+') as f:
        f.write(f"{strippedName} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}")

# function opens csv
def fileOpener(fn):
    data = pd.read_csv(fn)
    importCheckFile(data, fn)
    genHistogram(data, fn)
    genCorrMatrix(data, fn)
    
def main():
    pathlist = Path("datasets").rglob("*.csv")
    for p in pathlist:
        strPath = str(p)
        fileOpener(strPath)

main()