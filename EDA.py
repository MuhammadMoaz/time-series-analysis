import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import makedirs
from pathlib import Path

'''
This File takes csv files in as input and performs an EDA on them, 
it then outputs data into folders with appropriate titling, these 
folders will contain graphs of different types to visualise data
'''
# function calls all EDA making functions
def edaMaker():
    print("edaMaker")

# output import check as txt file, and graphs as pngs, 
# store them in folders related by ticker codes
# eda function 1
# eda function 2
# eda function 3
# eda function 4

def nameStripper(name):
    strName = name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]


def genCorrMatrix(data, sname):
    dirName = f"EDAOutput/EDA_{sname}"
    file_path = f"{dirName}/{sname}_corr.png"

    columns = data.columns.tolist()
    # columns.remove('Date')
    col_data = data[columns]
    corr_matrix = col_data.corr()
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
    plt.savefig(file_path)


def importCheckFile(data, sname):
    dirName = f"EDAOutput/EDA_{sname}"
    makedirs(dirName, exist_ok=True)

    with open(f"EDAOutput/EDA_{sname}/{sname}.txt", 'w+') as f:
        f.write(f"{sname} EDA Output:\n")
        f.write(f"Variables: {data.columns.tolist()}\n")
        f.write(f"Head: \n{data.head()}\n")
        f.write(f"Tail: \n{data.tail()}\n")
        f.write(f"Shape: \n{data.shape}\n")
        f.write(f"{data.info(verbose=True)}\n")
        f.write(f"Empty Cells: \n{data.isnull().sum()}")

# function opens csv
def fileOpener(fn):
    data = pd.read_csv(fn)
    sname = nameStripper(fn)
    importCheckFile(data, sname)
    
def main():
    pathlist = Path("datasets").rglob("*.csv")
    for p in pathlist:
        strPath = str(p)
        fileOpener(strPath)