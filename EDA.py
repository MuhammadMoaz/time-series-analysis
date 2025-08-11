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

# prints verifying statistical checks
def importCheck(data, fn):
    strippedName = fn.replace("datasets\\","").rstrip(".csv")
    print(f"{strippedName} START:")
    print(f"Variables: {data.columns.tolist()}")
    print(f"Head: \n{data.head()}")
    print(f"Tail: \n{data.tail()}")
    print(f"Shape: \n{data.shape}")
    print(f"Description: \n{data.describe()}")
    print(f"Info:")
    data.info(verbose=True)
    print(f"Empty Cells: \n{data.isnull().sum()}")
    print(f"{strippedName} END")

def importCheckFile(data, fn):
    strippedName = fn.replace("datasets\\","").rstrip("AX_data.csv")
    dirName = f"EDAOutput/EDA_{strippedName}"
    makedirs(dirName, exist_ok=True)

    with open(f"EDAOutput/EDA_{strippedName}/{strippedName}.txt", 'w+') as f:
        f.write(f"{strippedName} EDA Output:")
        f.write(f"Variables: {data.columns.tolist()}")
        f.write(f"Head: \n{data.head()}")
        f.write(f"Tail: \n{data.tail()}")
        f.write(f"Shape: \n{data.shape}")
        f.write(f"{data.info(verbose=True)}")
        f.write(f"Empty Cells: \n{data.isnull().sum()}")

# function opens csv
def fileOpener(fn):
    data = pd.read_csv(fn)
    importCheckFile(data, fn)
    
def main():
    pathlist = Path("datasets").rglob("*.csv")
    for p in pathlist:
        strPath = str(p)
        fileOpener(strPath)

main()