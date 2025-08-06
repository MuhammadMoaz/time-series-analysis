import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

'''
This File takes csv files in as input and performs a EDA on them, 
it then outputs data into folders with appropriate titling, these 
folders will contain graphs of different types to visualise data
'''


# function calls all EDA making functions
def edaMaker():
    print("edaMaker")

# eda function 1
# eda function 2
# eda function 3
# eda function 4

# prints verifying statistical checks
def importCheck(data, fn):
    print(f"{fn} START")
    print(data.head())
    print(data.tail())
    print(data.shape)
    print(data.describe())
    print(data.info())
    print(f"{fn} END")


# function opens csv
def fileOpener(fn):
    data = pd.read_csv(fn)
    importCheck(data, fn)
    

def main():
    pathlist = Path("datasets").rglob("*.csv")
    for p in pathlist:
        strPath = str(p)
        fileOpener(strPath)



main()