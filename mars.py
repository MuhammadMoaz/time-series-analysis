import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from pyearth import Earth
from sklearn.metrics import root_mean_squared_error
from pathlib import Path

def create_output_folder(file_name, ticker):
    dir_name = f"PDAOutput/PDA_{ticker}"
    os.makedirs(dir_name, exist_ok=True)

def get_ticker(file_name):
    strName = file_name.replace("datasets\\","")
    NameParts = strName.split('.')
    return NameParts[0]

def main():
    datasets = Path("datasets").rglob("*.csv")

    for data in datasets:
        file_name = str(data)
        df = pd.read_csv(data)
        ticker = get_ticker(file_name)
        create_output_folder(file_name, ticker)

        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

main()