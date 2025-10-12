import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SECTOR_HASHMAP = {
    "WBC": "Financials", "CBA": "Financials", "NAB": "Financials",
    "CSL": "Materials", "BHP": "Materials", "RIO": "Materials",
    "RMD": "Pharmaceuticlas", "PME": "Pharmaceuticlas", "AMC": "Pharmaceuticlas"
}

def main():
    metrics = pd.read_csv("metrics.csv")

    metrics["Sector"] = metrics["Ticker"].map(SECTOR_HASHMAP)

    # Average performance by model
    print(metrics.groupby("Model")[["MAE", "RMSE", "MAPE", "R2"]].mean())

    # Average perfromance by sector and model
    print(metrics.groupby(["Sector", "Model"])[["MAE", "RMSE", "MAPE", "R2"]].mean())

    # Create folder for visualisations
    os.makedirs("EvalOutput", exist_ok=True)

main()