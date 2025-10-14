import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

SECTOR_HASHMAP = {
    "WBC": "Financials", "CBA": "Financials", "NAB": "Financials",
    "CSL": "Materials", "BHP": "Materials", "RIO": "Materials",
    "RMD": "Pharma", "PME": "Pharma", "AMC": "Pharma"
}

def main():
    metrics = pd.read_csv("metrics.csv")

    metrics["Sector"] = metrics["Ticker"].map(SECTOR_HASHMAP)

    # Create folder for visualisations
    os.makedirs("EvalOutput", exist_ok=True)

    # Init output text file
    out_fn = 'model_eval.txt'

    with open(out_fn, 'w+') as f:

        # Average performance by model
        f.write(str(metrics.groupby("Model")[["MAE", "RMSE", "MAPE", "R2"]].mean()))

        f.write("\n\n")

        # Average perfromance by sector and model
        f.write(str(metrics.groupby(["Sector", "Model"])[["MAE", "RMSE", "MAPE", "R2"]].mean()))

main()