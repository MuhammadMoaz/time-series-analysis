import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def create_output_folder(ticker):
    dirName = f"PDAOutput/PDA_{ticker}"
    os.makedirs(dirName, exist_ok=True)


def get_ticker(file_name):
    strName = file_name.replace("datasets\\", "")
    NameParts = strName.split('.')
    return NameParts[0]


def series_transform(data, in_n=1, out_n=1):
    """Convert a series into supervised learning format (lags + target)."""
    df = pd.DataFrame(data)
    colmns = []

    # input lags
    for i in range(in_n, 0, -1):
        colmns.append(df.shift(i))

    # output (future values)
    for i in range(0, out_n):
        colmns.append(df.shift(-i))

    aggregated = pd.concat(colmns, axis=1)
    aggregated = aggregated.dropna()
    return aggregated.values


def rf_model(data, ticker):
    df = pd.read_csv(data)

    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Get closing prices
    series = df["Close"].copy()

    # Turn into supervised dataset
    data = series_transform(series, in_n=3, out_n=1)

    # 80/20 split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    trainX, trainy = train[:, :-1], train[:, -1]
    testX, testy = test[:, :-1], test[:, -1]

    # Train model ONCE
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(trainX, trainy)

    # Predict test set
    yhat = model.predict(testX)

    # Metrics
    rmse = root_mean_squared_error(testy, yhat)
    mape = mean_absolute_percentage_error(testy, yhat)
    mae = mean_absolute_error(testy, yhat)
    r2 = r2_score(testy, yhat)

    print(f"{ticker} | MAE: {mae:.3f}, RMSE: {rmse:.3f}, "
          f"MAPE: {mape*100:.2f}%, R2: {r2:.3f}")

    # Save metrics to CSV
    results = {
        "Ticker": ticker,
        "Model": "RF",
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv("metrics.csv", mode='a+', header=not os.path.exists("metrics.csv"), index=False)

    # Plot actual vs predicted
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(df["Date"], df["Close"], label="Actual")
    ax1.plot(df["Date"].iloc[len(df) - len(yhat):], yhat, label="Predicted")

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.set_ylabel("Close Price")
    plt.xticks(rotation=70)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"PDAOutput/PDA_{ticker}/{ticker}_RFFinalMo2.png")
    plt.clf()


def main():
    datasets = Path("datasets").rglob("*.csv")
    dataset_list = []
    ticker_list = []

    for data in datasets:
        filename = str(data)
        ticker = get_ticker(str(data))
        create_output_folder(ticker)
        dataset_list.append(filename)
        ticker_list.append(ticker)

    # Example: run model on the 9th dataset
    rf_model(dataset_list[8], ticker_list[8])


if __name__ == "__main__":
    main()
