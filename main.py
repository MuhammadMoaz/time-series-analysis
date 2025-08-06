import yfinance as yf
import pandas as pd
import matplotlib

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
    filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/{filename}")
    return data

def preprocessing(df):
    print("preprocessing")

def perform_eda(df):
    print(df.head())
    print(df.tail())
    print(df.shape)
    print(df.describe())

def perform_pda(df):
    print("pda")

def main():
    cba_data = fetch_data("CBA.AX")
    perform_eda(cba_data)

    wbc_data = fetch_data("WBC.AX")
    perform_eda(wbc_data)

    nab_data = fetch_data("NAB.AX")
    perform_eda(nab_data)

main()