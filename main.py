import yfinance as yf
import pandas as pd
import matplotlib
import EDA

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
    # filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/temp")
    return data

def preprocessing(df):
    print("preprocessing")

def perform_eda(df):
    # print(df.head())
    # print(df.tail())
    # print(df.shape)
    # print(df.describe())
    EDA.main()

def perform_pda(df):
    print("pda")

def main():
    # Banking Stocks
    banking_tickers = ["CBA.AX", "WBC.AX", "NAB.AX"]
    banking_data = fetch_data(banking_tickers)

    # cba_data = fetch_data("CBA.AX")
    # perform_eda(cba_data)

    # wbc_data = fetch_data("WBC.AX")
    # perform_eda(wbc_data)

    # nab_data = fetch_data("NAB.AX")
    # perform_eda(nab_data)

    # Materials Stocks
    # bhp_data = fetch_data("BHP.AX")
    # perform_eda(bhp_data)

    # nem_data = fetch_data("NEM.AX")
    # perform_eda(nem_data)

    # fmg_data = fetch_data("FMG.AX")
    # perform_eda(fmg_data)

    # Pharma Stocks
    
    
main()