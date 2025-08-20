import yfinance as yf
import pandas as pd
import matplotlib
import EDA

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
    filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/{filename}")

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
    # Financials
    cba_data = fetch_data("CBA.AX")
    perform_eda(cba_data)

    wbc_data = fetch_data("WBC.AX")
    perform_eda(wbc_data)

    nab_data = fetch_data("NAB.AX")
    perform_eda(nab_data)

    # Materials
    bhp_data = fetch_data("BHP.AX")
    amc_data = fetch_data("AMC.AX")
    rio_data = fetch_data("RIO.AX")

    # Healthcare
    csl_data = fetch_data("CSL.AX")
    rmd_data = fetch_data("RMD.AX")
    pme_data = fetch_data("PME.AX")

main()