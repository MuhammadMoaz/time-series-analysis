import yfinance as yf
import pandas as pd
import matplotlib
import EDA

def fetch_data(ticker_symbol):
    # 261 does withot weekends in a year
    # 261 * 24 years is 6000 or something
    data = yf.download(ticker_symbol, start="2001-01-01")
    filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/{filename}")

    # filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/temp")
    return data

def preprocessing(df):
    print("preprocessing")

def perform_eda():
    EDA.main()

def perform_pda(df):
    print("pda")

def main():
    # Financials
    cba_data = fetch_data("CBA.AX")
    wbc_data = fetch_data("WBC.AX")
    nab_data = fetch_data("NAB.AX")

    # Materials
    bhp_data = fetch_data("BHP.AX")
    amc_data = fetch_data("AMC.AX")
    rio_data = fetch_data("RIO.AX")

    # Healthcare
    csl_data = fetch_data("CSL.AX")
    rmd_data = fetch_data("RMD.AX")
    pme_data = fetch_data("PME.AX")

    perform_eda()
    
    
main()