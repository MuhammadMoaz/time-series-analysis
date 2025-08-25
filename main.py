import yfinance as yf
import pandas as pd
import matplotlib
import EDA

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
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
    # Banking Stocks
    cba_data = fetch_data("CBA.AX")
    wbc_data = fetch_data("WBC.AX")
    nab_data = fetch_data("NAB.AX")
    banking_tickers = ["CBA.AX", "WBC.AX", "NAB.AX"]
    banking_data = fetch_data(banking_tickers)

    # cba_data = fetch_data("CBA.AX")
    # perform_eda(cba_data)

    # wbc_data = fetch_data("WBC.AX")
    # perform_eda(wbc_data)

    # nab_data = fetch_data("NAB.AX")
    # perform_eda(nab_data)

    # Materials
    bhp_data = fetch_data("BHP.AX")
    amc_data = fetch_data("AMC.AX")
    rio_data = fetch_data("RIO.AX")

    # Healthcare
    csl_data = fetch_data("CSL.AX")
    rmd_data = fetch_data("RMD.AX")
    pme_data = fetch_data("PME.AX")

    perform_eda()
    # Materials Stocks
    # bhp_data = fetch_data("BHP.AX")
    # perform_eda(bhp_data)

    # nem_data = fetch_data("NEM.AX")
    # perform_eda(nem_data)

    # fmg_data = fetch_data("FMG.AX")
    # perform_eda(fmg_data)

    # Pharma Stocks
    
    
main()