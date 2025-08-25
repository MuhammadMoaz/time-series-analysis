import yfinance as yf
import pandas as pd
import matplotlib
import EDA

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
<<<<<<< HEAD
    # filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/temp")
=======
    filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/{filename}")

>>>>>>> 58a6ba63bc27b7d6bfe0f6c95f4657743b11c22d
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
<<<<<<< HEAD
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
    
    
=======
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
>>>>>>> 58a6ba63bc27b7d6bfe0f6c95f4657743b11c22d
main()