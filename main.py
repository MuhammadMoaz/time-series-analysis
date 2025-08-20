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
<<<<<<< HEAD
    # Financials
=======
    # Banking Stocks
>>>>>>> 44e69293fc162daf522c247b7013b389c970fdf1
    cba_data = fetch_data("CBA.AX")
    perform_eda(cba_data)

    wbc_data = fetch_data("WBC.AX")
    perform_eda(wbc_data)

    nab_data = fetch_data("NAB.AX")
    perform_eda(nab_data)

<<<<<<< HEAD
    # Materials
    bhp_data = fetch_data("BHP.AX")
    amc_data = fetch_data("AMC.AX")
    rio_data = fetch_data("RIO.AX")

    # Healthcare
    csl_data = fetch_data("CSL.AX")
    rmd_data = fetch_data("RMD.AX")
    pme_data = fetch_data("PME.AX")

=======
    # Materials Stocks
    bhp_data = fetch_data("BHP.AX")
    perform_eda(bhp_data)

    nem_data = fetch_data("NEM.AX")
    perform_eda(nem_data)

    fmg_data = fetch_data("FMG.AX")
    perform_eda(fmg_data)

    # Pharma Stocks
    
    
>>>>>>> 44e69293fc162daf522c247b7013b389c970fdf1
main()