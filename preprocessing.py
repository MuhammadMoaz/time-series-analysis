import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, end="2025-01-01", start="2001-01-01", multi_level_index=False)
    filename = ticker_symbol + "_data.csv"
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.to_csv(f"datasets/{filename}")

def main():
    # Banks
    fetch_data("CBA.AX")
    fetch_data("WBC.AX")
    fetch_data("NAB.AX")

    # Materials
    fetch_data("BHP.AX")
    fetch_data("AMC.AX")
    fetch_data("RIO.AX")

    # Healthcare
    fetch_data("CSL.AX")
    fetch_data("RMD.AX")
    fetch_data("PME.AX")

main()