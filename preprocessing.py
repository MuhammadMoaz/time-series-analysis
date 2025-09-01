import yfinance as yf
import pandas as pd

def fetch_data(ticker_symbol):
    # 261 does withot weekends in a year
    # 261 * 24 years is 6000 or something
    data = yf.download(ticker_symbol, start="2001-01-01")
    # data = data.reset_index(['Date'])
    # filename = ticker_symbol + "_data.csv"
    data.to_csv(f"datasets/temp_data.csv")

def main():
    # Banks
    # fetch_data("CBA.AX")
    # fetch_data("WBC.AX")
    # fetch_data("NAB.AX")

    # # Materials
    # fetch_data("BHP.AX")
    # fetch_data("AMC.AX")
    # fetch_data("RIO.AX")

    # # Healthcare
    # fetch_data("CSL.AX")
    # fetch_data("RMD.AX")
    # fetch_data("PME.AX")

    tks = ['CBA.AX', 'WBC.AX', 'NAB.AX']
    fetch_data(tks)
