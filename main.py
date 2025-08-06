import yfinance as yf
import matplotlib

def fetch_data(ticker_symbol):
    data = yf.download(ticker_symbol, period="max")
    filename = ticker_symbol + "_data.csv"
    data.to_csv(filename)
    
    print(data.head())
    print(data.shape)

def preprocessing():
    print("preprocessing")

def eda():
    print("eda")

def pda():
    print("pda")

def main():
    fetch_data("CBA.AX")
    fetch_data("WBC.AX")
    fetch_data("NAB.AX")

    preprocessing()
    eda()
    pda()

main()