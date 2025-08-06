import yfinance as yf
import matplotlib

# Fetching, downloading, and saving CBA.AX, WBC.AX, and NAB.AX data
# CBA.AX
cba_ticker = "CBA.AX"
cba_data = yf.download(cba_ticker, period="max")
print(cba_data.head())
print(cba_data.shape)
cba_data.to_csv("CBA_Market_Data.csv")

# WBC.AX
wbc_ticker = "WBC.AX"
wbc_data = yf.download(wbc_ticker, period="max")
print(wbc_data.head())
print(wbc_data.shape)
wbc_data.to_csv("WBC_Market_Data.csv")

# NAB.AX
nab_ticker = "NAB.AX"
nab_data = yf.download(nab_ticker, period="max")
print(nab_data.head())
print(nab_data.shape)
nab_data.to_csv("NAB_Market_Data.csv")

