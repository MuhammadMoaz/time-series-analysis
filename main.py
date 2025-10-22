# def preprocessing(df):
#     print("preprocessing yipee")

# def perform_eda():
#     EDA.main()

# def perform_pda(df):
#     print("pda")

def uiChooser():
    eda_choice = 'y'
    pda_choice = 0

    print("--------------------------------------------")
    print("Time Series Forcasting EDA and PDA tool")
    print("Enter corrosponding keys to choose settings:")
    print("EDA: (y)es (n)o")
    eda_choice = input("Input: ").lower()
    print("Model choice:")
    print("ARIMA: 1")
    print("LSTM: 2")
    print("MARS: 3")
    print("Random Forest: 4")
    print("All: 5 (May take a while)")
    pda_choice = int(input("Input: "))
    print("--------------------------------------------")

def main():
    uiChooser()
    
main()