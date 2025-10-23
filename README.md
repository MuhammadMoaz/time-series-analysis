# Time-Series-Analysis
Time Series Analysis as part of the UC ICT Final Project Unit

## Overview
This project focuses on forecasting ASX-listed stock prices using different time-series analysis models. The aim was to compare traditional statistical approaches with machine learning and deep learning methods, and to evaluate their performance across multiple sectors (Financials, Materials, and Healthcare).  

## Datasets
The datasets have been sourced from Yahoo Finance using the yfinance API. API Reference: https://ranaroussi.github.io/yfinance/reference/index.html

## Metrics Used
The models were benchmarked using four evaluation metrics:  
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  
- MAPE (Mean Absolute Percentage Error)  
- R-Sqaured (Coefficient of Determination)  

## Usage Instructions
The project deliverables, including the codebase, datasets, and visualisations, are available in the GitHub repository. The repository is structured into folders for datasets, EDA outputs, PDA outputs, and evaluation outputs, with model scripts and other python files stored in the root folder. The main script (main.py) is also stored in the root folder and handles the entire workflow. To run the pipeline, users must set up the environment (Python 3.9+ with all dependencies listed in requirements.txt) and execute the main script to perform the entire project process. New datasets can be added into the /datasets folder in CSV format by calling the fetch_data function with the respective ticker symbol as a parameter in the preprocessing.py file. Once a new dataset is added, rerunning the pipeline will automatically update the EDA, PDA, and performance evaluation sections.

## Ongoing Maintenance Tips
In terms of ongoing maintenance, it is recommended to manage code through GitHub using appropriate version control practices such as push and pull requests. Performance should be monitored by reviewing the metrics and visualisation in the respective folders. If unusual results or visualisation occur, appropriate troubleshooting practices should be taken, starting from the preprocessing steps. 

## Known Limitations
There are some known limitations with the current system. ARIMA consistently performed poorly, with very high errors and negative R-Squared values, showing its poor fit on volatile stock data. LSTM produced mixed results, highlighting its inconsistency across sectors. Random Forest was the most consistent model, but improper implementation with aspects such as feature engineering can lead to underfitting or flatlining issues.

## Future Work
This project provides a lot of opportunity for future development. Future work on this project could focus on three main areas. First, supplementing the time series analytics project with sentiment analysis from news and social media would help in identifying market trends and making predictions. Secondly, scaling to cover stocks in all ASX sectors would broaden the scope of insights, providing a more holistic view of a model’s performance. Thirdly, testing additional models, such as Transformer-based models or Gradient Boosting models, could boost performance and benchmark results against more complex architectures.

