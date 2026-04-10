import pandas as pd
import numpy as np

B = 3000
dates = pd.date_range("2018-01-01", periods = B, freq = "B")
df = pd.DataFrame({"Date" : dates, "weekday" : dates.weekday})

tickers = ["AAPL", "AMZN", "JPM", "TSLA", "MSFT"]
for tick in tickers: 
    df[tick] = np.random.normal(0, 0.01, len(dates))

df.to_csv("Stocks_logret.csv", index = False)