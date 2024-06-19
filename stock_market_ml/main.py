import yfinance as yf
from matplotlib import pyplot as plt

# Download historical data
sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history("max")
print(sp500.columns)

# Check on the index
# print(sp500.index)

#Plot the graph of sp500
sp500.plot.line(y='Close', use_index=True)
plt.title('S&P 500 Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.savefig('sp500_close_prices.png')

# Data Cleaning
# Deleting columns df.drop('column name', inplace=True) or del df['col']
sp500.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

# ************ MAchine Learning ************
#Define the target variable *** .shift(-1)->use previous days price as todays price while .shift(1)->compare previous price with current price (calculate daily returns)
sp500['Tomorrow'] = sp500['Close'].shift(-1)  #Creates the target column by shifting tomorrows price alongside the closing price

sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)

# Take relevant data of the stock market
sp500 = sp500.loc['2000-01-01':].copy()  #.copy()-> used to avoid pd errors
print(sp500.head())
