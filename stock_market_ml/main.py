import pandas as pd
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

# ************ MACHINE LEARNING ************
"""
The strategy here involves setting tommorows price in todays prediction column(Target). This is to help the machine learning algorithm to predict tomorows price from today's price.
"""
#Define the target variable *** .shift(-1)->use previous days price as todays price while .shift(1)->compare previous price with current price (calculate daily returns)
sp500['Tomorrow'] = sp500['Close'].shift(-1)  #Creates the target column by shifting tomorrows price alongside the closing price

sp500['Target'] = (sp500['Tomorrow'] > sp500['Close']).astype(int)

# Take relevant data of the stock market
sp500 = sp500.loc['2000-01-01':].copy()  #.copy()-> used to avoid pd errors
print(sp500.head())

# Use Random Forest
from sklearn.ensemble import RandomForestClassifier

train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close","Volume","High","Low"]
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
# Train the model
model.fit(train[predictors], train['Target'])  # Give the model the training set data and the target column to train

# Predict
preds = model.predict(test[predictors])  # Give the model the test data to perform predictions on data not trained with.
preds = pd.Series(preds, index=test.index)

# Calculate the accuracy score of the model
from sklearn.metrics import precision_score

score = precision_score(test['Target'], preds) # calculate the score between the already available data and the predictions
print(score)

# Plot the prediction with the real results
combined = pd.concat([test['Target'], preds], axis=1)
combined.plot()