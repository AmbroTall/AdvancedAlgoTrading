from itertools import product
import numpy as np
import pandas as pd

sma1 = range(20, 61, 4)
sma2 = range(180, 281, 10)

# Load the CSV file
csv_dir = '../data/'
csv_file = csv_dir + 'tr_eikon_eod_data.csv'
raw = pd.read_csv(csv_file)
symbol = 'AAPL.O'
data = pd.DataFrame(raw[symbol]).dropna()
results = []

for SMA1, SMA2 in product(sma1, sma2):
    data = pd.DataFrame(raw[symbol])
    data.dropna(inplace=True)
    data['Returns'] = np.log(data[symbol] / data[symbol].shift(1))
    data['SMA1'] = data[symbol].rolling(SMA1).mean()
    data['SMA2'] = data[symbol].rolling(SMA2).mean()
    data.dropna(inplace=True)
    data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
    data['Strategy'] = data['Position'].shift(1) * data['Returns']
    data.dropna(inplace=True)
    perf = np.exp(data[['Returns', 'Strategy']].sum())
    results.append({
        'SMA1': SMA1,
        'SMA2': SMA2,
        'MARKET': perf['Returns'],
        'STRATEGY': perf['Strategy'],
        'OUT': perf['Strategy'] - perf['Returns']
    })

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)
results_df =results_df.sort_values('OUT', ascending=False).head(7)
print(results_df)
