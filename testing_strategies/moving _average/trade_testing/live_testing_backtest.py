import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Set the seaborn style
sns.set(style='darkgrid')

# Other matplotlib settings
plt.rcParams['font.family'] = 'serif'

# Load the CSV file
csv_dir = '../../data/'
csv_file = csv_dir + 'EURUSD_historical_data.csv'
raw = pd.read_csv(csv_file)

# Display the DataFrame info (optional)
# raw.info()

symbol = 'EURUSD'
data = pd.DataFrame(raw['Low']).dropna()
print(data)
# Strategy
SMA1 = 10
SMA2 = 44

data['SMA1'] = data['Low'].rolling(SMA1).mean()
data['SMA2'] = data['Low'].rolling(SMA2).mean()

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
data.plot(ax=ax)
plt.title(f'eurusd Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')

# Save the plot to a file
plot_filename_initial = 'eurusd_price_and_moving_averages_initial.png'
plt.savefig(plot_filename_initial, dpi=300)
plt.show()
print(f"Initial plot saved as {plot_filename_initial}")

# Data processing for the second plot
data.dropna(inplace=True)

#Delete rows with 0s
rows_with_zeros = data[(data == 0).any(axis=1)]
data = data[~(data == 0).any(axis=1)]

data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)


# ++++++++++++++++TESTING++++++++++++++++
# This is the log returns  (Benchmark Investment)

# Returns : Actual value we would get when holding the asset.
# Strategy : Money we would get when using our strategy.
data['Returns'] = np.log(data['Low'] / data['Low'].shift(1))
data['Strategy'] = data['Position'].shift(1) * data['Returns']
data.dropna(inplace=True)
np.exp(data[['Returns','Strategy']].sum())
# annualized volatility for the strategy and the benchmark investment
data[['Returns','Strategy']].std() * 252 ** 0.5
print(data.tail())
print(np.exp(data[['Returns','Strategy']].sum()))

# Plotting cumulative returns comparing the original retruns and strategy returns
fig, ax = plt.subplots(figsize=(10, 6))
data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(ax=ax)
data['Position'].plot(ax=ax, secondary_y='Position', style='--')
ax.get_legend().set_bbox_to_anchor((0.25, 0.85))

# Save the plot to a file
plot_filename = 'eurusd_cumulative_returns_and_position.png'
plt.savefig(plot_filename, dpi=300)
