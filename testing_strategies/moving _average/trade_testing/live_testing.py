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
csv_dir = '../data/'
csv_file = csv_dir + 'EURUSD_historical_data.csv'
raw = pd.read_csv(csv_file)

# Display the DataFrame info (optional)
# raw.info()

data = pd.DataFrame(raw['Low']).dropna()
print(data['Low'].head())

# Strategy
SMA1 = 42
SMA2 = 252

data['SMA1'] = data['Low'].rolling(SMA1).mean()
data['SMA2'] = data['Low'].rolling(SMA2).mean()

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
data.plot(ax=ax)
plt.title(f'eurusdPrice and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')

# Save the plot to a file
plot_filename_initial = 'eurusd_price_and_moving_averages_initial.png'
plt.savefig(plot_filename_initial, dpi=300)
plt.show()
print(f"Initial plot saved as {plot_filename_initial}")

# Data processing for the second plot
data.dropna(inplace=True)
data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)

# Print data info
print(data.tail())
print(data['Position'].value_counts())

# Plotting the second graph
fig, ax = plt.subplots(figsize=(10, 6))
data[['Low', 'SMA1', 'SMA2']].plot(ax=ax)
data['Position'].plot(ax=ax, secondary_y='Position', style='g', alpha=0.3)
plt.title(f'eurusd Price, Moving Averages and Position')
plt.xlabel('Date')
plt.ylabel('Price')

# Save the second plot to a file
plot_filename_final = 'eurusd_price_moving_averages_and_position.png'
plt.savefig(plot_filename_final, dpi=300)
plt.show()

print(f"Final plot saved as {plot_filename_final}")
