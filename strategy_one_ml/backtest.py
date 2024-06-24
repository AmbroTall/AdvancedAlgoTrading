import time
import pytz
from datetime import datetime
from ib_insync import *
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Parameters
lookback_minutes = 30
lookforward_minutes = 5
random_state = 42
n_estimators = 400
n_jobs = 1

# Set timezone to Kenya (EAT)
timezone = pytz.timezone('Africa/Nairobi')

# Define market hours (EAT)
market_open = timezone.localize(datetime.strptime("00:00", "%H:%M"))
market_close = timezone.localize(datetime.strptime("23:59", "%H:%M"))

# Connect to IB TWS
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the forex contracts
forex_pairs = [
    Forex('EURUSD', exchange='IDEALPRO'),
    Forex('GBPUSD', exchange='IDEALPRO'),
    Forex('USDJPY', exchange='IDEALPRO')
]

# Function to fetch historical data with retry mechanism
def fetch_historical_data_with_retry(contract, end_datetime='', duration_str='1 M', bar_size='5 mins', retries=5, delay=5):
    for attempt in range(retries):
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end_datetime,
                durationStr=duration_str,
                barSizeSetting=bar_size,
                whatToShow='MIDPOINT',
                useRTH=True,
                formatDate=1
            )
            if bars:
                df = util.df(bars)
                df.set_index('date', inplace=True)
                df['symbol'] = contract.symbol
                return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {contract.symbol} with error: {e}")
            time.sleep(delay)
    print(f"Failed to fetch data for {contract.symbol} after {retries} attempts.")
    return None

# Function to fetch historical data for missing dates
def fetch_missing_data(contract, last_date):
    end_datetime = ''
    duration_str = '1 M'
    bar_size = '5 mins'
    dfs = []

    while True:
        df = fetch_historical_data_with_retry(contract, end_datetime, duration_str, bar_size)
        if df is not None:
            df = df[df['date'] > last_date]
            if df.empty:
                break
            df['symbol'] = contract.symbol
            dfs.append(df)
            end_datetime = df.index[0].strftime('%Y%m%d %H:%M:%S')
            time.sleep(1)  # To respect rate limits
        else:
            break

    if dfs:
        return pd.concat(dfs)
    else:
        return None

# Function to fetch and prepare historical data
def fetch_and_prepare_data(contract):
    end_datetime = ''
    duration_str = '1 Y'
    bar_size = '5 mins'
    return fetch_historical_data_with_retry(contract, end_datetime, duration_str, bar_size)

# Function to save data to CSV
def save_data(df, filename):
    df.to_csv(filename)

# Function to load data from CSV
def load_data(filename):
    return pd.read_csv(filename, index_col='date', parse_dates=True)

# Function to update data with new fetched data
def update_data(existing_df, new_df):
    combined_df = pd.concat([existing_df, new_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    return combined_df

# Function to create up-down DataFrame
def create_up_down_dataframe(df, lookback_minutes=30, lookforward_minutes=5, up_down_factor=2.0, percent_factor=0.01):
    ts = df.copy()
    ts.index = pd.to_datetime(ts.index)

    ts.drop(['open', 'high', 'low', 'volume', 'average', 'barCount'], axis=1, inplace=True)

    for i in range(lookback_minutes):
        ts[f"Lookback{i + 1}"] = ts["close"].shift(i + 1)
    for i in range(lookforward_minutes):
        ts[f"Lookforward{i + 1}"] = ts["close"].shift(-(i + 1))
    ts.dropna(inplace=True)

    ts["Lookback0"] = ts["close"].pct_change() * 100.0
    for i in range(lookback_minutes):
        ts[f"Lookback{i + 1}"] = ts[f"Lookback{i + 1}"].pct_change() * 100.0
    for i in range(lookforward_minutes):
        ts[f"Lookforward{i + 1}"] = ts[f"Lookforward{i + 1}"].pct_change() * 100.0
    ts.dropna(inplace=True)

    up = up_down_factor * percent_factor
    down = percent_factor

    down_cols = [ts[f"Lookforward{i + 1}"] > -down for i in range(lookforward_minutes)]
    up_cols = [ts[f"Lookforward{i + 1}"] > up for i in range(lookforward_minutes)]

    down_tot = down_cols[0]
    for c in down_cols[1:]:
        down_tot = down_tot & c
    up_tot = up_cols[0]
    for c in up_cols[1:]:
        up_tot = up_tot | c

    ts["UpDown"] = np.sign(ts["Lookforward1"])
    ts["UpDown"] = ts["UpDown"].astype(int)
    ts.replace({'UpDown': {0: -1}}, inplace=True)

    # Encode the currency pair symbol
    ts = pd.get_dummies(ts, columns=['symbol'], drop_first=True)
    return ts

# Function to load and prepare data
def load_and_prepare_data(forex_pairs):
    all_data = []

    for pair in forex_pairs:
        filename = f'data_{pair.symbol}.csv'
        if os.path.exists(filename):
            df = load_data(filename)
            ts = create_up_down_dataframe(df)
            all_data.append(ts)
        else:
            print(f"Data file for {pair.symbol} not found.")

    if all_data:
        combined_data = pd.concat(all_data)
        combined_data = combined_data.dropna()
        return combined_data
    else:
        return None

# Kelly criterion calculation
def kelly_criterion(win_prob, win_loss_ratio):
    return win_prob - ((1 - win_prob) / win_loss_ratio)

# Backtest function
def backtest(X_test, y_test, y_pred):
    initial_balance = 100000
    current_balance = initial_balance
    trade_log = []
    positions = []
    max_balance = current_balance  # For drawdown calculation

    for i in range(len(X_test)):
        prediction = y_pred[i]
        entry_price = X_test.iloc[i]['Lookback1']  # Assuming 'Lookback1' as a proxy for the entry price

        if prediction == 1:  # Buy signal
            stop_loss_price = entry_price - (entry_price * (1 / 2.0))  # Stop loss 2:1
            quantity = current_balance * 0.2  # Use 20% of balance
            positions.append((entry_price, stop_loss_price, 'BUY', quantity))
        elif prediction == -1:  # Sell signal
            stop_loss_price = entry_price + (entry_price * (1 / 2.0))  # Stop loss 2:1
            quantity = current_balance * 0.2  # Use 20% of balance
            positions.append((entry_price, stop_loss_price, 'SELL', quantity))

        # Update balance based on positions
        for entry, stop, direction, qty in positions:
            if direction == 'BUY' and X_test.iloc[i]['Lookback0'] < stop:
                current_balance -= qty
                positions.remove((entry, stop, direction, qty))
            elif direction == 'SELL' and X_test.iloc[i]['Lookback0'] > stop:
                current_balance += qty
                positions.remove((entry, stop, direction, qty))

        # Log trades
        trade_log.append((X_test.index[i], entry_price, stop_loss_price, direction, quantity, current_balance))

        # Update max balance for drawdown calculation
        if current_balance > max_balance:
            max_balance = current_balance

    return trade_log, max_balance

# Main function
def main():
    all_data = []
    for contract in forex_pairs:
        df = fetch_and_prepare_data(contract)
        if df is not None:
            ts = create_up_down_dataframe(df, lookback_minutes, lookforward_minutes)
            all_data.append(ts)

    if all_data:
        # Combine all data
        combined_data = pd.concat(all_data)
        combined_data.dropna(inplace=True)

        # Prepare features and labels
        X = combined_data[[f"Lookback{i}" for i in range(lookback_minutes)] + [col for col in combined_data.columns if col.startswith('symbol_')]]
        y = combined_data["UpDown"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, max_depth=10)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy}")
        print(confusion_matrix(y_test, y_pred))

        # Backtest
        trade_log, max_balance = backtest(X_test, y_test, y_pred)

        # Convert trade log to DataFrame for analysis
        trade_df = pd.DataFrame(trade_log, columns=['Date', 'Entry Price', 'Stop Loss', 'Direction', 'Quantity', 'Balance'])

        # Calculate drawdown
        trade_df['Max Balance'] = trade_df['Balance'].cummax()
        trade_df['Drawdown'] = trade_df['Max Balance'] - trade_df['Balance']

        # Plot balance over time
        plt.figure(figsize=(12, 6))
        plt.plot(trade_df['Date'], trade_df['Balance'], label='Balance')
        plt.plot(trade_df['Date'], trade_df['Max Balance'], label='Max Balance')
        plt.fill_between(trade_df['Date'], trade_df['Balance'], trade_df['Max Balance'], color='red', alpha=0.3, label='Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.title('Balance Over Time')
        plt.legend()
        plt.show()

        # Save trade log to CSV
        trade_df.to_csv('trade_log.csv', index=False)
    else:
        print("No data available for backtesting.")

# Run the backtest
main()

# Disconnect from IB
ib.disconnect()
