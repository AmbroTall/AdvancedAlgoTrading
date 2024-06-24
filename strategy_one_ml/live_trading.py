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
from sklearn.metrics import confusion_matrix

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
ib.connect('127.0.0.1', 7497, clientId=2)

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
    duration_str = '5 D'
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

# Function to place orders
def place_order(contract, action, quantity, stop_loss_price):
    order = MarketOrder(action, quantity)
    trade = ib.placeOrder(contract, order)
    stop_order = StopOrder('SELL' if action == 'BUY' else 'BUY', quantity, stop_loss_price)
    ib.placeOrder(contract, stop_order)
    return trade

# Function to log trades
def log_trade(trade_log, timestamp, contract, action, quantity, price, balance):
    trade_log.append([timestamp, contract.symbol, action, quantity, price, balance])
    df = pd.DataFrame(trade_log, columns=['Timestamp', 'Symbol', 'Action', 'Quantity', 'Price', 'Balance'])
    df.to_csv('trade_log.csv', index=False)

# Function to get current account balance
def get_account_balance():
    account_summary = ib.accountSummary()
    for summary in account_summary:
        if summary.tag == 'TotalCashBalance' and summary.currency == 'USD':
            return float(summary.value)
    return None

# Main function to fetch, update, retrain, and trade
def main():
    global trade_log

    # Update data and retrain model
    for contract in forex_pairs:
        filename = f'data_{contract.symbol}.csv'
        if os.path.exists(filename):
            existing_df = load_data(filename)
            last_date = existing_df.index[-1]
            new_df = fetch_missing_data(contract, last_date)
            if new_df is not None:
                updated_df = update_data(existing_df, new_df)
                save_data(updated_df, filename)
                print(f"Updated data for {contract.symbol}")
            else:
                print(f"No new data for {contract.symbol}")
        else:
            existing_df = fetch_and_prepare_data(contract)
            if existing_df is not None:
                save_data(existing_df, filename)
                print(f"Saved data for {contract.symbol} to {filename}")
            else:
                print(f"Failed to prepare data for {contract.symbol}")

    combined_data = load_and_prepare_data(forex_pairs)
    if combined_data is not None:
        # Prepare features and labels for the model
        X = combined_data[[f"Lookback{i}" for i in range(lookback_minutes)] + [col for col in combined_data.columns if col.startswith('symbol_')]]
        y = combined_data["UpDown"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

        # Train the model
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, max_depth=10)
        model.fit(X_train, y_train)

        # Evaluate the model
        print(f"Hit-Rate: {model.score(X_test, y_test)}")
        print(confusion_matrix(model.predict(X_test), y_test))

        # Save the model
        joblib.dump(model, 'ml_model_rf_combined.pkl')
    else:
        print("No data available for training.")

    # Load model for live trading
    model = joblib.load('ml_model_rf_combined.pkl')

    # Get current account balance
    current_balance = get_account_balance()
    if current_balance is None:
        print("Failed to retrieve account balance.")
        return

    # Kelly criterion parameters
    risk_fraction = 0.2
    win_prob = 0.6  # Example win probability
    win_loss_ratio = 2.0
    kelly_fraction = kelly_criterion(win_prob, win_loss_ratio)
    trade_amount = current_balance * risk_fraction * kelly_fraction

    # Live trading
    for contract in forex_pairs:
        filename = f'data_{contract.symbol}.csv'
        df = load_data(filename)
        ts = create_up_down_dataframe(df)
        if ts is not None:
            X_live = ts[[f"Lookback{i}" for i in range(lookback_minutes)] + [col for col in ts.columns if col.startswith('symbol_')]].tail(1)
            prediction = model.predict(X_live)[0]
            entry_price = ts['close'].iloc[-1]

            # Place order if no existing positions
            positions = ib.positions()
            if not any(pos.contract.symbol == contract.symbol for pos in positions):
                if prediction == 1:  # Buy signal
                    stop_loss_price = entry_price - (entry_price * (1 / win_loss_ratio))
                    trade = place_order(contract, 'BUY', trade_amount, stop_loss_price)
                    current_balance -= trade_amount
                    log_trade(trade_log, datetime.now(), contract, 'BUY', trade_amount, entry_price, current_balance)
                elif prediction == -1:  # Sell signal
                    stop_loss_price = entry_price + (entry_price * (1 / win_loss_ratio))
                    trade = place_order(contract, 'SELL', trade_amount, stop_loss_price)
                    current_balance += trade_amount
                    log_trade(trade_log, datetime.now(), contract, 'SELL', trade_amount, entry_price, current_balance)

# Initialize trade log
trade_log = []

# Function to check if the current time is within trading hours
def is_trading_hours():
    now = timezone.localize(datetime.now())
    return market_open.time() <= now.time() <= market_close.time()

# Run the trading loop
while True:
    if is_trading_hours():
        main()
    time.sleep(300)  # Sleep for 5 minutes

# Disconnect from IB (this line will never be reached in the current while loop setup)
ib.disconnect()
