from ib_insync import *
import pandas as pd
import numpy as np
import time

# Connect to IB TWS
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Define the forex contracts
forex_pairs = [
    Forex('EURUSD', exchange='IDEALPRO'),
    Forex('GBPUSD', exchange='IDEALPRO'),
    Forex('USDJPY', exchange='IDEALPRO')
]

# Function to fetch historical data in chunks and prepare the DataFrame
def fetch_and_prepare_data(contract):
    end_datetime = ''
    duration_str = '1 M'
    bar_size = '5 mins'
    total_duration = 2  # 2 months
    dfs = []

    for _ in range(total_duration):
        for _ in range(5):  # Retry up to 5 times for each chunk
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
                    dfs.append(df)
                    end_datetime = df['date'].iloc[0].strftime('%Y%m%d %H:%M:%S')
                    break
            except Exception as e:
                print(f"Failed to fetch data for {contract.symbol}, retrying... ({e})")
                time.sleep(5)  # Wait for 5 seconds before retrying
        else:
            print(f"Failed to fetch data for {contract.symbol} after multiple attempts.")
            return None

    if dfs:
        df = pd.concat(dfs)
        df.set_index('date', inplace=True)
        return create_up_down_dataframe(df)
    else:
        return None

# Data preparation function
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
    return ts


# Fetch historical data and prepare DataFrames for each forex pair
# for contract in forex_pairs:
#     ts = fetch_and_prepare_data(contract)
#     if ts is not None:
#         print(f"Prepared data for {contract.symbol}")
#         print(ts.head())
#     else:
#         print(f"Failed to prepare data for {contract.symbol}")


