import threading
import time
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

# Configurable Parameters
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
DURATION = "2 D"
CANDLE_SIZE = "5 mins"
VWAP_BACKCANDLES = 15
RSI_LENGTH = 16
BBANDS_LENGTH = 14
BBANDS_STD = 2.0
POSITION_SIZE = 1000  # Number of units per trade
RISK_PER_TRADE = 0.01  # Risk 1% of capital per trade
CAPITAL = 100000  # Total capital


class TradingApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = {}
        self.orders = {}
        self.nextOrderId = None

    def historicalData(self, reqId, bar):
        if reqId not in self.data:
            self.data[reqId] = []
        self.data[reqId].append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical data fetched for ReqId {reqId}: {len(self.data[reqId])} bars")

    def nextValidId(self, orderId):
        self.nextOrderId = orderId

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print(f"Order {orderId} status: {status}, filled: {filled}, remaining: {remaining}")

    def openOrder(self, orderId, contract, order, orderState):
        print(f"Open order {orderId}: {contract.symbol} {order.action} {order.totalQuantity} @ {order.lmtPrice}")

    def error(self, reqId, errorCode, errorString):
        print(f"Error {reqId}, Code {errorCode}, Msg: {errorString}")


def websocket_con():
    app = TradingApp()
    app.connect("127.0.0.1", 7497, clientId=123)
    con_thread = threading.Thread(target=app.run, daemon=True)
    con_thread.start()
    time.sleep(1)
    return app


def forex_contract(pair_code):
    contract = Contract()
    contract.symbol = pair_code[:3]
    contract.secType = 'CASH'
    contract.exchange = 'IDEALPRO'
    contract.currency = pair_code[3:]
    return contract


def fetch_historical_data(app, contract, duration, candle_size):
    req_id = 1
    app.data[req_id] = []  # Initialize the data list for the request ID
    app.reqHistoricalData(reqId=req_id, contract=contract, endDateTime='',
                          durationStr=duration, barSizeSetting=candle_size,
                          whatToShow='MIDPOINT', useRTH=0, formatDate=1, keepUpToDate=False,
                          chartOptions=[])
    time.sleep(5)  # Give some time to fetch the data
    if req_id in app.data and app.data[req_id]:
        print(f"Data fetched for request ID {req_id}: {len(app.data[req_id])} records")
        df = pd.DataFrame(app.data[req_id], columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['DateTime'] = pd.to_datetime(df['DateTime'])  # Ensure DateTime column is in datetime format
        df.set_index('DateTime', inplace=True)  # Set DateTime column as index
        return df
    else:
        print(f"No data fetched for request ID {req_id}")
        return pd.DataFrame(columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])


def add_technical_indicators(df):
    df["VWAP"] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    df['RSI'] = ta.rsi(df['Close'], length=RSI_LENGTH)
    df = df.join(ta.bbands(df['Close'], length=BBANDS_LENGTH, std=BBANDS_STD))
    return df


def calculate_vwap_signals(df, backcandles):
    df['Max_Open_Close'] = np.maximum(df['Open'], df['Close'])
    df['Min_Open_Close'] = np.minimum(df['Open'], df['Close'])
    df['upt'] = 1
    df['dnt'] = 1
    df.loc[(df['Max_Open_Close'] >= df['VWAP']), 'dnt'] = 0
    df.loc[(df['Min_Open_Close'] <= df['VWAP']), 'upt'] = 0
    df['sig_dnt'] = df['dnt'].rolling(backcandles + 1, min_periods=1).min()
    df['sig_upt'] = df['upt'].rolling(backcandles + 1, min_periods=1).min()
    df['VWAPSignal'] = 0
    df.loc[(df['sig_upt'] == 1) & (df['sig_dnt'] == 1), 'VWAPSignal'] = 3
    df.loc[(df['sig_upt'] == 1) & (df['sig_dnt'] == 0), 'VWAPSignal'] = 2
    df.loc[(df['sig_upt'] == 0) & (df['sig_dnt'] == 1), 'VWAPSignal'] = 1
    return df


def backtest_strategy(df, capital, risk_per_trade, stop_loss_pct):
    position_size = int((capital * risk_per_trade) / stop_loss_pct)
    initial_capital = capital
    trades = []
    equity_curve = [capital]

    for index, row in df.iterrows():
        if row['VWAPSignal'] == 2:  # Buy Signal
            entry_price = row['Close']
            stop_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + 2 * stop_loss_pct)
            trade = {
                'type': 'BUY',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'exit_price': None,
                'profit': None
            }
            trades.append(trade)
        elif row['VWAPSignal'] == 1:  # Sell Signal
            entry_price = row['Close']
            stop_price = entry_price * (1 + stop_loss_pct)
            take_profit_price = entry_price * (1 - 2 * stop_loss_pct)
            trade = {
                'type': 'SELL',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'take_profit_price': take_profit_price,
                'exit_price': None,
                'profit': None
            }
            trades.append(trade)

    # Calculate profit/loss for each trade
    for trade in trades:
        if trade['type'] == 'BUY':
            if (df['Close'] >= trade['take_profit_price']).any():  # Check if any Close prices are >= take profit
                exit_price = trade['take_profit_price']
            else:
                exit_price = trade['stop_price']
        else:  # SELL
            if (df['Close'] <= trade['take_profit_price']).any():  # Check if any Close prices are <= take profit
                exit_price = trade['take_profit_price']
            else:
                exit_price = trade['stop_price']

        trade['exit_price'] = exit_price
        trade['profit'] = (exit_price - trade['entry_price']) * position_size if trade['type'] == 'BUY' else (trade['entry_price'] - exit_price) * position_size
        capital += trade['profit']
        equity_curve.append(capital)

    total_profit = capital - initial_capital
    win_rate = len([t for t in trades if t['profit'] > 0]) / len(trades) * 100 if trades else 0
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve)
    num_trades = len(trades)

    return trades, total_profit, win_rate, max_drawdown, num_trades, equity_curve


def plot_results(df, trades, pair, equity_curve):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['VWAP'], label='VWAP', linestyle='--')
    plt.fill_between(df.index, df['BBL_14_2.0'], df['BBU_14_2.0'], color='gray', alpha=0.3, label='Bollinger Bands')

    buy_signals = df[df['VWAPSignal'] == 2]
    sell_signals = df[df['VWAPSignal'] == 1]

    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal')

    plt.title(f'Backtest Results for {pair}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f'backtest_results_{pair}.png')
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve, label='Equity Curve')
    plt.title(f'Equity Curve for {pair}')
    plt.xlabel('Trade Number')
    plt.ylabel('Equity')
    plt.legend()
    plt.savefig(f'equity_curve_{pair}.png')
    plt.show()


def main():
    app = websocket_con()
    for pair in FOREX_PAIRS:
        contract = forex_contract(pair)
        df = fetch_historical_data(app, contract, DURATION, CANDLE_SIZE)
        if df.empty:
            print(f"No data available for {pair}. Skipping.")
            continue

        df = add_technical_indicators(df)
        df = calculate_vwap_signals(df, VWAP_BACKCANDLES)

        stop_loss_pct = 0.01  # Example: 1% stop loss
        trades, total_profit, win_rate, max_drawdown, num_trades, equity_curve = backtest_strategy(df, CAPITAL, RISK_PER_TRADE, stop_loss_pct)

        print(f"Backtest results for {pair}:")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Number of Trades: {num_trades}")
        print(f"Max Drawdown: ${max_drawdown:.2f}")
        print("Trades:")
        for trade in trades:
            print(trade)

        plot_results(df, trades, pair, equity_curve)


if __name__ == "__main__":
    main()
