# main.py
from datetime import datetime

import numpy as np

from live_trading import IntradayMachineLearningPredictionStrategy
from qstrader.event import SignalEvent, EventType
from qstrader.price_parser import PriceParser
from qstrader.strategy.base import AbstractStrategy
import queue

if __name__ == "__main__":
    tickers = ['AAPL']
    events_queue = queue.Queue()
    strategy = IntradayMachineLearningPredictionStrategy(tickers, events_queue, 'ml_model_rf.pkl', lags=5)

    # Mock event to test the strategy
    class MockEvent:
        def __init__(self, time, close_price):
            self.type = EventType.BAR
            self.time = time
            self.close_price = close_price

    # Simulate receiving market data and running the strategy
    for i in range(100):
        event = MockEvent(datetime.now(), PriceParser.parse(100 + np.random.randn()))
        print("Event Type:", event.type)
        print("Event Time:", event.time)
        print("Close Price:", event.close_price)
        signal = strategy.calculate_signals(event)
        print(signal)

