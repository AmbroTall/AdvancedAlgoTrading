# live_trading.py
import numpy as np
import pandas as pd
import joblib
from qstrader.price_parser import PriceParser
from qstrader.event import SignalEvent, EventType
from qstrader.strategy.base import AbstractStrategy
from datetime import datetime


class IntradayMachineLearningPredictionStrategy(AbstractStrategy):
    def __init__(self, tickers, events_queue, model_pickle_file, lags=5):
        self.tickers = tickers
        self.events_queue = events_queue
        self.model_pickle_file = model_pickle_file
        self.lags = lags
        self.invested = False
        self.cur_prices = np.zeros(self.lags + 1)
        self.cur_returns = np.zeros(self.lags)
        self.minutes = 0
        self.qty = 10000
        self.model = joblib.load(model_pickle_file)
        self.new_data = []
        self.signals = []  # Store signals here

    def _update_current_returns(self, event):
        for i, f in reversed(list(enumerate(self.cur_prices))):
            if i > 0:
                self.cur_prices[i] = self.cur_prices[i - 1]
            else:
                self.cur_prices[i] = event.close_price / float(PriceParser.PRICE_MULTIPLIER)
        if self.minutes > (self.lags + 1):
            for i in range(self.lags):
                self.cur_returns[i] = ((self.cur_prices[i] / self.cur_prices[i + 1]) - 1.0) * 100.0
        self.new_data.append(self.cur_returns.copy())

    def predict(self):
        data_for_prediction = np.array([self.cur_returns]).reshape(1, -1)
        return self.model.predict(data_for_prediction)[0]

    def calculate_signals(self, event):
        if event.type == EventType.BAR:
            self._update_current_returns(event)
        self.minutes += 1
        if self.minutes > (self.lags + 2):
            prediction = self.predict()
            print("Prediction", prediction)
            if not self.invested and prediction == 1:
                self.signals.append(1)  # Long signal
                self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.qty))
                self.invested = True
            elif self.invested and prediction == -1:
                self.signals.append(-1)  # Close long signal
                self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.qty))
                self.invested = False
            else:
                self.signals.append(0)  # No signal
            if len(self.new_data) > 1000:
                self.retrain_model()

        return self.signals  # Return signals

    def retrain_model(self):
        new_data_df = pd.DataFrame(self.new_data, columns=[f'feature_{i}' for i in range(self.lags)])
        X_new = new_data_df.values
        y_new = (X_new[:, 0] > 0).astype(int)
        self.model.fit(X_new, y_new)
        joblib.dump(self.model, self.model_pickle_file)
        self.new_data = []
