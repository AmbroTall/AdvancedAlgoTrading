def calculate_signals(self, event):
    if event.type == EventType.BAR:
        self._update_current_returns(event)

    self.minutes += 1
    if self.minutes > (self.lags + 2):
        pred = self.model.predict(self.cur_returns.reshape((1, -1)))[0]

    # Long only strategy modified to include short selling
    if not self.invested:
        if pred == 1:
            print("LONG: %s" % event.time)
            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.qty))
            self.invested = 'long'
        elif pred == -1:
            print("SHORT: %s" % event.time)
            self.events_queue.put(SignalEvent(self.tickers[0], "SELL", self.qty))
            self.invested = 'short'

    elif self.invested == 'long' and pred == -1:
        print("CLOSING LONG: %s" % event.time)
        self.events_queue.put(SignalEvent(self.tickers[0], "SLD", self.qty))
        self.invested = False
    elif self.invested == 'short' and pred == 1:
        print("CLOSING SHORT: %s" % event.time)
        self.events_queue.put(SignalEvent(self.tickers[0], "BUY", self.qty))
        self.invested = False


# HEDGING WITH OPTIONS
def calculate_signals(self, event):
    """Using options (calls and puts) can provide a way to hedge positions. For example, buying put options to hedge long positions, or buying call options to hedge short positions, ensures that the maximum loss is limited to the premium paid for the options.

    Implementation:

    Incorporate an options strategy such that when a position is taken on a stock, an option is also bought to limit downside risk.
    For a long position in the stock, buy a put option. For a short position, buy a call option."""
    ...
    if not self.invested:
        if pred == 1:
            self.events_queue.put(SignalEvent(self.tickers[0], "BOT", self.qty))
            self.events_queue.put(OptionEvent(self.tickers[0], "BUY_PUT", self.qty, "appropriate_strike"))
            self.invested = 'long'
        elif pred == -1:
            self.events_queue.put(SignalEvent(self.tickers[0], "SELL", self.qty))
            self.events_queue.put(OptionEvent(self.tickers[0], "BUY_CALL", self.qty, "appropriate_strike"))
            self.invested = 'short'
    ...
