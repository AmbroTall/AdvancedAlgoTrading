# model_training.py
import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def create_up_down_dataframe(
        csv_filepath,
        lookback_minutes=30,
        lookforward_minutes=5,
        up_down_factor=2.0,
        percent_factor=0.01,
        start=None, end=None
):
    ts = pd.read_csv(
        csv_filepath,
        names=[
            "Date", "Open", "High", "Low",
            "Close", "Adj Close", "Volume"
        ],
        header=0,
        index_col="Date",
        parse_dates=True,
        date_format='%Y-%m-%d'
    )
    ts.index = pd.to_datetime(ts.index)
    if start is not None:
        ts = ts[ts.index >= start]
    if end is not None:
        ts = ts[ts.index <= end]
    ts.drop(
        ["Open", "Low", "High", "Volume", "Adj Close"],
        axis=1, inplace=True
    )
    for i in range(0, lookback_minutes):
        ts["Lookback%s" % str(i + 1)] = ts["Close"].shift(i + 1)
    for i in range(0, lookforward_minutes):
        ts["Lookforward%s" % str(i + 1)] = ts["Close"].shift(-(i + 1))
    ts.dropna(inplace=True)
    ts["Lookback0"] = ts["Close"].pct_change() * 100.0
    for i in range(0, lookback_minutes):
        ts["Lookback%s" % str(i + 1)] = ts["Lookback%s" % str(i + 1)].pct_change() * 100.0
    for i in range(0, lookforward_minutes):
        ts["Lookforward%s" % str(i + 1)] = ts["Lookforward%s" % str(i + 1)].pct_change() * 100.0
    ts.dropna(inplace=True)
    up = up_down_factor * percent_factor
    down = percent_factor
    down_cols = [
        ts["Lookforward%s" % str(i + 1)] > -down
        for i in range(0, lookforward_minutes)
    ]
    up_cols = [
        ts["Lookforward%s" % str(i + 1)] > up
        for i in range(0, lookforward_minutes)
    ]
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

if __name__ == "__main__":
    random_state = 42
    n_estimators = 400
    n_jobs = 1
    csv_filepath = "data/SPY.csv"
    lookback_minutes = 30
    lookforward_minutes = 5
    print("Importing and creating CSV DataFrame...")
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2024, 1, 1)
    ts = create_up_down_dataframe(
        csv_filepath,
        lookback_minutes=lookback_minutes,
        lookforward_minutes=lookforward_minutes,
        start=start_date, end=end_date
    )
    print("Preprocessing data...")
    X = ts[
        [
            "Lookback%s" % str(i)
            for i in range(0, 5)
        ]
    ]
    y = ts["UpDown"]
    print("Creating train/test split of data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    print("Fitting classifier model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        max_depth=10
    )
    model.fit(X_train, y_train)
    print("Outputting metrics...")
    print("Hit-Rate: %s" % model.score(X_test, y_test))
    print("%s\n" % confusion_matrix(model.predict(X_test), y_test))
    print("Pickling model...")
    joblib.dump(model, 'ml_model_rf.pkl')
