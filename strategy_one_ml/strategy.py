from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from strategy_one_ml.retrive_data import ts

lookback_minutes = 30
lookforward_minutes = 5
random_state = 42
n_estimators = 400
n_jobs = 1

X = ts[[f"Lookback{i}" for i in range(lookback_minutes)]]
y = ts["UpDown"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, max_depth=10)
model.fit(X_train, y_train)

print("Hit-Rate: %s" % model.score(X_test, y_test))
print(confusion_matrix(model.predict(X_test), y_test))

joblib.dump(model, 'ml_model_rf.pkl')
