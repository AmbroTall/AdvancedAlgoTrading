def retrain_model(df, model_filepath='ml_model_rf.pkl'):
    ts = create_up_down_dataframe(df)
    X = ts[[f"Lookback{i}" for i in range(lookback_minutes)]]
    y = ts["UpDown"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, max_depth=10)
    model.fit(X_train, y_train)

    print("Hit-Rate: %s" % model.score(X_test, y_test))
    print(confusion_matrix(model.predict(X_test), y_test))

    joblib.dump(model, model_filepath)

# Schedule this script to run periodically using a task scheduler or cron job
