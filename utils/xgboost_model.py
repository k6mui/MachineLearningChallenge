import xgboost as xgb

# Función de entrenamiento del XGboosts
def train_xg(X_train, y_train):
    xg = xgb.XGBClassifier(
        n_estimators = 2000,
        learning_rate = 0.01,
        gamma = 0.2,
        objective ='softmax',
        num_class = 7, 
        random_state = 42,
        n_jobs = -1,
        max_depth = 40,
        verbosity = 2
    )
    xg.fit(X_train, y_train)
    return xg
