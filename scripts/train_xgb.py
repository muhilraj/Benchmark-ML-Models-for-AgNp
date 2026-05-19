import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def train_xgb(X_train, y_train):

    model = xgb.XGBRegressor(random_state=42)

    param_dist = {
        'n_estimators': [100, 300, 500, 700],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_lambda': [10, 50, 100],
        'reg_alpha': [10, 50, 100]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=5,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search
