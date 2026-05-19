import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

def train_lgbm(X_train, y_train):

    model = lgb.LGBMRegressor(random_state=42)

    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7, None],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [20, 31, 40],
        'min_child_samples': [5, 10, 20]
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
