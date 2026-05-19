from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_hgb(X_train, y_train):

    model = HistGradientBoostingRegressor(random_state=42)

    param_dist = {
        'max_iter': [100, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_regularization': [1, 3, 5]
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
