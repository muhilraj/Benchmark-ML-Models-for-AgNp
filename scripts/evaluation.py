import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "R2": round(r2, 3),
        "RMSE": round(rmse, 3)
    }, y_pred
