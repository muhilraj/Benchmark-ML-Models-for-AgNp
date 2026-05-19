from preprocessing import load_and_preprocess
from train_xgb import train_xgb
from train_lgbm import train_lgbm
from train_catboost import train_catboost
from train_hgb import train_hgb
from evaluation import evaluate_model
from plotting import actual_vs_predicted
from shap_analysis import run_shap

X_train, X_test, y_train, y_test = load_and_preprocess(
    "data/sample_dataset.csv"
)

models = {
    "XGBoost": train_xgb(X_train, y_train),
    "LightGBM": train_lgbm(X_train, y_train),
    "CatBoost": train_catboost(X_train, y_train),
    "HistGradientBoosting": train_hgb(X_train, y_train)
}

results = {}

for name, model in models.items():

    metrics, predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[name] = metrics

    actual_vs_predicted(
        y_test,
        predictions,
        name
    )

best_model_name = max(results, key=lambda x: results[x]["R2"])

best_model = models[best_model_name]

run_shap(
    best_model.best_estimator_,
    X_train,
    best_model_name
)

print(results)
