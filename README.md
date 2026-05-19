# Benchmark-ML-Models-for-AgNp
This repository contains representative implementations of the machine learning workflows used in the study, including:

- data preprocessing
- model benchmarking
- hyperparameter optimization
- regression performance evaluation
- SHAP-based explainability analysis

## Included Models

- XGBoost
- LightGBM
- CatBoost
- HistGradientBoostingRegressor

## Repository Structure

scripts/
    preprocessing.py
    train_xgb.py
    train_lgbm.py
    train_catboost.py
    train_hgb.py
    evaluation.py
    shap_analysis.py
    plotting.py
    run_all_models.py

data/
    sample_dataset.csv

outputs/
    generated figures and model outputs

## Usage

Run:

python scripts/run_all_models.py

## Notes

- Users may need to adapt preprocessing and feature configurations according to their local dataset structure.
- The repository contains representative workflows and does not include auxiliary exploratory scripts used during method development.
- Some preprocessing assumptions may require adjustment depending on dataset formatting.

## Environment

Tested using:
- Python 3.10
