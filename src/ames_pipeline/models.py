"""Definizione dei modelli candidati e delle pipeline complete.

Tre famiglie di modelli con caratteristiche complementari:

- **Ridge** (lineare regolarizzato): baseline interpretabile. Richiede
  scaling e codifiche dummy. Robusto a multicollinearità grazie alla
  regolarizzazione L2.

- **RandomForest** (ensemble di tree): baseline non lineare. Cattura
  interazioni e non lineari senza scaling. Tendenzialmente sotto-performa
  i gradient boosting su tabular ma è più stabile.

- **XGBoost** (gradient boosting): tipicamente lo state-of-the-art su
  dataset tabulari di queste dimensioni. Più sensibile al tuning.

Ognuno è esposto come `sklearn.pipeline.Pipeline` con i suoi step di
preprocessing aggiuntivi (es. scaling solo per Ridge), così che le
metriche di cross-validation siano sempre fair (no leakage).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .config import RANDOM_STATE


def linear_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Ridge regression con scaling sulle feature post-preprocessing.

    Lo `StandardScaler` viene applicato DOPO il `ColumnTransformer`:
    a quel punto tutte le colonne sono numeriche (ordinal/one-hot
    incluse) e possono essere scalate uniformemente. Questo ordine è
    importante: scalare prima del one-hot non avrebbe senso.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False perché OHE è sparso/zero-inflated
        ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
    ])


def random_forest_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Random Forest (no scaling necessario)."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_split=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def xgboost_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """XGBoost regressor con iperparametri di partenza ragionevoli.

    `tree_method='hist'` è il backend più veloce su CPU per dataset
    di questa scala. `early_stopping_rounds` non è impostato qui perché
    GridSearchCV non lo supporta nativamente; il tuning lo gestisce in
    `tuning.py`.
    """
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBRegressor(
                n_estimators=600,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method="hist",
                # Verboso disabilitato: GridSearchCV stamperebbe altrimenti
                # migliaia di righe.
                verbosity=0,
            ),
        ),
    ])


def get_all_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Restituisce tutte le pipeline candidate in un dict ordinato (nome → pipeline)."""
    return {
        "Ridge":         linear_pipeline(preprocessor),
        "RandomForest":  random_forest_pipeline(preprocessor),
        "XGBoost":       xgboost_pipeline(preprocessor),
    }


__all__ = [
    "linear_pipeline",
    "random_forest_pipeline",
    "xgboost_pipeline",
    "get_all_pipelines",
]
