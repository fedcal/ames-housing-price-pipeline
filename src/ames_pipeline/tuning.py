"""Hyperparameter tuning con K-fold cross-validation.

Per i modelli con grid piccola (Ridge, RF) usiamo `GridSearchCV` —
esauriente, deterministico. Per XGBoost, dove la grid è combinatoria
(>100 combinazioni) usiamo `RandomizedSearchCV` che campiona uniformemente
e ottiene risultati comparabili in tempi accettabili.

**Punto chiave didattico**: il tuning ottimizza la metrica di scoring
sulla scala log del target. Lo trasformiamo dentro `TransformedTargetRegressor`
così che K-fold rispetti `np.log1p` come parte della pipeline e tutto
sia gestito automaticamente.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .config import (
    LINEAR_PARAM_GRID,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
    PipelineConfig,
    DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)


def wrap_with_log_target(estimator: Pipeline) -> TransformedTargetRegressor:
    """Avvolge un estimator con log1p/expm1 sul target.

    Perché log1p+expm1 e non log+exp:
        - log1p(y) = log(1 + y) gestisce y=0 (non capita su SalePrice ma
          è una buona pratica difensiva).
        - expm1 è l'inversa esatta.

    Perché DENTRO la pipeline e non a mano:
        Le metriche di CV vengono calcolate sulla scala originale,
        gestendo automaticamente la trasformazione. Esponendo il target
        log all'esterno avremmo dovuto invertire ovunque a mano,
        introducendo bug sottili.
    """
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,  # log1p/expm1 non sono perfettamente invertibili in float64 ai bordi.
    )


def _prefix_param_grid(grid: dict[str, list], prefix: str = "regressor__") -> dict[str, list]:
    """`TransformedTargetRegressor` espone l'estimator interno come `regressor__*`.

    Aggiunge il prefisso a tutte le chiavi della grid. Esempio:
        {'model__alpha': [...]} → {'regressor__model__alpha': [...]}
    """
    return {f"{prefix}{k}": v for k, v in grid.items()}


@dataclass
class TuningResult:
    """Risultato del tuning di un singolo modello."""
    model_name: str
    best_estimator: TransformedTargetRegressor
    best_params: dict[str, Any]
    best_score: float                     # negative RMSE log scale (sklearn convention)
    cv_results: dict[str, Any]
    duration_seconds: float


def _make_cv(config: PipelineConfig) -> KFold:
    """K-fold semplice con shuffle. Niente stratificazione (regressione)."""
    return KFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )


def tune_grid(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> TuningResult:
    """Tuning esauriente via GridSearchCV su grid prefissata.

    `scoring='neg_root_mean_squared_error'` ottimizza l'RMSE (negato
    perché sklearn massimizza). Calcolato sulla scala log del target.
    """
    target_aware_pipeline = wrap_with_log_target(pipeline)
    grid = _prefix_param_grid(param_grid)

    search = GridSearchCV(
        estimator=target_aware_pipeline,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=_make_cv(config),
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - t0
    logger.info("[%s] tuning completato in %.1fs. Best score (RMSE log)=%.4f",
                name, duration, -search.best_score_)
    return TuningResult(
        model_name=name,
        best_estimator=search.best_estimator_,
        best_params={k.replace("regressor__", ""): v for k, v in search.best_params_.items()},
        best_score=-search.best_score_,
        cv_results=search.cv_results_,
        duration_seconds=duration,
    )


def tune_random(
    name: str,
    pipeline: Pipeline,
    param_grid: dict[str, list],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> TuningResult:
    """RandomizedSearchCV: campionamento uniforme dalla grid.

    Più rapido di GridSearch quando lo spazio è grande, e in pratica
    raggiunge spesso il 95-99% del best score della Grid completa.
    """
    target_aware_pipeline = wrap_with_log_target(pipeline)
    grid = _prefix_param_grid(param_grid)

    search = RandomizedSearchCV(
        estimator=target_aware_pipeline,
        param_distributions=grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=_make_cv(config),
        n_jobs=config.n_jobs,
        verbose=config.verbose,
        refit=True,
        random_state=config.random_state,
        return_train_score=True,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    duration = time.perf_counter() - t0
    logger.info("[%s] random search completata in %.1fs. Best score (RMSE log)=%.4f",
                name, duration, -search.best_score_)
    return TuningResult(
        model_name=name,
        best_estimator=search.best_estimator_,
        best_params={k.replace("regressor__", ""): v for k, v in search.best_params_.items()},
        best_score=-search.best_score_,
        cv_results=search.cv_results_,
        duration_seconds=duration,
    )


def tune_all_models(
    pipelines: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
    xgb_n_iter: int = 30,
) -> dict[str, TuningResult]:
    """Esegue il tuning su tutti i modelli registrati.

    Strategia per modello:
        - Ridge      → GridSearchCV (grid piccola, esauriente).
        - RandomForest → GridSearchCV.
        - XGBoost    → RandomizedSearchCV (grid grande, efficienza).
    """
    results: dict[str, TuningResult] = {}
    for name, pipeline in pipelines.items():
        if name == "Ridge":
            results[name] = tune_grid(name, pipeline, LINEAR_PARAM_GRID, X, y, config)
        elif name == "RandomForest":
            results[name] = tune_grid(name, pipeline, RF_PARAM_GRID, X, y, config)
        elif name == "XGBoost":
            results[name] = tune_random(
                name, pipeline, XGB_PARAM_GRID, X, y,
                n_iter=xgb_n_iter, config=config,
            )
        else:
            raise ValueError(f"Tuning strategy non definita per modello '{name}'.")
    return results


def summarize_tuning(results: dict[str, TuningResult]) -> pd.DataFrame:
    """Tabella di riepilogo ordinata per RMSE crescente."""
    rows = [
        {
            "model": r.model_name,
            "rmse_log_cv": r.best_score,
            "duration_s": r.duration_seconds,
            "best_params": r.best_params,
        }
        for r in results.values()
    ]
    return pd.DataFrame(rows).sort_values("rmse_log_cv").reset_index(drop=True)


__all__ = [
    "TuningResult",
    "wrap_with_log_target",
    "tune_grid",
    "tune_random",
    "tune_all_models",
    "summarize_tuning",
]
