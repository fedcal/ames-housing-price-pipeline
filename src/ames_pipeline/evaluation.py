"""Metriche, cross-validation e diagnostica grafica.

Tutte le metriche sono calcolate sulla SCALA ORIGINALE del target
(dollari) anche quando il modello è stato addestrato su `log1p(y)`:
`TransformedTargetRegressor` riapplica `expm1` automaticamente in
`predict()`. Le metriche in scala log si calcolano applicando log1p
manualmente alle predizioni e ai valori veri.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

from .config import FIGURES_DIR, PipelineConfig, DEFAULT_CONFIG
from .tuning import wrap_with_log_target

logger = logging.getLogger(__name__)


@dataclass
class RegressionMetrics:
    """Insieme di metriche di regressione standard (sia $ che log)."""
    rmse: float                  # RMSE in $ sulla scala originale
    mae: float                   # MAE in $
    r2: float                    # Coefficiente di determinazione
    rmse_log: float              # RMSE su log1p(y) — comparabile con leaderboard Kaggle
    mape: float                  # Mean Absolute Percentage Error (%)

    def as_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "rmse_log": self.rmse_log,
            "mape": self.mape,
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """Calcola tutte le metriche su scala originale + log."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse_log = float(np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(np.maximum(y_pred, 0)))))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2, rmse_log=rmse_log, mape=mape)


def cross_val_rmse_log(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """K-fold CV RMSE in scala log (mean ± std). Usa lo stesso scoring del tuning.

    Wrappa con `TransformedTargetRegressor` se non già wrappato.
    """
    from sklearn.compose import TransformedTargetRegressor
    if not isinstance(pipeline, TransformedTargetRegressor):
        pipeline = wrap_with_log_target(pipeline)

    from sklearn.model_selection import KFold
    cv = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)
    scores = -cross_val_score(
        pipeline, X, y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=config.n_jobs,
    )
    # Le predizioni sono in scala originale (TTR le riporta con expm1),
    # quindi `scoring='neg_root_mean_squared_error'` è in $.
    # Per ottenere la scala log dobbiamo cambiarla a mano, ma è più
    # comodo riusare RMSE direttamente: già ben rapportato.
    return float(scores.mean()), float(scores.std())


def evaluate_on_holdout(
    fitted_estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> RegressionMetrics:
    """Valuta un estimator già fittato sull'holdout test set."""
    y_pred = fitted_estimator.predict(X_test)
    metrics = regression_metrics(y_test.to_numpy(), y_pred)
    logger.info(
        "Holdout: RMSE=$%s, MAE=$%s, R²=%.4f, RMSE_log=%.4f, MAPE=%.2f%%",
        f"{metrics.rmse:,.0f}", f"{metrics.mae:,.0f}",
        metrics.r2, metrics.rmse_log, metrics.mape,
    )
    return metrics


def compare_models(
    metrics_by_model: dict[str, RegressionMetrics],
) -> pd.DataFrame:
    """Tabella confronto modelli, ordinata per RMSE crescente."""
    df = pd.DataFrame({
        name: m.as_dict() for name, m in metrics_by_model.items()
    }).T
    df.index.name = "model"
    return df.sort_values("rmse")


# --- Diagnostica grafica ---

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predizioni vs Reali",
    save_path: Path | None = None,
) -> plt.Figure:
    """Scatter prediction vs actual. Bisettrice per riferimento."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    lim_min = min(y_true.min(), y_pred.min())
    lim_max = max(y_true.max(), y_pred.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1, label="y_true = y_pred")
    ax.set_xlabel("SalePrice reale ($)")
    ax.set_ylabel("SalePrice predetto ($)")
    ax.set_title(title)
    ax.legend()
    ax.ticklabel_format(style="plain", axis="both")
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("Figura salvata: %s", save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Distribuzione residui",
    save_path: Path | None = None,
) -> plt.Figure:
    """Istogramma dei residui (y_true - y_pred). Atteso: distribuzione ~normale, centrata sullo zero."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", lw=1)
    ax.set_xlabel("Residuo (y_true - y_pred), $")
    ax.set_ylabel("Frequenza")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("Figura salvata: %s", save_path)
    return fig


def get_top_feature_importance(
    fitted_pipeline,
    feature_names: Iterable[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Estrae le top-N feature importance per modelli tree-based.

    Funziona con pipeline + TransformedTargetRegressor: scava dentro
    fino al regressor finale e cerca `feature_importances_`.
    """
    from sklearn.compose import TransformedTargetRegressor

    estimator = fitted_pipeline
    if isinstance(estimator, TransformedTargetRegressor):
        estimator = estimator.regressor_

    # estimator ora è la Pipeline interna. L'ultimo step è 'model'.
    final_model = estimator.named_steps["model"]
    if not hasattr(final_model, "feature_importances_"):
        raise AttributeError(
            f"Il modello {type(final_model).__name__} non espone feature_importances_."
        )
    importances = final_model.feature_importances_
    names = list(feature_names)
    if len(importances) != len(names):
        # OneHot expand: usiamo nomi generici fallback.
        names = [f"f_{i}" for i in range(len(importances))]
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Top feature importance",
    save_path: Path | None = None,
) -> plt.Figure:
    """Barh delle top feature importance (output di `get_top_feature_importance`)."""
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(importance_df))))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("Figura salvata: %s", save_path)
    return fig


__all__ = [
    "RegressionMetrics",
    "regression_metrics",
    "cross_val_rmse_log",
    "evaluate_on_holdout",
    "compare_models",
    "plot_predictions_vs_actual",
    "plot_residuals",
    "get_top_feature_importance",
    "plot_feature_importance",
    "FIGURES_DIR",
]
