"""Orchestratore end-to-end della pipeline Ames Housing.

Esegue, nell'ordine:
    1. Download/caricamento dataset.
    2. Wrangling (NaN strutturali + rimozione outlier opzionale).
    3. Train/test split.
    4. Costruzione preprocessor + feature engineering + pipeline candidate.
    5. Tuning K-fold di tutti i modelli.
    6. Valutazione holdout test set.
    7. Selezione miglior modello.
    8. Persistenza su disco (joblib) + report metriche (CSV/JSON).

Eseguibile come modulo:

    python -m ames_pipeline.pipeline           # full run
    python -m ames_pipeline.pipeline --quick   # tuning ridotto, smoke-test

Le funzioni pubbliche sono pensate anche per essere chiamate da
notebook (`from ames_pipeline.pipeline import run_full_pipeline`).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import (
    LINEAR_PARAM_GRID,
    MODELS_DIR,
    REPORTS_DIR,
    RF_PARAM_GRID,
    XGB_PARAM_GRID,
    DEFAULT_CONFIG,
    PipelineConfig,
)
from .data import load_raw, make_train_test_split
from .evaluation import (
    evaluate_on_holdout,
    plot_predictions_vs_actual,
    plot_residuals,
)
from .features import AmesFeatureEngineer
from .models import get_all_pipelines
from .preprocessing import build_preprocessor, infer_column_groups
from .tuning import (
    TuningResult,
    summarize_tuning,
    tune_all_models,
)
from .wrangling import fill_structural_missing, remove_grliv_area_outliers

logger = logging.getLogger(__name__)


def _attach_feature_engineering(pipelines: dict[str, Pipeline]) -> dict[str, Pipeline]:
    """Antepone `AmesFeatureEngineer` davanti a ogni pipeline esistente.

    Il feature engineer crea TotalSF, HouseAge, ecc.; il preprocessor
    successivo le inferisce automaticamente come numeriche grazie a
    `infer_column_groups`. Inserire l'engineer come step della pipeline
    (e non a mano sul DataFrame) è essenziale per il `cross_val_score`:
    deve girare sui fold di train senza vedere il test.
    """
    out: dict[str, Pipeline] = {}
    for name, pipe in pipelines.items():
        steps = [("feature_engineer", AmesFeatureEngineer())] + list(pipe.steps)
        out[name] = Pipeline(steps=steps)
    return out


def prepare_data(
    config: PipelineConfig = DEFAULT_CONFIG,
    drop_outliers: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Pipeline pre-modello: load + wrangling + split.

    Returns:
        (X_train, X_test, y_train, y_test, df_full_after_wrangling).
        L'ultimo elemento è utile per EDA in notebook.
    """
    drop_outliers = config.drop_outliers_grliv_area if drop_outliers is None else drop_outliers

    df = load_raw()
    df = fill_structural_missing(df)
    if drop_outliers:
        df = remove_grliv_area_outliers(df)

    X_train, X_test, y_train, y_test = make_train_test_split(df, config)
    logger.info(
        "Split: train=%d, test=%d, ratio=%.2f",
        len(X_train), len(X_test), len(X_test) / (len(X_train) + len(X_test)),
    )
    return X_train, X_test, y_train, y_test, df


def build_candidate_pipelines(X_train_after_fe: pd.DataFrame) -> dict[str, Pipeline]:
    """Costruisce tutte le pipeline candidate.

    Args:
        X_train_after_fe: DataFrame X_train DOPO `AmesFeatureEngineer`,
            usato solo per inferire i gruppi di colonne.
    """
    groups = infer_column_groups(X_train_after_fe)
    preprocessor = build_preprocessor(
        numeric_cols=groups["numeric"],
        ordinal_cols=groups["ordinal"],
        nominal_cols=groups["nominal"],
    )
    base_pipelines = get_all_pipelines(preprocessor)
    return _attach_feature_engineering(base_pipelines)


def select_best_model(tuning_results: dict[str, TuningResult]) -> tuple[str, TuningResult]:
    """Sceglie il modello con il miglior RMSE log su CV."""
    best_name = min(tuning_results, key=lambda k: tuning_results[k].best_score)
    return best_name, tuning_results[best_name]


def save_artifacts(
    best_name: str,
    best_estimator,
    holdout_metrics: dict,
    cv_summary: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Path]:
    """Persiste modello + report metriche."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Modello best (sempre salvato anche con il nome generico per inference.py).
    model_named_path = models_dir / f"{best_name.lower()}_best.joblib"
    model_default_path = models_dir / "best_model.joblib"
    joblib.dump(best_estimator, model_named_path)
    joblib.dump(best_estimator, model_default_path)

    # Tabella confronto CV.
    cv_path = reports_dir / "cv_summary.csv"
    cv_summary.to_csv(cv_path, index=False)

    # Metriche holdout.
    metrics_path = reports_dir / "holdout_metrics.json"
    metrics_path.write_text(json.dumps(holdout_metrics, indent=2))

    return {
        "model_named": model_named_path,
        "model_default": model_default_path,
        "cv_summary": cv_path,
        "holdout_metrics": metrics_path,
    }


def run_full_pipeline(
    config: PipelineConfig = DEFAULT_CONFIG,
    quick: bool = False,
) -> dict:
    """Esegue l'intera pipeline e restituisce un dizionario di risultati.

    Args:
        config: configurazione (test_size, cv_folds, random_state, ...).
        quick: se True, riduce la grid di tuning per smoke-test rapido.

    Returns:
        Dizionario con keys:
            - best_model_name
            - best_estimator (sklearn)
            - cv_summary (DataFrame)
            - holdout_metrics (dict per modello)
            - artifacts (dict di Path)
    """
    logger.info("=" * 70)
    logger.info("Avvio pipeline Ames Housing (quick=%s)", quick)
    logger.info("=" * 70)

    # --- 1-2-3. Data preparation ---
    X_train, X_test, y_train, y_test, _ = prepare_data(config)

    # --- 4. Pipeline candidate ---
    # Applichiamo feature engineering una volta sul train SOLO per
    # inferire i gruppi di colonne; durante il fit/predict il transformer
    # è dentro la Pipeline e gira correttamente per fold.
    fe = AmesFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    pipelines = build_candidate_pipelines(X_train_fe)

    # --- 5. Tuning ---
    if quick:
        # Grid ridotte per smoke-test (eseguibile in <2 min su laptop).
        from . import config as cfg
        cfg.LINEAR_PARAM_GRID["model__alpha"] = [10.0]
        cfg.RF_PARAM_GRID["model__n_estimators"] = [100]
        cfg.RF_PARAM_GRID["model__max_depth"] = [12]
        cfg.RF_PARAM_GRID["model__min_samples_split"] = [2]
        cfg.RF_PARAM_GRID["model__max_features"] = ["sqrt"]
        cfg.XGB_PARAM_GRID["model__n_estimators"] = [200]
        cfg.XGB_PARAM_GRID["model__max_depth"] = [5]
        cfg.XGB_PARAM_GRID["model__learning_rate"] = [0.1]
        cfg.XGB_PARAM_GRID["model__subsample"] = [1.0]
        cfg.XGB_PARAM_GRID["model__colsample_bytree"] = [1.0]
        cfg.XGB_PARAM_GRID["model__reg_alpha"] = [0.0]
        cfg.XGB_PARAM_GRID["model__reg_lambda"] = [1.0]
        xgb_n_iter = 2
    else:
        xgb_n_iter = 30

    tuning_results = tune_all_models(
        pipelines=pipelines,
        X=X_train,
        y=y_train,
        config=config,
        xgb_n_iter=xgb_n_iter,
    )
    cv_summary = summarize_tuning(tuning_results)
    logger.info("\nRiepilogo tuning (CV):\n%s", cv_summary.to_string(index=False))

    # --- 6. Holdout evaluation ---
    holdout_metrics: dict = {}
    for name, result in tuning_results.items():
        m = evaluate_on_holdout(result.best_estimator, X_test, y_test)
        holdout_metrics[name] = m.as_dict()

    holdout_table = pd.DataFrame(holdout_metrics).T.sort_values("rmse")
    holdout_table.index.name = "model"
    logger.info("\nRiepilogo holdout test:\n%s", holdout_table.to_string())

    # --- 7. Selection ---
    best_name, best_result = select_best_model(tuning_results)
    logger.info("\n>>> Miglior modello: %s (RMSE log CV = %.4f)\n",
                best_name, best_result.best_score)

    # --- 8. Persist ---
    artifacts = save_artifacts(
        best_name=best_name,
        best_estimator=best_result.best_estimator,
        holdout_metrics=holdout_metrics,
        cv_summary=cv_summary,
    )

    # Plot diagnostici per il miglior modello (best-effort, non bloccante).
    try:
        from .config import FIGURES_DIR
        y_pred_best = best_result.best_estimator.predict(X_test)
        plot_predictions_vs_actual(
            y_test.to_numpy(), y_pred_best,
            title=f"{best_name}: predizioni vs reali (test)",
            save_path=FIGURES_DIR / f"{best_name.lower()}_pred_vs_actual.png",
        )
        plot_residuals(
            y_test.to_numpy(), y_pred_best,
            title=f"{best_name}: distribuzione residui (test)",
            save_path=FIGURES_DIR / f"{best_name.lower()}_residuals.png",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Plot diagnostici saltati: %s", exc)

    return {
        "best_model_name": best_name,
        "best_estimator": best_result.best_estimator,
        "cv_summary": cv_summary,
        "holdout_metrics": holdout_metrics,
        "holdout_table": holdout_table,
        "artifacts": artifacts,
        "tuning_results": tuning_results,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ames Housing — pipeline end-to-end.")
    parser.add_argument(
        "--quick", action="store_true",
        help="Tuning ridotto per smoke-test (~1-2 min).",
    )
    parser.add_argument(
        "--no-outliers-removal", action="store_true",
        help="Disabilita la rimozione degli outlier GrLivArea (per ablation).",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    args = _build_arg_parser().parse_args()
    config = DEFAULT_CONFIG
    if args.no_outliers_removal:
        config = PipelineConfig(
            random_state=DEFAULT_CONFIG.random_state,
            test_size=DEFAULT_CONFIG.test_size,
            cv_folds=DEFAULT_CONFIG.cv_folds,
            log_transform_target=DEFAULT_CONFIG.log_transform_target,
            drop_outliers_grliv_area=False,
            n_jobs=DEFAULT_CONFIG.n_jobs,
            verbose=DEFAULT_CONFIG.verbose,
        )
    run_full_pipeline(config=config, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "prepare_data",
    "build_candidate_pipelines",
    "select_best_model",
    "save_artifacts",
    "run_full_pipeline",
]
