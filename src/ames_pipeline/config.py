"""Configurazione globale della pipeline Ames Housing.

Centralizza path, costanti e iperparametri di default.
Mantiene il codice pulito (no magic numbers/path sparsi) e
facilita l'esecuzione riproducibile da CLI o notebook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = REPORTS_DIR / "models"

DATASET_URL: str = "https://jse.amstat.org/v19n3/decock/AmesHousing.txt"
DATASET_FILENAME: str = "AmesHousing.txt"
DATASET_SHA256: str = (
    "6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc"
)

TARGET_COLUMN: str = "SalePrice"
ID_COLUMNS: tuple[str, ...] = ("Order", "PID")

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5


# Mappa qualità ordinali (Po=Poor, Fa=Fair, TA=Typical/Average, Gd=Good, Ex=Excellent).
# Convenzione di De Cock; ordinata DAL PEGGIORE AL MIGLIORE.
QUALITY_ORDER: list[str] = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

EXPOSURE_ORDER: list[str] = ["None", "No", "Mn", "Av", "Gd"]
FINISH_ORDER: list[str] = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
SLOPE_ORDER: list[str] = ["Sev", "Mod", "Gtl"]
SHAPE_ORDER: list[str] = ["IR3", "IR2", "IR1", "Reg"]
UTILITIES_ORDER: list[str] = ["ELO", "NoSeWa", "NoSewr", "AllPub"]
LANDSLOPE_ORDER: list[str] = ["Sev", "Mod", "Gtl"]
PAVEDDRIVE_ORDER: list[str] = ["N", "P", "Y"]


@dataclass(frozen=True)
class PipelineConfig:
    """Iperparametri e flag della pipeline.

    Frozen=True per evitare mutazioni accidentali dopo l'inizializzazione
    (immutabilità: ogni esperimento crea un proprio config).
    """
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    log_transform_target: bool = True
    drop_outliers_grliv_area: bool = True
    n_jobs: int = -1
    verbose: int = 1


DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


# Iperparametri per il tuning. Tenuti volutamente piccoli per consentire
# l'esecuzione in tempi didattici (~10 min) su laptop. Ampliarli per
# produzione/competizioni.
LINEAR_PARAM_GRID: dict[str, list] = {
    "model__alpha": [0.1, 1.0, 5.0, 10.0, 30.0, 100.0],
}

RF_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 12, 20],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", 0.5],
}

XGB_PARAM_GRID: dict[str, list] = {
    "model__n_estimators": [400, 800],
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
    "model__reg_alpha": [0.0, 0.1],
    "model__reg_lambda": [1.0, 5.0],
}


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "MODELS_DIR",
    "DATASET_URL",
    "DATASET_FILENAME",
    "DATASET_SHA256",
    "TARGET_COLUMN",
    "ID_COLUMNS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "CV_FOLDS",
    "QUALITY_ORDER",
    "EXPOSURE_ORDER",
    "FINISH_ORDER",
    "SLOPE_ORDER",
    "SHAPE_ORDER",
    "UTILITIES_ORDER",
    "PAVEDDRIVE_ORDER",
    "LANDSLOPE_ORDER",
    "PipelineConfig",
    "DEFAULT_CONFIG",
    "LINEAR_PARAM_GRID",
    "RF_PARAM_GRID",
    "XGB_PARAM_GRID",
]
