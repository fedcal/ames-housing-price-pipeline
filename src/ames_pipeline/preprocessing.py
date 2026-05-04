"""Costruzione del preprocessor sklearn (ColumnTransformer).

Il preprocessor combina tre branch parallele:

- **Numeriche**: imputazione mediana (per `LotFrontage` e residui).
- **Ordinali**: imputazione 'None' + `OrdinalEncoder` con ordinamento
  semantico (es. Po<Fa<TA<Gd<Ex). Il modello usa la distanza ordinale
  reale, non un codice arbitrario.
- **Nominali**: imputazione 'None' + `OneHotEncoder(handle_unknown='ignore')`.

Tutto è contenuto in un `ColumnTransformer` per due ragioni:

1. **No data leakage**: l'imputer/encoder calcolano statistiche solo
   sul training set e le riapplicano in test/inferenza.
2. **Riproducibilità**: serializzando il `ColumnTransformer` con
   joblib/pickle, l'inferenza è bit-identica al training.
"""
from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import (
    EXPOSURE_ORDER,
    FINISH_ORDER,
    LANDSLOPE_ORDER,
    PAVEDDRIVE_ORDER,
    QUALITY_ORDER,
    SHAPE_ORDER,
    UTILITIES_ORDER,
)

logger = logging.getLogger(__name__)


# Mapping colonna → ordine semantico delle categorie. Tutte le ordinali
# di Ames sono "qualità decrescenti" o equivalenti.
ORDINAL_CATEGORIES_MAP: dict[str, list[str]] = {
    "ExterQual":     QUALITY_ORDER,
    "ExterCond":     QUALITY_ORDER,
    "BsmtQual":      QUALITY_ORDER,
    "BsmtCond":      QUALITY_ORDER,
    "BsmtExposure":  EXPOSURE_ORDER,
    "BsmtFinType1":  FINISH_ORDER,
    "BsmtFinType2":  FINISH_ORDER,
    "HeatingQC":     QUALITY_ORDER,
    "KitchenQual":   QUALITY_ORDER,
    "FireplaceQu":   QUALITY_ORDER,
    "GarageQual":    QUALITY_ORDER,
    "GarageCond":    QUALITY_ORDER,
    "GarageFinish":  ["None", "Unf", "RFn", "Fin"],
    "PoolQC":        QUALITY_ORDER,
    "LotShape":      SHAPE_ORDER,
    "Utilities":     UTILITIES_ORDER,
    "LandSlope":     LANDSLOPE_ORDER,
    "PavedDrive":    PAVEDDRIVE_ORDER,
}


def _build_ordinal_branch(ordinal_cols: Sequence[str]) -> Pipeline:
    """Branch ordinale: imputer 'None' + OrdinalEncoder con categorie esplicite."""
    categories = [ORDINAL_CATEGORIES_MAP[col] for col in ordinal_cols]
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        (
            "encoder",
            OrdinalEncoder(
                categories=categories,
                # Categorie inattese in test/inferenza vengono codificate
                # come -1 invece di sollevare eccezione: robusto in
                # produzione, ma da monitorare (drift dati).
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            ),
        ),
    ])


def _build_nominal_branch() -> Pipeline:
    """Branch nominale: imputer 'None' + OneHotEncoder."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                # min_frequency rimuove categorie ultra-rare (≤1
                # occorrenza nel training): riduce dimensionalità e
                # rumore senza perdere segnale.
                min_frequency=2,
            ),
        ),
    ])


def _build_numeric_branch() -> Pipeline:
    """Branch numerica: imputer mediana. Lo scaling viene applicato
    SOLO nelle pipeline che lo richiedono (lineari)."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])


def build_preprocessor(
    numeric_cols: Sequence[str],
    ordinal_cols: Sequence[str],
    nominal_cols: Sequence[str],
) -> ColumnTransformer:
    """Costruisce il `ColumnTransformer` completo dato il grouping di colonne.

    Args:
        numeric_cols: colonne numeriche continue/discrete senza ordine implicito.
        ordinal_cols: colonne con ordinamento semantico (qualità, esposizione, ...).
        nominal_cols: colonne categoriche senza ordine (quartiere, tipologia, ...).

    Returns:
        ColumnTransformer parametrizzato. Va `fit` su X_train e poi
        `transform` su X_test e dati di inferenza.
    """
    transformers: list[tuple] = []
    if numeric_cols:
        transformers.append(("num", _build_numeric_branch(), list(numeric_cols)))
    if ordinal_cols:
        transformers.append(("ord", _build_ordinal_branch(ordinal_cols), list(ordinal_cols)))
    if nominal_cols:
        transformers.append(("nom", _build_nominal_branch(), list(nominal_cols)))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Esplicito: ogni colonna non gestita viene scartata.
        verbose_feature_names_out=False,
    )
    logger.info(
        "Preprocessor pronto: %d numeriche, %d ordinali, %d nominali.",
        len(numeric_cols), len(ordinal_cols), len(nominal_cols),
    )
    return preprocessor


def infer_column_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Inferisce i tre gruppi di colonne dal DataFrame post-feature-engineering.

    Le colonne ordinali sono quelle nella mappa `ORDINAL_CATEGORIES_MAP`
    e presenti in `X`; le nominali sono le rimanenti `object`/`category`;
    le numeriche sono i dtype numerici residui.
    """
    ordinal_cols = [c for c in ORDINAL_CATEGORIES_MAP if c in X.columns]
    nominal_cols = [
        c for c in X.select_dtypes(include=["object", "category"]).columns
        if c not in ordinal_cols
    ]
    numeric_cols = [
        c for c in X.select_dtypes(include="number").columns
        if c not in ordinal_cols
    ]
    return {
        "numeric": numeric_cols,
        "ordinal": ordinal_cols,
        "nominal": nominal_cols,
    }


__all__ = [
    "ORDINAL_CATEGORIES_MAP",
    "build_preprocessor",
    "infer_column_groups",
]
