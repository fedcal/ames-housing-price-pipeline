"""Data wrangling: missing values e outlier handling.

Distinzione critica: per Ames Housing molti `NaN` non sono dati mancanti
ma rappresentano l'ASSENZA di una caratteristica (es. `PoolQC=NaN` →
"casa senza piscina"). Questi vanno mappati a una categoria 'None',
NON imputati. Solo poche colonne (es. `LotFrontage`, `MasVnrArea`)
hanno mancanti "veri" che vanno imputati con statistiche.
"""
from __future__ import annotations

import logging
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)


# Categoriche dove NaN = "feature non presente". Da De Cock (2011),
# DataDocumentation.txt: questi attributi descrivono componenti
# opzionali della casa (piscina, garage, scantinato, ...).
NA_AS_NONE_CATEGORICAL: Final[tuple[str, ...]] = (
    "Alley",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature",
    "MasVnrType",
)

# Numeriche dove NaN deriva dalla stessa logica strutturale (assenza
# del componente): vanno imputate a 0, non con mediana.
NA_AS_ZERO_NUMERIC: Final[tuple[str, ...]] = (
    "GarageYrBlt",
    "MasVnrArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
    "GarageArea", "GarageCars",
)

# Numeriche con missing "veri" (errori di rilevazione, non strutturali).
# Vanno imputate con la mediana del training set (no leakage: l'imputer
# di sklearn lo fa correttamente fuori dal nostro codice).
NA_AS_MEDIAN_NUMERIC: Final[tuple[str, ...]] = (
    "LotFrontage",
)


def fill_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Mappa i NaN "strutturali" alle categorie corrette PRIMA del split.

    Questo NON è data leakage: stiamo solo decodificando il significato
    semantico del NaN secondo la documentazione del dataset (NaN ='non
    presente'), una trasformazione nota a priori.
    L'imputazione statistica delle altre colonne va invece fatta dentro
    la pipeline sklearn per evitare leakage.

    Args:
        df: DataFrame con valori NaN da risolvere.

    Returns:
        Nuovo DataFrame (immutabilità: l'input non viene modificato).
    """
    out = df.copy()
    for col in NA_AS_NONE_CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].fillna("None")
    for col in NA_AS_ZERO_NUMERIC:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    n_remaining = int(out.isna().sum().sum())
    logger.info(
        "fill_structural_missing: rimossi NaN strutturali. Rimangono %d NaN su %d colonne.",
        n_remaining, int((out.isna().sum() > 0).sum()),
    )
    return out


def remove_grliv_area_outliers(df: pd.DataFrame, target_col: str = "SalePrice") -> pd.DataFrame:
    """Rimuove i ~5 outlier raccomandati da De Cock (2011).

    Sono case con `GrLivArea > 4000` ft² ma prezzo basso: vendite
    anomale (es. liquidazioni, vendite parziali) che fuorviano i modelli
    lineari. De Cock le segnala esplicitamente nel paper.

    Returns:
        DataFrame senza gli outlier.
    """
    if "GrLivArea" not in df.columns or target_col not in df.columns:
        return df.copy()
    mask = ~((df["GrLivArea"] > 4000) & (df[target_col] < 300_000))
    n_removed = int((~mask).sum())
    logger.info("remove_grliv_area_outliers: rimossi %d record.", n_removed)
    return df.loc[mask].copy()


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Suddivide le colonne in numeriche, ordinali, nominali.

    Decisione progettuale: la classificazione qui è basata SOLO sui
    dtypes pandas + un set hardcoded di colonne ordinali note dalla
    documentazione di De Cock. Non usiamo cardinalità per inferire
    "categorica vs continua" perché in Ames alcune colonne numeriche
    discrete (es. `OverallQual` ∈ 1..10) sarebbero classificate male.
    """
    ordinal_cols = [
        c for c in (
            "ExterQual", "ExterCond",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "HeatingQC",
            "KitchenQual",
            "FireplaceQu",
            "GarageQual", "GarageCond", "GarageFinish",
            "PoolQC",
            "LotShape",
            "Utilities",
            "LandSlope",
            "PavedDrive",
        )
        if c in df.columns
    ]
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in ordinal_cols
    ]
    nominal_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in ordinal_cols
    ]
    return {
        "numeric": numeric_cols,
        "ordinal": ordinal_cols,
        "nominal": nominal_cols,
    }


__all__ = [
    "NA_AS_NONE_CATEGORICAL",
    "NA_AS_ZERO_NUMERIC",
    "NA_AS_MEDIAN_NUMERIC",
    "fill_structural_missing",
    "remove_grliv_area_outliers",
    "get_column_groups",
]
