"""Feature engineering domain-specific per Ames Housing.

Tutte le trasformazioni qui sono `BaseEstimator` + `TransformerMixin`
per essere componibili dentro `sklearn.pipeline.Pipeline`. Usare un
transformer (vs una funzione standalone) garantisce due cose:

1. **No leakage**: fit_transform su train, transform su test.
2. **Riproducibilità**: la trasformazione può essere serializzata con
   joblib insieme al modello e riapplicata in inferenza.

Le feature derivate sono ispirate alla letteratura su Ames (Kaggle
notebooks classici di Pedro Marcelino, Serigne, ecc.) e alla logica
del dominio immobiliare.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Aggiunge feature derivate al DataFrame.

    Caratteristiche create:
        - HouseAge:           YrSold - YearBuilt
        - YearsSinceRemodel:  YrSold - YearRemodAdd
        - TotalSF:            superficie totale (basement + 1° + 2° piano)
        - TotalBath:          conteggio bagni pesato (full=1, half=0.5)
        - TotalPorchSF:       somma di portico/veranda
        - HasGarage/HasPool/HasBasement/HasFireplace: flag binarie
        - QualityScore:       OverallQual * OverallCond (interazione)

    Tutte le feature sono progettate per essere interpretabili e per
    catturare informazioni latenti su cui i modelli (specialmente
    quelli lineari) faticano altrimenti.
    """

    def __init__(self, drop_originals: bool = False) -> None:
        # Se True, rimuove le colonne originali dopo aver derivato
        # quelle aggregate (es. droppa 1stFlrSF, 2ndFlrSF, TotalBsmtSF
        # dopo aver creato TotalSF). Default False per non perdere
        # informazione: i modelli non lineari possono comunque
        # imparare le interazioni dalle colonne grezze.
        self.drop_originals = drop_originals

    # sklearn richiede una signature compatibile (X, y=None).
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "AmesFeatureEngineer":
        # Salviamo i nomi delle colonne di input. Necessario perché il
        # primo step di una Pipeline è quello che espone
        # `feature_names_in_` al wrapper esterno (TransformedTargetRegressor),
        # e a sua volta consente alla funzione di inferenza di sapere
        # quali colonne pre-FE servono.
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
            self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Se l'input è un dict/array (non DataFrame), riconverti.
        if not isinstance(X, pd.DataFrame):
            if hasattr(self, "feature_names_in_"):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                raise TypeError("AmesFeatureEngineer richiede un pd.DataFrame in input.")
        out = X.copy()

        # --- Feature di età della casa ---
        if {"YrSold", "YearBuilt"}.issubset(out.columns):
            out["HouseAge"] = out["YrSold"] - out["YearBuilt"]
            # Casi di immobili venduti l'anno stesso della costruzione: 0 → no anomalia.
            # Età negativa è un errore di data entry: clamp a 0.
            out["HouseAge"] = out["HouseAge"].clip(lower=0)

        if {"YrSold", "YearRemodAdd"}.issubset(out.columns):
            out["YearsSinceRemodel"] = (out["YrSold"] - out["YearRemodAdd"]).clip(lower=0)

        # --- Superficie totale ---
        sf_cols = [c for c in ("1stFlrSF", "2ndFlrSF", "TotalBsmtSF") if c in out.columns]
        if sf_cols:
            out["TotalSF"] = out[sf_cols].sum(axis=1)

        # --- Bagni totali pesati ---
        bath_full_cols = [c for c in ("FullBath", "BsmtFullBath") if c in out.columns]
        bath_half_cols = [c for c in ("HalfBath", "BsmtHalfBath") if c in out.columns]
        full = out[bath_full_cols].sum(axis=1) if bath_full_cols else 0
        half = out[bath_half_cols].sum(axis=1) * 0.5 if bath_half_cols else 0
        out["TotalBath"] = full + half

        # --- Portico/veranda totale ---
        porch_cols = [
            c for c in ("OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF")
            if c in out.columns
        ]
        if porch_cols:
            out["TotalPorchSF"] = out[porch_cols].sum(axis=1)

        # --- Flag binarie di presenza ---
        if "GarageArea" in out.columns:
            out["HasGarage"] = (out["GarageArea"] > 0).astype(int)
        if "PoolArea" in out.columns:
            out["HasPool"] = (out["PoolArea"] > 0).astype(int)
        if "TotalBsmtSF" in out.columns:
            out["HasBasement"] = (out["TotalBsmtSF"] > 0).astype(int)
        if "Fireplaces" in out.columns:
            out["HasFireplace"] = (out["Fireplaces"] > 0).astype(int)
        if "2ndFlrSF" in out.columns:
            out["Has2ndFloor"] = (out["2ndFlrSF"] > 0).astype(int)

        # --- Score qualità composito ---
        if {"OverallQual", "OverallCond"}.issubset(out.columns):
            out["QualityScore"] = out["OverallQual"] * out["OverallCond"]

        # --- Rapporti area/stanze ---
        if {"GrLivArea", "TotRmsAbvGrd"}.issubset(out.columns):
            # +1 per evitare divisione per zero.
            out["AreaPerRoom"] = out["GrLivArea"] / (out["TotRmsAbvGrd"] + 1)

        if self.drop_originals:
            originals_to_drop = [
                "1stFlrSF", "2ndFlrSF", "TotalBsmtSF",
                "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath",
                "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF",
            ]
            out = out.drop(columns=[c for c in originals_to_drop if c in out.columns])

        return out

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        # Necessario per sklearn>=1.0 quando il transformer è usato in
        # ColumnTransformer/Pipeline e si vogliono i nomi di output.
        if input_features is None:
            return np.array([], dtype=object)
        return np.array(list(input_features), dtype=object)


__all__ = ["AmesFeatureEngineer"]
