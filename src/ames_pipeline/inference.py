"""Inferenza end-to-end su nuovi dati.

Espone `predict_price(input_dict)` come da specifica del project work.
Il modello viene caricato dal disco una sola volta tramite cache lazy.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODELS_DIR

logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"


@lru_cache(maxsize=4)
def _load_model(model_path: str) -> Any:
    """Carica un modello serializzato; cached per evitare I/O ripetuti.

    `lru_cache` accetta solo argomenti hashable: per questo riceve
    `str`, non `Path`.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modello non trovato a {path}. "
            "Esegui prima la pipeline di training: "
            "`python -m ames_pipeline.pipeline`."
        )
    logger.info("Carico modello da %s", path)
    return joblib.load(path)


def _expected_columns(model: Any) -> list[str] | None:
    """Estrae i nomi colonna attesi dal modello serializzato.

    Naviga la struttura: TransformedTargetRegressor → Pipeline →
    AmesFeatureEngineer → ColumnTransformer. Le colonne attese sono
    quelle viste dal feature_engineer al fit (cioè quelle del
    DataFrame di training PRIMA del feature engineering).

    Il `ColumnTransformer` finale richiede le colonne POST-FE; il
    feature_engineer invece può lavorare con un sottoinsieme: per
    `predict_price` ci basta replicare le colonne pre-FE.

    Returns:
        Lista dei nomi colonna se ricavabile, None altrimenti.
    """
    try:
        from sklearn.compose import TransformedTargetRegressor
        estimator = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
        # estimator è la Pipeline esterna. `feature_names_in_` è settato dal primo step durante il fit.
        return list(estimator.feature_names_in_)
    except (AttributeError, KeyError):
        return None


def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Riempie le colonne mancanti con NaN, scarta le extra, riordina.

    Strategia: per le colonne mancanti l'imputer della pipeline
    sostituirà i NaN con il valore appropriato (mediana per numeriche,
    'None' per categoriche/ordinali strutturali). Quindi `predict_price`
    può essere chiamata con un input parziale e produrre comunque una
    stima (con incertezza maggiore, ovviamente).
    """
    aligned = df.reindex(columns=expected)
    return aligned


def predict_price(
    input_dict: dict[str, Any] | pd.DataFrame,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> float | list[float]:
    """Predice il prezzo di vendita per una o più abitazioni.

    Args:
        input_dict: dizionario con le feature di una singola casa
            (chiavi = nomi colonna del dataset Ames in convenzione
            Kaggle/CamelCase: 'GrLivArea', 'OverallQual',
            'Neighborhood', 'YearBuilt', 'YearRemodAdd', ...),
            oppure un `pd.DataFrame` per batch prediction.
        model_path: path del modello serializzato (joblib).

    Returns:
        Prezzo predetto in dollari (float per input dict singolo,
        list[float] per DataFrame).

    Note pratiche:
        - Le colonne MANCANTI nell'input vengono trattate come NaN dal
          preprocessor (imputazione automatica). L'utente non deve
          fornire tutti gli 80+ attributi: solo quelli noti.
        - Le colonne EXTRA non presenti nel training vengono ignorate.
    """
    model = _load_model(str(model_path))

    if isinstance(input_dict, dict):
        df = pd.DataFrame([input_dict])
        is_single = True
    elif isinstance(input_dict, pd.DataFrame):
        df = input_dict.copy()
        is_single = False
    else:
        raise TypeError(
            f"input_dict deve essere dict o DataFrame, ricevuto {type(input_dict).__name__}."
        )

    expected = _expected_columns(model)
    if expected is not None:
        df = _align_to_expected(df, expected)

    predictions = model.predict(df)
    if is_single:
        return float(predictions[0])
    return [float(x) for x in predictions]


def example_input() -> dict[str, Any]:
    """Esempio di input minimo per smoke-test della funzione predict_price.

    Casa "media" Ames: superficie media, qualità media, quartiere medio.
    """
    return {
        "MSSubClass": 60,
        "MSZoning": "RL",
        "LotArea": 9500,
        "OverallQual": 6,
        "OverallCond": 5,
        "YearBuilt": 1995,
        "YearRemodAdd": 2000,
        "1stFlrSF": 900,
        "2ndFlrSF": 700,
        "TotalBsmtSF": 850,
        "GrLivArea": 1600,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 7,
        "Fireplaces": 1,
        "GarageArea": 480,
        "GarageCars": 2,
        "Neighborhood": "NAmes",
        "HouseStyle": "2Story",
        "ExterQual": "TA",
        "KitchenQual": "TA",
        "BsmtQual": "TA",
        "YrSold": 2010,
        "MoSold": 6,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


__all__ = ["predict_price", "example_input", "DEFAULT_MODEL_PATH"]
