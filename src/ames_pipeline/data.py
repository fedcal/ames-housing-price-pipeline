"""Caricamento, validazione e split del dataset Ames Housing.

Single responsibility: tutto ciò che riguarda I/O dati grezzi, controlli
di integrità e creazione di train/test set vive qui. Nessuna logica di
preprocessing applicata: quella sta in `wrangling.py` / `preprocessing.py`.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Final

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import (
    DATASET_FILENAME,
    DATASET_SHA256,
    DATASET_URL,
    ID_COLUMNS,
    RAW_DIR,
    TARGET_COLUMN,
    PipelineConfig,
    DEFAULT_CONFIG,
)

logger = logging.getLogger(__name__)

EXPECTED_ROWS: Final[int] = 2930
EXPECTED_COLUMNS: Final[int] = 82


def _sha256_of(path: Path) -> str:
    """SHA-256 del file. Streaming per evitare di caricare in RAM file grandi."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(target_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Scarica il dataset Ames Housing dal mirror JSE e ne valida l'hash.

    Args:
        target_dir: cartella di destinazione (`data/raw/` di default).
        force: se True, riscarica anche se il file esiste già.

    Returns:
        Path al file `AmesHousing.txt` locale.

    Raises:
        ValueError: se l'hash non corrisponde a quello atteso.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / DATASET_FILENAME

    if target.exists() and not force:
        logger.info("Dataset già presente: %s", target)
    else:
        logger.info("Download dataset da %s", DATASET_URL)
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
        logger.info("Salvato %s (%d byte)", target, target.stat().st_size)

    actual = _sha256_of(target)
    if actual != DATASET_SHA256:
        raise ValueError(
            f"SHA256 mismatch su {target}.\n"
            f"  atteso : {DATASET_SHA256}\n"
            f"  ottenuto: {actual}\n"
            "Il file potrebbe essere corrotto o il mirror potrebbe essere cambiato."
        )
    return target


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza i nomi colonna del file originale alla convenzione Kaggle.

    Trasformazioni:
        - Rimuove gli spazi: 'MS SubClass' → 'MSSubClass'.
        - Sostituisce '/' con '': 'Year Remod/Add' → 'YearRemodAdd'.
        - Mappature manuali per casi non riducibili:
            '3Ssn Porch' → '3SsnPorch' (già coperto dal punto 1).

    Il dataset originale (De Cock 2011) usa nomi con spazi e qualche
    slash; la letteratura post-Kaggle usa CamelCase concatenato.
    Allineiamoci alla seconda convenzione per coerenza con notebook
    esterni e per non avere caratteri ambigui (`/`) nei nomi feature.
    """
    return df.rename(columns=lambda c: c.replace(" ", "").replace("/", ""))


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Carica il dataset grezzo da disco, normalizza i nomi colonna e valida le shape attese.

    Se il file non esiste, viene scaricato automaticamente.
    """
    if path is None:
        path = RAW_DIR / DATASET_FILENAME
    if not path.exists():
        download_dataset()

    df = pd.read_csv(path, sep="\t")
    df = _normalize_columns(df)

    if df.shape[0] != EXPECTED_ROWS:
        raise ValueError(
            f"Numero righe inatteso: {df.shape[0]} (attese {EXPECTED_ROWS})"
        )
    if df.shape[1] != EXPECTED_COLUMNS:
        raise ValueError(
            f"Numero colonne inatteso: {df.shape[1]} (attese {EXPECTED_COLUMNS})"
        )
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Colonna target '{TARGET_COLUMN}' assente.")

    logger.info(
        "Dataset caricato: %d righe × %d colonne. Target='%s'",
        df.shape[0], df.shape[1], TARGET_COLUMN,
    )
    return df


def split_features_target(
    df: pd.DataFrame, drop_id: bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (feature) da y (target).

    Le colonne identificatore (Order, PID) vanno SEMPRE rimosse: sono ID
    e non hanno potere predittivo, ma se lasciate possono causare
    leakage o sovradattamento (specialmente su `Order` che è correlato
    con il tempo).
    """
    columns_to_drop = [TARGET_COLUMN]
    if drop_id:
        columns_to_drop = list(columns_to_drop) + [c for c in ID_COLUMNS if c in df.columns]
    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN].astype(float)
    return X, y


def make_train_test_split(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split casuale train/test stratificato per fascia di prezzo.

    Stratifichiamo per quintile di SalePrice: in regressione la
    "stratificazione" non è standard, ma garantire che train e test
    coprano lo stesso intervallo di prezzo riduce la varianza delle
    metriche e rende i confronti più stabili.
    """
    X, y = split_features_target(df)
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    return train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y_bins,
    )


__all__ = [
    "EXPECTED_ROWS",
    "EXPECTED_COLUMNS",
    "download_dataset",
    "load_raw",
    "split_features_target",
    "make_train_test_split",
]
