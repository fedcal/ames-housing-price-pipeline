"""Pipeline ML end-to-end per la previsione del prezzo di vendita
del dataset Ames Housing (De Cock 2011).

API pubbliche principali:

    from ames_pipeline.pipeline import run_full_pipeline
    from ames_pipeline.inference import predict_price
    from ames_pipeline.data import load_raw

Per il dettaglio del flusso e delle scelte tecniche, vedi
`docs/scelte_tecniche/architettura.md` e i notebook in `notebooks/`.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
