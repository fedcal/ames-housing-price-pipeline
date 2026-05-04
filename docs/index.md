---
title: Ames Housing — Pipeline ML End-to-End
description: >-
  Pipeline ML production-friendly per la previsione dei prezzi immobiliari
  sul dataset Ames Housing (De Cock 2011). Preprocessing modulare, 3 modelli
  candidati (Ridge / RandomForest / XGBoost), tuning K-fold, predict_price().
---

<div class="hero-banner">
  <h1>Ames Housing — Pipeline ML End-to-End</h1>
  <p>
    Dal CSV grezzo a <code>predict_price()</code>: wrangling, feature
    engineering, encoder ordinali/nominali, 3 modelli candidati, tuning K-fold,
    valutazione su holdout. Riproducibile, modulare, GitHub Pages-ready.
  </p>
</div>

## Repository GitHub

> **Nome del repository pubblico**: `ames-housing-price-pipeline`
> URL: <https://github.com/fedcal/ames-housing-price-pipeline>

Il deploy della documentazione (questo sito) avviene automaticamente a ogni push su `main` tramite il workflow [`.github/workflows/docs.yml`](https://github.com/fedcal/ames-housing-price-pipeline/blob/main/.github/workflows/docs.yml).

## Quick start

```bash
git clone https://github.com/fedcal/ames-housing-price-pipeline.git
cd ames-housing-price-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

ames-train              # full tuning ~5-10 min
ames-train --quick      # smoke test ~30s
```

Inferenza:

```python
from ames_pipeline.inference import predict_price

casa = {"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr",
        "YearBuilt": 2005, "TotalBsmtSF": 900, "GarageArea": 480}
print(f"${predict_price(casa):,.0f}")
```

## Risultati di riferimento (full tuning)

| Modello | RMSE ($) | MAE ($) | R² | RMSE-log | MAPE |
|---|---|---|---|---|---|
| **XGBoost** | $18,350 | $11,939 | 0.9471 | 0.1055 | 7.19% |
| **Ridge** | $18,509 | $12,470 | 0.9461 | 0.1075 | 7.59% |
| RandomForest | $20,825 | $13,265 | 0.9318 | 0.1144 | 7.90% |

Lettura: XGBoost vince per ~1% sul holdout; in CV Ridge è marginalmente migliore. Differenze nel rumore — su Ames *il preprocessing accurato pesa più della scelta del modello*.

## Mappa della documentazione

- **[Teoria](teoria/01_regressione_e_target_log.md)** — regressione, regolarizzazione, tree-based models, metriche, prevenzione del leakage.
- **[Scelte tecniche](scelte_tecniche/architettura.md)** — architettura, decisioni di modellazione, trade-off espliciti.

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di DataMasters/Skiller.

Per altri progetti e contatti: [federicocalo.dev](https://federicocalo.dev).
