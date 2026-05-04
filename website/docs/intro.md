---
sidebar_position: 1
title: Introduzione
description: |
  Pipeline ML end-to-end per la previsione dei prezzi immobiliari sul dataset Ames Housing. Architettura modulare con tre modelli candidati e tuning K-fold.
slug: /intro
---

# Ames Housing — Pipeline ML End-to-End

Pipeline didattica e **production-friendly** che, a partire dal dataset originale di **De Cock (2011)**, costruisce un sistema completo di stima del prezzo di vendita: data wrangling → feature engineering → preprocessing → tre modelli candidati (Ridge, Random Forest, XGBoost) → tuning K-fold → valutazione su holdout → inferenza tramite `predict_price()`.

:::tip In una riga
*Da CSV grezzo a `predict_price()` con pochi comandi e zero leakage.*
:::

## Repository GitHub

| Item | Link |
|---|---|
| Repo | [`fedcal/ames-housing-price-pipeline`](https://github.com/fedcal/ames-housing-price-pipeline) |
| Documentazione | [https://fedcal.github.io/ames-housing-price-pipeline/](https://fedcal.github.io/ames-housing-price-pipeline/) |
| Licenza | MIT |
| Stack docs | Docusaurus 3 + TypeScript + KaTeX |

## Risultati di riferimento (full tuning)

| Modello | RMSE ($) | MAE ($) | R² | RMSE-log | MAPE |
|:--|--:|--:|--:|--:|--:|
| **XGBoost** | $18,350 | $11,939 | **0.9471** | 0.1055 | **7.19%** |
| **Ridge** | $18,509 | $12,470 | 0.9461 | 0.1075 | 7.59% |
| RandomForest | $20,825 | $13,265 | 0.9318 | 0.1144 | 7.90% |

XGBoost vince sul holdout per ~1%; in cross-validation Ridge è marginalmente migliore. La differenza è nel rumore: su Ames *il preprocessing accurato pesa più della scelta del modello*.

## Quick start

```bash
git clone https://github.com/fedcal/ames-housing-price-pipeline.git
cd ames-housing-price-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -e ".[notebooks]"

ames-train              # full tuning ~5-10 min
ames-train --quick      # smoke test ~30s
```

Inferenza programmatica:

```python
from ames_pipeline.inference import predict_price

casa = {
    "OverallQual": 7,
    "GrLivArea": 1800,
    "Neighborhood": "CollgCr",
    "YearBuilt": 2005,
    "TotalBsmtSF": 900,
    "GarageArea": 480,
}
print(f"${predict_price(casa):,.0f}")
# ~$248,000
```

## Mappa della documentazione

### [Teoria](/docs/category/teoria)

Cinque articoli che costruiscono progressivamente i concetti necessari a leggere il progetto:

1. [Regressione & target log](./teoria/01-regressione-target-log.md) — perché OLS, perché log-trasformare il prezzo.
2. [Regolarizzazione](./teoria/02-regolarizzazione.md) — Ridge, Lasso, ElasticNet.
3. [Random Forest & Boosting](./teoria/03-random-forest-boosting.md) — perché i tree-based dominano su tabular.
4. [Metriche di regressione](./teoria/04-metriche-regressione.md) — RMSE, MAE, MAPE, R².
5. [Pipeline & data leakage](./teoria/05-pipeline-data-leakage.md) — `Pipeline` sklearn, prevenzione leakage.

### [Scelte tecniche](/docs/category/scelte-tecniche)

Decisioni architetturali e razionali documentati:

- [Architettura](./scelte-tecniche/architettura.md) — moduli, flusso dati, CLI.
- [Scelte di modellazione](./scelte-tecniche/scelte-modello.md) — trade-off espliciti.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.12 |
| ML | scikit-learn, xgboost |
| Data | pandas, numpy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, jupytext |
| Persistenza | joblib |
| Documentazione | Docusaurus 3 + TypeScript + KaTeX |
| CI/CD | GitHub Actions + GitHub Pages |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso *Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti: [**federicocalo.dev**](https://federicocalo.dev).
