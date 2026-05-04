---
layout: home
title: Home
nav_order: 1
description: >-
  Pipeline ML production-friendly per la previsione dei prezzi immobiliari sul
  dataset Ames Housing (De Cock 2011). Preprocessing modulare, 3 modelli
  candidati (Ridge, Random Forest, XGBoost), tuning K-fold, predict_price().
permalink: /
---

<div class="hero-banner" markdown="0">
  <h1>Ames Housing &mdash; Pipeline ML End&#8209;to&#8209;End</h1>
  <p>
    Dal CSV grezzo a <code>predict_price()</code>: wrangling, feature engineering,
    encoder ordinali e nominali, 3 modelli candidati, tuning K&#8209;fold,
    valutazione su holdout. Riproducibile, modulare, production&#8209;friendly.
  </p>
</div>

## In sintesi

Progetto di riferimento del percorso **Machine Learning Engineer** di
[DataMasters](https://datamasters.it/)/Skiller. Implementa l'intero flusso
di lavoro di un modello tabular di regressione, dal dato grezzo all'inferenza,
con focus su **rigorosità metodologica**, **prevenzione del leakage** e
**riproducibilità**.

<div class="kpi-grid" markdown="0">
  <div class="kpi-card">
    <div class="kpi-label">Best RMSE (XGBoost)</div>
    <div class="kpi-value">$18,350</div>
    <div>holdout test set</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">R² holdout</div>
    <div class="kpi-value">0.9471</div>
    <div>varianza spiegata</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">MAPE</div>
    <div class="kpi-value">7.19%</div>
    <div>errore percentuale medio</div>
  </div>
</div>

## Repository GitHub

- **Nome del repository**: `ames-housing-price-pipeline`
- **URL**: [github.com/fedcal/ames-housing-price-pipeline](https://github.com/fedcal/ames-housing-price-pipeline)
- **Documentazione (questo sito)**: pubblicata via **GitHub Pages** dalla cartella
  [`/docs`](https://github.com/fedcal/ames-housing-price-pipeline/tree/main/docs).

{: .note }
> La documentazione viene servita direttamente dai file Markdown della cartella
> `docs/`, processati da Jekyll con il tema **Just the Docs**.
> Ogni push su `main` aggiorna automaticamente il sito.

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

## Risultati di riferimento (full tuning)

| Modello | RMSE ($) | MAE ($) | R² | RMSE&#8209;log | MAPE |
|:--|--:|--:|--:|--:|--:|
| **XGBoost** | $18,350 | $11,939 | 0.9471 | 0.1055 | 7.19% |
| **Ridge** | $18,509 | $12,470 | 0.9461 | 0.1075 | 7.59% |
| RandomForest | $20,825 | $13,265 | 0.9318 | 0.1144 | 7.90% |

{: .tip }
> XGBoost vince sull'holdout per ~1%; in cross-validation Ridge è marginalmente
> migliore. La differenza è nel rumore: su Ames *il preprocessing accurato pesa
> più della scelta del modello*.

## Mappa della documentazione

### [Teoria](teoria/)

Fondamenti per leggere i risultati del progetto:

- [Regressione & target log](teoria/01_regressione_e_target_log/) — perché OLS, perché log-trasformare il prezzo.
- [Regolarizzazione](teoria/02_regolarizzazione/) — Ridge, Lasso, ElasticNet.
- [Random Forest & Boosting](teoria/03_random_forest_e_boosting/) — perché i tree-based dominano su tabular.
- [Metriche di regressione](teoria/04_metriche_regressione/) — RMSE, MAE, MAPE, R².
- [Pipeline & data leakage](teoria/05_pipeline_e_data_leakage/) — sklearn `Pipeline`, prevenzione leakage.

### [Scelte tecniche](scelte_tecniche/)

Decisioni architetturali e di modellazione:

- [Architettura](scelte_tecniche/architettura/) — moduli, flusso dati, CLI.
- [Scelte di modellazione](scelte_tecniche/scelte_modello/) — trade-off espliciti.

## Stack tecnologico

| Layer | Tecnologie |
|:--|:--|
| Linguaggio | Python 3.12 |
| ML | scikit-learn, xgboost |
| Data | pandas, numpy |
| Plotting | matplotlib, seaborn |
| Notebook | jupyter, jupytext |
| Persistenza | joblib |
| Documentazione | Jekyll + Just the Docs |

## Autore

Progetto realizzato da **Federico Calò** come parte del percorso
*Machine Learning Engineer* di [DataMasters](https://datamasters.it/)/Skiller.

Per altri progetti, articoli e contatti:
[**federicocalo.dev**](https://federicocalo.dev){: .btn .btn-purple }
