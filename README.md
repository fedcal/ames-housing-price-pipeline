# Ames Housing — Pipeline ML End-to-End per la Previsione dei Prezzi Immobiliari

> Pipeline didattica e production-friendly che, a partire dal dataset originale di **De Cock (2011)**, costruisce un sistema completo di stima del prezzo di vendita: data wrangling → feature engineering → preprocessing → tre modelli candidati (Ridge, Random Forest, XGBoost) → tuning K-fold → valutazione su holdout → inferenza tramite `predict_price()`.

[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/xgboost-2.1%2B-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-Just%20the%20Docs-7253ed.svg)](https://fedcal.github.io/ames-housing-price-pipeline/)

## Repository GitHub

**Nome del repository pubblico**: `ames-housing-price-pipeline`
URL: <https://github.com/fedcal/ames-housing-price-pipeline>

La documentazione (sito statico mobile-first, SEO-ready, con sidebar gerarchica e search) è scritta in Markdown nella cartella [`docs/`](docs/) e viene servita automaticamente da **GitHub Pages** tramite il tema Jekyll [Just the Docs](https://just-the-docs.com/). Setup richiesto una volta sola: *Settings → Pages → Source = **Deploy from a branch**, Branch = `main` / `/docs`*. Ogni push su `main` aggiorna il sito.

---

## Indice

- [Contesto del progetto](#contesto-del-progetto)
- [Risultati](#risultati)
- [Quick start](#quick-start)
- [Struttura del repository](#struttura-del-repository)
- [Documentazione didattica](#documentazione-didattica)
- [API di inferenza](#api-di-inferenza)
- [Riproducibilità](#riproducibilità)
- [Roadmap](#roadmap)
- [Riferimenti](#riferimenti)

---

## Contesto del progetto

Il dataset **Ames Housing** raccoglie 2930 vendite immobiliari avvenute ad Ames, Iowa, fra il 2006 e il 2010. Per ogni casa sono disponibili 80+ attributi: superficie, qualità materiali, anno di costruzione, quartiere, presenza di garage/piscina/scantinato, e così via. È il dataset di riferimento per insegnare regressione end-to-end: ricco abbastanza da essere realistico, piccolo abbastanza da essere processabile su un laptop.

Il **Project Work** chiede di:

1. caricare e validare i dati,
2. gestire valori mancanti e outlier,
3. progettare una pipeline di preprocessing modulare e riproducibile,
4. confrontare almeno tre modelli di regressione con tuning iperparametrico,
5. valutare le performance e fornire una funzione di inferenza `predict_price(input)`.

Tutti i requisiti sono soddisfatti — vedi [`docs/scelte_tecniche/architettura.md`](docs/scelte_tecniche/architettura.md) per la mappatura specifica.

---

## Risultati

Holdout test set (586 osservazioni, 20% stratificato per quintili di prezzo). Tuning K=5 fold su training set:

| Modello       | RMSE ($) | MAE ($) | R²     | RMSE-log | MAPE   |
|---------------|----------|---------|--------|----------|--------|
| **XGBoost**   | $18,350  | $11,939 | 0.9471 | 0.1055   | 7.19%  |
| **Ridge**     | $18,509  | $12,470 | 0.9461 | 0.1075   | 7.59%  |
| RandomForest  | $20,825  | $13,265 | 0.9318 | 0.1144   | 7.90%  |

Lettura: **XGBoost vince per 1% di RMSE su Ridge** (test holdout). In cross-validation la classifica si inverte (Ridge vince per pochi dollari) — segnale che le due famiglie sono sostanzialmente equivalenti su questo dataset. La pipeline seleziona automaticamente il modello con il miglior CV come `best_model.joblib`. La differenza fra XGBoost e RandomForest è invece ~13% di RMSE — il bagging puro non aggiunge valore senza il boosting sequenziale.

> **Lezione didattica**: su Ames Housing, la qualità del preprocessing (encoding ordinale corretto, log-target, gestione semantica dei NaN) pesa più della scelta del modello.

Plot diagnostici disponibili in [`reports/figures/`](reports/figures/) dopo il primo run.

---

## Quick start

### 1. Setup

```bash
git clone https://github.com/fedcal/ames-housing-price-pipeline.git
cd ames-housing-price-pipeline

python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[notebooks]"       # installa il pacchetto + dipendenze notebook
```

### 2. Pipeline completa (training + valutazione)

```bash
ames-train              # full tuning, ~5-10 min su laptop senza GPU
ames-train --quick      # smoke-test, ~30 secondi
```

Output:

- `reports/models/best_model.joblib` — pipeline serializzata pronta per l'inferenza
- `reports/cv_summary.csv` — risultati K-fold per ogni modello
- `reports/holdout_metrics.json` — metriche test set per ogni modello
- `reports/figures/*.png` — predizioni vs reali, residui

### 3. Inferenza

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
print(f"Prezzo stimato: ${predict_price(casa):,.0f}")
# Prezzo stimato: $202,500
```

L'API accetta dizionari parziali (le colonne mancanti vengono imputate) o `pd.DataFrame` per batch prediction.

### 4. Notebook didattici

```bash
jupyter lab notebooks/
```

I 4 notebook sono pensati per essere letti in sequenza:

1. **`01_eda.ipynb`** — esplorazione dataset, distribuzioni, missing analysis, outlier.
2. **`02_preprocessing_features.ipynb`** — pipeline sklearn, feature engineering.
3. **`03_modeling_tuning.ipynb`** — confronto modelli, K-fold CV, GridSearch/RandomizedSearch.
4. **`04_evaluation_inference.ipynb`** — metriche, diagnostica, predict_price.

I notebook sono **rigenerabili** da `scripts/build_notebooks.py` (sorgente di verità in Python — diff Git puliti).

---

## Struttura del repository

```
src/ames_pipeline/        Libreria Python installabile
├── config.py               Path, costanti, iperparametri
├── data.py                 Download + load + split
├── wrangling.py            Missing values + outlier
├── features.py             AmesFeatureEngineer (sklearn transformer)
├── preprocessing.py        ColumnTransformer (numeric/ordinal/nominal)
├── models.py               3 pipeline candidate
├── tuning.py               Grid/Randomized search + log-target wrapper
├── evaluation.py           Metriche + plot
├── inference.py            predict_price()
└── pipeline.py             Orchestrator + CLI

notebooks/                Documentazione esecutiva
docs/
├── teoria/                 5 spiegazioni didattiche dei concetti ML
└── scelte_tecniche/        Architettura, decisioni di modello
data/                     Dataset (gitignored)
reports/                  Output (figures, metrics, models — gitignored)
```

Dettaglio completo: [`docs/scelte_tecniche/architettura.md`](docs/scelte_tecniche/architettura.md).

---

## Documentazione didattica

I 5 file in [`docs/teoria/`](docs/teoria/) coprono i concetti di base necessari per capire il progetto:

| File | Contenuto |
|---|---|
| [`01_regressione_e_target_log.md`](docs/teoria/01_regressione_e_target_log.md) | Regressione lineare; perché trasformare il target con log1p; `TransformedTargetRegressor`. |
| [`02_regolarizzazione.md`](docs/teoria/02_regolarizzazione.md) | Ridge vs Lasso vs ElasticNet; ruolo dello scaling; tuning di α. |
| [`03_random_forest_e_boosting.md`](docs/teoria/03_random_forest_e_boosting.md) | Decision tree, bagging (RF), gradient boosting (XGBoost); iperparametri chiave. |
| [`04_metriche_regressione.md`](docs/teoria/04_metriche_regressione.md) | RMSE, MAE, R², RMSE-log, MAPE; quando usare quale; diagnostica residui. |
| [`05_pipeline_e_data_leakage.md`](docs/teoria/05_pipeline_e_data_leakage.md) | Anatomia del data leakage; come prevenirlo con `sklearn.pipeline`; sanity check. |

Per le decisioni progettuali (perché Ridge e non Lasso, perché K=5, perché RandomizedSearch su XGB), vedi [`docs/scelte_tecniche/`](docs/scelte_tecniche/).

---

## API di inferenza

```python
from ames_pipeline.inference import predict_price, example_input

# Casa "media": qualità 6, GrLivArea 1600, NAmes
base = example_input()
predict_price(base)             # → ~$165,000

# Casa di lusso
luxury = {**base, "OverallQual": 9, "GrLivArea": 2400, "Neighborhood": "NridgHt"}
predict_price(luxury)           # → ~$280,000

# Input parziale (4 feature)
minimal = {"OverallQual": 7, "GrLivArea": 1800, "Neighborhood": "CollgCr", "YearBuilt": 2005}
predict_price(minimal)          # → ~$200,000  [le altre feature sono imputate]

# Batch
import pandas as pd
df = pd.DataFrame([base, luxury, minimal])
predict_price(df)               # → [165000, 280000, 200000]
```

---

## Riproducibilità

Tre garanzie:

1. **Stesso dataset**: SHA-256 del file `AmesHousing.txt` validato in `data.py`. Se il mirror cambia byte, fallisce esplicitamente.
2. **Stesse versioni**: `pyproject.toml` e `requirements.txt` pinnati a range minor (es. `numpy>=2.0,<3.0`).
3. **Stesso seed**: `RANDOM_STATE=42` propagato a tutti gli step stocastici.

Risultato: `ames-train` produce metriche bit-identiche su esecuzioni successive (entro tolleranza float64 dovuta al thread parallelism `n_jobs=-1`; per riproducibilità bit-perfect impostare `n_jobs=1`).

---

## Roadmap

In ordine di valore aggiunto:

- [ ] **Ensemble stacking** Ridge + XGB con meta-learner.
- [ ] **Optuna** in alternativa a `RandomizedSearchCV` (TPE + pruning).
- [ ] **API REST** con FastAPI + Docker.
- [ ] **Split temporale** (train 2006-08, test 2009-10) per simulare deployment reale.
- [ ] **Calibration** via quantile regression per intervalli di predizione.
- [ ] **Drift detection** sulle feature in input (KS-test mensile).

---

## Riferimenti

### Dataset

- **De Cock, D.** (2011), *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*, [Journal of Statistics Education 19(3)](https://jse.amstat.org/v19n3/decock.pdf).
- Documentazione delle 82 variabili: [`data/raw/DataDocumentation.txt`](https://jse.amstat.org/v19n3/decock/DataDocumentation.txt) (scaricato runtime).

### Metodologia

- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*, Springer, cap. 3, 10, 15.
- **Chen, T. & Guestrin, C.** (2016), *XGBoost: A Scalable Tree Boosting System*, KDD.
- **Breiman, L.** (2001), *Random Forests*, Machine Learning 45(1).

### Stack tecnico

- [scikit-learn](https://scikit-learn.org/) — pipeline, encoder, CV.
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting.
- [pandas](https://pandas.pydata.org/) — manipolazione dati.
- [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) — plotting.

---

## Autore & licenza

**Creazione di Federico Calò** — Project Work del percorso *Machine Learning Engineer* (DataMasters/Skiller, 2026).

Per altri progetti, contatti e portfolio: <https://federicocalo.dev>.

[MIT License](LICENSE) © 2026 Federico Calò.

Il dataset Ames Housing è di pubblico dominio (Dean De Cock, 2011).
