---
sidebar_position: 1
title: Architettura del progetto
description: |
  Layout del progetto Ames Housing Pipeline: struttura dei moduli src/, flusso dati end-to-end, CLI, dipendenze.
---

## 1. Layout

```
ames-housing-price-pipeline/
├── README.md                    Quick-start, risultati, badge.
├── LICENSE                      MIT.
├── pyproject.toml               Build config + dipendenze.
├── requirements.txt             Lock approssimativo per chi non usa pyproject.
│
├── src/ames_pipeline/           Codice sorgente (installabile via pip install -e .)
│   ├── __init__.py
│   ├── config.py                Path, costanti, iperparametri di default.
│   ├── data.py                  Download + load + split.
│   ├── wrangling.py             NaN strutturali + outlier.
│   ├── features.py              AmesFeatureEngineer (sklearn transformer).
│   ├── preprocessing.py         build_preprocessor → ColumnTransformer.
│   ├── models.py                3 pipeline candidate (Ridge, RF, XGB).
│   ├── tuning.py                Grid/Randomized search con TransformedTargetRegressor.
│   ├── evaluation.py            Metriche, plot, feature importance.
│   ├── inference.py             predict_price() per inferenza singola/batch.
│   └── pipeline.py              Orchestratore end-to-end + CLI.
│
├── notebooks/                   Documentazione esecutiva e didattica.
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_features.ipynb
│   ├── 03_modeling_tuning.ipynb
│   └── 04_evaluation_inference.ipynb
│
├── data/                        Tutti i dataset (gitignored eccetto .gitkeep)
│   ├── raw/                     File originali da JSE (AmesHousing.txt, doc).
│   ├── processed/               Output preprocessing (uso interno).
│   └── external/                Eventuali dataset arricchimento.
│
├── reports/
│   ├── figures/                 PNG dei plot (predizioni, residui, importance).
│   ├── models/                  Modelli serializzati (.joblib) — gitignored.
│   ├── cv_summary.csv           Tabella CV scores.
│   └── holdout_metrics.json     Metriche holdout JSON.
│
├── scripts/
│   ├── build_notebooks.py       Genera i 4 notebook da sorgente Python.
│   └── run_full.sh              Pipeline completa (training + notebook).
│
├── tests/                       Smoke test della pipeline.
│
├── docs/
│   ├── teoria/                  5 file Markdown didattici.
│   └── scelte_tecniche/         Documenti di design (questo file).
│
└── venv/                        Virtual environment (gitignored).
```

## 2. Principi di design

### 2.1 Separazione `src/` vs `notebooks/`

Il **codice riutilizzabile** vive in `src/ames_pipeline/`. I **notebook** sono solo presentazione: contengono solo importazioni, chiamate alle funzioni di `src/`, e narrazione didattica.

Vantaggi:
- Modifiche al codice riflesse automaticamente nei notebook (basta riavviare il kernel).
- Niente duplicazione: la logica esiste una sola volta.
- Test unitari possibili sul codice in `src/` (i notebook non sono test-friendly).

### 2.2 Notebook generati da script

I 4 notebook sono prodotti da `scripts/build_notebooks.py`. Per modificare un notebook si edita lo script e si rilancia. Vantaggi:

- **Diff Git puliti**: niente cambi spuri di metadata, output cells, kernel hash.
- **Riproducibilità**: chiunque può rigenerare i notebook identici.
- **Sorgente in formato testuale**: il `.py` è searchable/grep-abile come ogni Python file.

### 2.3 Pipeline come oggetto sklearn

Tutte le trasformazioni che dipendono da statistiche del dataset sono dentro un `sklearn.pipeline.Pipeline`. Vantaggi:

- **No leakage** (vedi `docs/teoria/05_pipeline_e_data_leakage.md`).
- **Persistenza**: un singolo `joblib.dump(pipeline)` salva preprocessing + modello.
- **API uniforme**: `fit/predict/score` standard sklearn → integrabile con altri tool (ONNX, MLflow).

### 2.4 Configurazione centralizzata

Tutti i path, le costanti, le grid di iperparametri vivono in `config.py`. Niente magic number sparsi nel codice.

`PipelineConfig` è un `@dataclass(frozen=True)`: ogni esperimento crea un proprio config immutabile, nessuna mutazione accidentale.

## 3. Punti di estensione

### 3.1 Aggiungere un nuovo modello

1. Implementare la pipeline in `models.py` (es. `lightgbm_pipeline`).
2. Aggiungerla al dizionario in `get_all_pipelines`.
3. Definire la grid in `config.py` (`LIGHTGBM_PARAM_GRID`).
4. Aggiungere il caso in `tuning.tune_all_models`.

Il resto del codice (evaluation, inference) funziona senza modifiche grazie al polimorfismo sklearn.

### 3.2 Aggiungere una feature derivata

Modificare `AmesFeatureEngineer.transform` in `features.py`. La feature finisce automaticamente nelle pipeline candidate al successivo training, senza altre modifiche (le inferenze sui gruppi numeric/ordinal/nominal sono dinamiche).

### 3.3 Cambiare il dataset di download

In `config.py`:
- `DATASET_URL`: nuovo URL di download.
- `DATASET_SHA256`: hash atteso del nuovo file.
- `DATASET_FILENAME`: nome del file in `data/raw/`.

`data.py` fa già la validazione SHA-256 automaticamente.

### 3.4 Sostituire scoring/metric

Modificare in `tuning.py` la stringa di `scoring=` (sklearn ne supporta 30+). Per metriche custom, definire un `make_scorer`.

## 4. Trade-off espliciti

| Scelta | Pro | Contro |
|---|---|---|
| `Python 3.13` come target | Type hints moderni, performance | Alcune librerie ML potrebbero non avere wheel pronti |
| `xgboost` come dipendenza | State-of-the-art tabular | ~50MB di binari, dipendenza C++ |
| `sklearn 1.6+` | Output naming uniforme post-FE | Esclude utenti su versioni vecchie |
| Notebook generati da script | Diff Git puliti, riproducibilità | Editing meno comodo (no live cells) |
| Trasformazione log target | Migliora R² ~3 pt | Predizione richiede expm1 esplicito |
| Stratified split su quintili | Metriche più stabili | "Stratified regression" non è standard |
| `min_frequency=2` su OneHot | Riduce dimensionalità | Categorie rarissime perse |
| `RandomForest` con `max_features='sqrt'` | Decorrelazione classica Breiman | Sub-ottimale su segnale lineare |

## 5. Testing strategy

Il progetto non implementa una test suite completa — è un PW didattico, non production. Smoke test inclusi:

- `ames-train --quick` esegue tutta la pipeline in &lt;2 min con grid ridotte.
- Esecuzione dei 4 notebook end-to-end via `nbconvert --execute`.
- Test di `predict_price()` su 4 input (base, lusso, economica, parziale).

Per produzione si dovrebbero aggiungere:
- Unit test su `wrangling.fill_structural_missing` (proprietà: tutti i NaN strutturali → 0/'None').
- Property-based test su `AmesFeatureEngineer` (proprietà: `TotalSF >= 0` sempre).
- Integration test sull'intera pipeline (dato un seed, RMSE deve restare entro tolleranza).
- Test di drift detection sui dati nuovi.
