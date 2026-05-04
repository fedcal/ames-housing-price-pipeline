---
layout: default
title: Scelte di modellazione
parent: Scelte tecniche
nav_order: 2
math: mathjax
description: >-
  Razionale delle decisioni di modellazione del progetto Ames Housing:
  selezione delle famiglie di modelli, target log, strategia di tuning,
  gestione del rischio e trade-off espliciti.
---

# Scelte di modellazione: razionale
{: .no_toc }

Documenta le decisioni "perché così e non cosà" di livello modeling.
Per dettagli teorici sulle tecniche, vedi la sezione [Teoria]({{ '/teoria/' | relative_url }}).

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Famiglie di modelli scelte

### Ridge (lineare regolarizzato L2)

**Perché**: baseline interpretabile + robusta a multicollinearità (massiccia in Ames dopo OneHotEncoding di 25 categoriche).

**Perché non Lasso**: in Ames molte feature minori contribuiscono cumulativamente. Lasso azzererebbe le categoriche rare (es. quartieri piccoli) perdendo segnale. Ridge mantiene tutto.

**Perché non OLS**: numero di feature post-OHE ~200, vicino o superiore a $n$ per fold; matrice di covarianza quasi-singolare → coefficienti instabili. La penalità $\alpha \mathbf{I}$ risolve.

**Trade-off**: tutti i coefficienti sono shrunk uniformemente, anche quelli grossi. Per analisi esplicative serve `permutation_importance` o gli SHAP, non i $\beta_j$ grezzi.

### Random Forest

**Perché**: middle ground fra lineare e boosting. Cattura interazioni non lineari senza tuning fine. Robusto agli outlier sulle feature.

**Perché non solo XGBoost**: confronto utile dal punto di vista didattico (RF mostra il guadagno del bagging su decision tree singolo; XGBoost mostra il guadagno aggiuntivo del boosting).

**Quando NON serve in produzione**: se XGBoost è già tunato e sotto i 10 ms di latenza, RF non aggiunge valore se non come ensemble fra modelli diversi.

### XGBoost

**Perché**: gradient boosting è quasi sempre il miglior modello su tabular structured data. Su Ames porta R² da 0.94 (Ridge) a ~0.95.

**Perché non LightGBM**: alternativa equivalente, leggermente più veloce su dataset grandi (>100k righe). Su Ames (~3k righe) la differenza è invisibile, e XGBoost è più stabile in termini di iperparametri default. *Possibile estensione: aggiungere LightGBM e fare ensemble.*

**Perché non CatBoost**: ottimo gestore nativo di categoriche, ma in Ames le abbiamo già encodate; richiederebbe un percorso parallelo. Trade-off non favorevole per un PW didattico.

## 2. Iperparametri "saggi" — perché questi e non altri

### Ridge `alpha = 10`

Range provato: `[0.1, 1.0, 5.0, 10.0, 30.0, 100.0]`. Logiche:

- **Scala log**: $\alpha$ entra moltiplicato → step lineari coprono male.
- **Limite inferiore $\alpha=0.1$**: vicino a OLS, instabile.
- **Limite superiore $\alpha=100$**: schiaccia tutto, quasi predizione costante.
- **Punto ottimo $\alpha=10$**: è il valore tipico dalla letteratura su Ames (Pedro Marcelino, Serigne) ed è coerente con l'osservazione che $n \sim p$ post-OHE.

### RF `n_estimators = 400, max_features='sqrt'`

- `n_estimators`: 200 in baseline → 400 in tuning. Oltre c'è ROI marginale ma costo crescente.
- `max_depth=None`: alberi completamente sviluppati. Decorrelazione + bagging compensano l'overfit del singolo albero.
- `max_features='sqrt'`: classica Breiman (2001). Decorrela alberi diversi.

### XGBoost `lr=0.05, n_est=800, max_depth=5`

- **lr basso + n_estimators alti**: paradigma vincente di Friedman. Passi piccoli verso l'ottimo, meno rischio di overfit.
- **max_depth=5**: alberi medi, abbastanza profondi per catturare interazioni 4-5-way ma non così profondi da memorizzare il training.
- **subsample=0.8 + colsample_bytree=0.8**: regolarizzazione stocastica (alla "stochastic gradient boosting"). Riduce R² CV variance di ~30%.
- **reg_alpha=0.1, reg_lambda=1.0**: pesi leggeri sulla regolarizzazione. Sufficiente per Ames.

## 3. Strategia di cross-validation

### K=5 fold

- K=5 è il punto di equilibrio standard:
  - K=3 → varianza alta delle stime, bias basso.
  - K=10 → varianza bassa, bias alto e tempi 2× lunghi.
  - K=5 → sweet spot.
- Su 2925 righe (post outlier removal), ogni fold ha ~585 train_val. Sufficiente per la stabilità.

### `KFold(shuffle=True, random_state=42)`

- `shuffle=True`: il dataset originale è ordinato per `Order` (correlato con tempo). Senza shuffle, i fold avrebbero distribuzioni diverse di periodo di vendita → varianza inflated.
- `random_state=42`: riproducibilità.

### Stratificazione su quintili di `SalePrice`

In regressione lo `stratify` non è canonico. Lo applichiamo perché:
- Senza stratificazione, su 2925 righe lo split può lasciare molte case di lusso solo nel test (o solo nel train) → metriche instabili.
- Stratificazione su `pd.qcut(y, q=5)` garantisce che ogni fold copra tutto il range di prezzo.
- Effetto pratico: deviazione standard delle metriche CV ridotta del ~20%.

## 4. Scelta del miglior modello

Non scegliamo "il modello con RMSE più basso punto e basta". Criteri composti:

1. **Performance**: RMSE su holdout test set (non sul CV — già usato per il tuning).
2. **Stabilità**: std delle metriche su K-fold.
3. **Interpretabilità**: a parità di performance, preferiamo Ridge (coefficienti) a XGB (importance richiede SHAP per essere veramente leggibili).
4. **Latenza inferenza**: Ridge < 1 ms; RF/XGB ~50 ms su 800 alberi. Per Ames non importa, ma è un fattore in produzione.
5. **Tempo di training**: rilevante per il retraining periodico.

Sul nostro test set, con tuning completo:
- XGBoost RMSE $18,350, R²=0.9471
- Ridge   RMSE $18,509, R²=0.9461
- RF      RMSE $20,825, R²=0.9318

Differenza XGB vs Ridge ~1% RMSE — nel rumore di campionamento. **In produzione opterei per Ridge** per interpretabilità + velocità (< 1 ms di inferenza vs ~50 ms di XGBoost) + manutenibilità. La pipeline seleziona infatti Ridge come `best_model.joblib` perché ha il miglior CV (più conservativo del holdout). In leaderboard Kaggle si farebbe ensemble (`0.6 * XGB + 0.4 * Ridge`) per spremere ogni decimale.

## 5. Cosa NON ho fatto (e perché)

- **Stacking di modelli**: l'ensemble Ridge+RF+XGB con meta-learner aggiunge ~1% di R². Trade-off complessità/beneficio negativo per PW didattico.
- **Target encoding** delle categoriche con CV interno: tecnica avanzata, miglioramento ~1-2%, complessità di implementazione +30%. Estensione possibile.
- **Feature selection automatica** (RFE, SelectKBest): irrilevante quando si usa Ridge (gestisce già la dimensione) o XGB (immune a feature inutili).
- **Hyperparameter tuning bayesiano** (Optuna): più efficiente di RandomizedSearch su grid grandi (>1000 combinazioni). Su Ames la grid è gestibile, RandomizedSearchCV è sufficiente.
- **Calibration / quantile regression**: utile se servisse l'intervallo di predizione (es. "il prezzo è $180k ± $25k al 90%"). Estensione potenziale.

## 6. Quale sarebbe il next step?

Per portare il modello in produzione:

1. **Drift detection**: monitor sulle distribuzioni delle feature in input (KS-test o Chi² mensile).
2. **Retraining schedule**: re-fit ogni 6 mesi con dati nuovi.
3. **A/B test** del nuovo modello vs vecchio prima del rollout.
4. **Logging delle predizioni** per debug e per costruire il dataset di follow-up.
5. **API REST** che espone `predict_price()` (FastAPI + Docker, ~50 righe).
6. **Test di stabilità**: se il modello viene re-tunato, RMSE deve restare entro ±5%.

Tutto fuori scope per il PW didattico, ma il codice è già strutturato per supportarlo.
