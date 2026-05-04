---
sidebar_position: 5
title: Pipeline sklearn & data leakage
description: |
  Pipeline scikit-learn: come prevengono il data leakage, K-fold corretta, anti-pattern comuni e best practice di produzione.
---

## 1. Cos'è il data leakage

Il **data leakage** è la situazione in cui informazioni del test set (o del futuro, in problemi temporali) influenzano il training. Causa metriche di validazione **artificialmente buone** che NON si confermano in produzione.

È il bug più sottile e dannoso del ML applicato. È anche il più frequente.

## 2. I tre tipi di leakage in regressione tabular

### 2.1 Leakage da preprocessing globale

**Sintomo**: imputazione, scaling, encoding fatti sull'INTERO dataset prima dello split.

```python
# WRONG: leakage!
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())  # mediana globale
df_scaled = StandardScaler().fit_transform(df.values)                      # mu/sigma globali
X_train, X_test = train_test_split(df_scaled, ...)                          # tardi
```

**Perché è leakage**: la mediana e mu/sigma sono statistiche calcolate su training+test insieme. In produzione il test set non è disponibile, quindi le statistiche sarebbero diverse → predizioni diverse → metriche di validazione non rappresentano la realtà.

**Soluzione**: usare **`sklearn.pipeline.Pipeline`** che esegue `fit` solo sul training e `transform` su test/inferenza usando le statistiche del training.

```python
# RIGHT: niente leakage
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   Ridge()),
])
pipeline.fit(X_train, y_train)   # statistiche da X_train
pipeline.predict(X_test)         # riusa quelle statistiche
```

### 2.2 Leakage da feature derivate dal target

**Sintomo**: una feature è una funzione (anche indiretta) del target.

```python
# WRONG: la feature usa SalePrice!
df['price_per_sqft'] = df['SalePrice'] / df['GrLivArea']  # leakage diretto
df['neighborhood_avg'] = df.groupby('Neighborhood')['SalePrice'].transform('mean')  # target encoding senza CV
```

In Ames, il rischio principale è il **target encoding** (sostituire una categoria con la media del target per quella categoria). Va sempre fatto **dentro la CV**, non sul dataset completo.

**Regola**: nessuna trasformazione che dipende da `y` deve essere applicata fuori dalla pipeline.

### 2.3 Leakage da split temporale ignorato

**Sintomo**: in dati con dimensione temporale (eventi datati), uno split casuale mette nel training osservazioni successive a quelle del test set.

In Ames Housing, le case sono state vendute fra 2006 e 2010. Un modello realistico per *predire prezzi futuri* dovrebbe addestrarsi sul 2006-2008 e validarsi su 2009-2010, NON splitare casualmente. Per il nostro project work questo non è richiesto (split casuale è ammesso), ma è un'ottima estensione.

## 3. La struttura della pipeline Ames

Il design "no-leakage by construction" del nostro progetto:

```
                                      ┌─────────────────────────────┐
   load_raw()                          │  Trasformazioni "semantiche"│
        │                              │  che NON usano statistiche  │
        │                              │  → si possono fare prima    │
        │ fill_structural_missing      │    dello split.             │
        │ (NaN→'None' / 0)             │  - NaN → 'None'             │
        │                              │  - rimozione outlier        │
        │ remove_grliv_area_outliers   │    (regola fissa di De Cock)│
        │                              └─────────────────────────────┘
        ▼
   train_test_split (stratify)
        │
        ▼
  ╔═══════════════════════════════════════════╗
  ║  Pipeline sklearn (fit_transform su train,║
  ║   transform su test):                     ║
  ║                                           ║
  ║   1. AmesFeatureEngineer                  ║
  ║      └─ deterministic, no statistiche     ║
  ║                                           ║
  ║   2. ColumnTransformer:                   ║
  ║      ├─ SimpleImputer(median) sul train   ║
  ║      ├─ OrdinalEncoder fitted sul train   ║
  ║      └─ OneHotEncoder fitted sul train    ║
  ║                                           ║
  ║   3. StandardScaler (solo Ridge):         ║
  ║      └─ μ, σ dal training                 ║
  ║                                           ║
  ║   4. Modello (Ridge / RF / XGB)           ║
  ╚═══════════════════════════════════════════╝
        │
        ▼
   TransformedTargetRegressor (log1p / expm1):
   wrappa tutta la pipeline. Predict ritorna $.
```

**Ogni statistica è calcolata solo sul training**. K-fold CV applica la pipeline ad ognuno dei K fold separatamente: se un fold ha statistiche diverse, va bene — è proprio questa la varianza che vogliamo misurare.

## 4. Pre-split vs in-pipeline: quando è OK fuori dalla pipeline

Posso fare una trasformazione fuori dalla pipeline solo se è **deterministica e non dipende dal sample**. Esempi:

| Operazione | Dove? | Perché |
|---|---|---|
| Conversione tipo (`str` → `int`) | fuori | deterministica |
| Rinominare colonne | fuori | deterministica |
| Mappare NaN strutturali a 'None' | fuori | regola fissa, indipendente dal sample |
| Rimozione outlier con regola fissa (`GrLivArea > 4000 e prezzo < 300k`) | fuori | regola fissa |
| Imputazione con mediana | DENTRO | mediana dipende dal sample |
| Scaling | DENTRO | mu/sigma dipendono dal sample |
| OneHotEncoder | DENTRO | il vocabolario dipende dal sample |

## 5. Cross-validation con la pipeline

`cross_val_score(pipeline, X, y, cv=5)` esegue, per ognuno dei 5 fold:

1. Split di X, y in `(X_tr, y_tr)` e `(X_va, y_va)`.
2. `pipeline.fit(X_tr, y_tr)` → tutte le statistiche calcolate sul fold di training.
3. `pipeline.predict(X_va)` → riusa quelle statistiche sul fold di validation.
4. Calcolo metrica su `(y_va, y_pred)`.

**Ogni fold è una mini-simulazione del deployment**. Le metriche CV stimano la performance attesa in produzione.

## 6. Errori che ho visto fare nei progetti

1. **Standardize tutto il dataset prima del split** → leakage dello scaling.
2. **`OneHotEncoder().fit_transform(df_completo)`** prima del split → vocabolario contaminato dal test.
3. **Tuning iperparametri sul test set** ("guardo il test, ottimizzo, riguardo, riottimizzo") → test set non più valido.
4. **Feature engineering basata su statistiche di gruppo** (es. media prezzo per quartiere) calcolata sull'intero dataset.
5. **`pd.cut(df['SalePrice'], bins=...)` per stratificare** → in regressione lo stratify deve usare `pd.qcut` su SalePrice, ma il calcolo dei quintili deve essere esplicitamente sul `y_train` e poi applicato a `y` per il `train_test_split`. Nel nostro codice è fatto correttamente.
6. **Salvare il dataframe pre-processato** e poi splittarlo → uno qualsiasi dei punti sopra può essere già successo.

## 7. Sanity check: il modello "shuffle test"

Per assicurarsi che NON ci sia leakage, un check rapido:

```python
y_shuffled = y.sample(frac=1, random_state=0).reset_index(drop=True)
score = cross_val_score(pipeline, X, y_shuffled, scoring='r2', cv=5).mean()
print(f"R² con target casualizzato: {score:.4f}")
```

Atteso: $R^2 \approx 0$ (il modello non può imparare nulla da $y$ casuale). Se ottieni $R^2 > 0.1$, c'è leakage.

## 8. Riferimenti

- **Sklearn user guide**: [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html).
- **Kapoor & Narayanan** (2023), *Leakage and the Reproducibility Crisis in ML-based Science*, Patterns 4(9).
- **Kaufman et al.** (2012), *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 6(4).
