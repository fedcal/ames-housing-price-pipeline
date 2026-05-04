---
sidebar_position: 1
title: Regressione & target log
description: |
  Regressione lineare OLS, skewness del target SalePrice, trasformazione logaritmica con log1p e TransformedTargetRegressor di scikit-learn.
---

# Regressione lineare e trasformazione logaritmica del target

> *"All models are wrong, but some are useful."* — George E. P. Box

## 1. Cosa è una regressione e perché è il punto di partenza

Una **regressione** è un modello che mappa un vettore di feature $\mathbf{x} \in \mathbb{R}^p$ a un valore continuo $y \in \mathbb{R}$. Per Ames Housing, $\mathbf{x}$ contiene 80+ attributi della casa (superficie, qualità, quartiere, anno costruzione, ...) e $y$ è il prezzo di vendita in dollari.

La forma più semplice è la **regressione lineare**:

$$
\hat{y}(\mathbf{x}) = \beta_0 + \sum_{j=1}^{p} \beta_j x_j
$$

Si addestra minimizzando la **somma dei quadrati dei residui** (Ordinary Least Squares):

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^{n} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2
$$

**Perché iniziare da qui anche quando sappiamo che XGBoost vincerà?**

1. **Baseline interpretabile**: ogni $\beta_j$ ha unità di misura precisa (`$/ft²` per la superficie, `$/punto-qualità`, ...). Aiuta a sensibilizzarsi sul problema.
2. **Sanity check**: se Ridge ottiene già R²=0.94 sul test, sappiamo che il segnale lineare è dominante. Un boosting che ottiene R²=0.95 non è un miracolo — è un miglioramento marginale.
3. **Detector di errori**: se la baseline lineare dà R²=0.30, qualcosa nel preprocessing è probabilmente rotto (encoding sbagliato, target non trasformato, missing non gestiti).

## 2. Il problema della skewness del target

`SalePrice` in Ames ha una distribuzione **fortemente asimmetrica a destra** (skewness ≈ 1.74):

```
mediana ≈ $160k     media ≈ $180k     max ≈ $755k
                                               ↑
                                          long tail
```

Per un OLS, la presenza di pochi prezzi molto alti ha tre conseguenze:

### 2.1 Errori dominati dalle case costose

L'RMSE è una **media quadratica** dei residui: una casa da $700k con errore di $50k contribuisce per $50,000² = 2.5×10⁹ alla loss; una casa da $100k con lo stesso errore percentuale (5%) contribuisce per $5,000² = 2.5×10⁷ — cento volte meno. Il modello tende quindi ad allinearsi sulle case costose, sotto-performando sulla maggioranza.

### 2.2 Violazione delle assunzioni dell'OLS

L'inferenza statistica classica (intervalli di confidenza, p-value) assume residui $\epsilon \sim \mathcal{N}(0, \sigma^2)$ con varianza costante (omoscedasticità). Con un target skewed, la varianza dei residui cresce con $y$ — **eteroscedasticità**.

### 2.3 Sensibilità agli outlier

Pochi prezzi estremi sbilanciano la stima dei coefficienti.

## 3. Soluzione: trasformare il target

La trasformazione che applichiamo è $y' = \log(1 + y)$, codificata in NumPy come `np.log1p`:

```python
y_log = np.log1p(y)        # training su questa scala
y_pred_orig = np.expm1(y_pred_log)  # back-transform per le metriche $
```

**Effetti**:

- La distribuzione di `log1p(SalePrice)` ha skewness ≈ 0.12 — quasi gaussiana.
- L'errore quadratico in scala log corrisponde all'**errore quadratico relativo** in scala originale, perché $\log y_1 - \log y_2 = \log(y_1 / y_2)$. Una casa da $100k con +10% di errore pesa quanto una da $700k con +10%.
- È la metrica adottata dalla competizione Kaggle Ames House Prices, quindi i nostri risultati sono direttamente confrontabili con la leaderboard.

### 3.1 Perché `log1p` e non `log`?

`log1p(0) = 0` mentre `log(0) = -∞`. Per `SalePrice` non capita mai 0, ma **`log1p` è una buona pratica difensiva** — costa nulla, evita NaN se per errore qualcuno passa un dataset diverso.

### 3.2 Come la implementiamo nella pipeline

Usiamo `sklearn.compose.TransformedTargetRegressor`:

```python
from sklearn.compose import TransformedTargetRegressor

wrapped = TransformedTargetRegressor(
    regressor=my_pipeline,
    func=np.log1p,
    inverse_func=np.expm1,
)
wrapped.fit(X_train, y_train)         # internamente fit su log1p(y_train)
y_pred = wrapped.predict(X_test)      # internamente expm1(predizione)
```

Vantaggi rispetto a trasformare il target a mano:

- **No leakage**: la trasformazione fa parte del modello → applicata identica in CV.
- **Metriche corrette**: `cross_val_score(scoring='neg_root_mean_squared_error')` calcola RMSE in $ originali, perché TTR riapplica `expm1` prima dello scoring.
- **API uniforme**: chiamiamo sempre `predict(X)` e otteniamo `$`, niente conversioni a mano nei notebook.

## 4. Quando NON trasformare il target

La trasformazione log è quasi sempre buona per prezzi/costi/popolazioni. Non lo è quando:

- Il target ha valori negativi (es. variazioni di temperatura) → `log` non definito.
- Il target è già ~simmetrico → trasformare aumenta complessità senza beneficio.
- Il dominio è naturalmente lineare (es. distanze in metri tra due punti) — la log distorce l'unità di misura.

In Ames la trasformazione è la scelta giusta: lo conferma sia l'EDA (skewness 1.74 → 0.12) che la performance (RMSE in $ migliora di ~$3k passando da target lineare a target log su Ridge).

## 5. Riferimenti per approfondire

- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*, cap. 3 (linear regression).
- **De Cock, D.** (2011), *Ames, Iowa: Alternative to the Boston Housing Data...*, JSE 19(3) — paper originale del dataset.
- **Sklearn user guide**: [`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/compose.html#transforming-target-in-regression).
