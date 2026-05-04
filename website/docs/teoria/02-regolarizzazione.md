---
sidebar_position: 2
title: "Regolarizzazione: Ridge, Lasso, ElasticNet"
description: |
  Ridge L2, Lasso L1 ed ElasticNet: cosa sono, quando usarle, trade-off bias-variance e implementazione in scikit-learn.
---

## 1. Il problema dell'OLS senza regolarizzazione

Quando il numero di feature $p$ è grande rispetto al numero di osservazioni $n$, oppure quando le feature sono **collineari** (correlate fra loro), la regressione lineare ordinaria (OLS) ha due problemi pratici:

1. **Varianza esplosiva dei coefficienti**: piccole variazioni nel training set provocano grandi variazioni in $\hat{\boldsymbol{\beta}}$. Il modello è instabile.
2. **Overfitting**: con tante feature, l'OLS può adattarsi al rumore — RMSE basso sul training, alto sul test.

Per Ames Housing, dopo OneHotEncoding di tutte le categoriche, il numero di feature può salire a 200+ — vicino al numero di osservazioni nei singoli quartieri. Il rischio di overfitting è reale.

## 2. La regolarizzazione: aggiungere una penalità

Modifichiamo la loss aggiungendo una penalità che disincentiva coefficienti grandi:

$$
\mathcal{L}_{\text{reg}}(\boldsymbol{\beta}) = \underbrace{\sum_{i} (y_i - \mathbf{x}_i^\top \boldsymbol{\beta})^2}_{\text{loss OLS}} + \alpha \cdot R(\boldsymbol{\beta})
$$

Tre famiglie comuni:

| Modello       | Penalità $R(\boldsymbol{\beta})$              | Effetto                  |
|---------------|-----------------------------------------------|--------------------------|
| **Ridge**     | $\sum_j \beta_j^2$                            | Shrink uniforme          |
| **Lasso**     | $\sum_j |\beta_j|$                            | Shrink + sparsità (alcuni $\beta_j = 0$) |
| **ElasticNet**| $\rho \sum_j |\beta_j| + (1-\rho) \sum_j \beta_j^2$ | Mix dei due       |

L'iperparametro $\alpha \geq 0$ controlla la forza:
- $\alpha = 0$ → torniamo all'OLS;
- $\alpha \to \infty$ → tutti i $\beta_j \to 0$ (predizione = media).

## 3. Perché Ridge è la scelta predefinita per Ames

In Ames usiamo **Ridge** ($L_2$). Tre motivi:

### 3.1 Multicollinearità intrinseca

Molte feature sono correlate per costruzione:
- `1stFlrSF + 2ndFlrSF + TotalBsmtSF ≈ TotalSF` (somma esatta dopo feature engineering),
- `OverallQual` correlato con `ExterQual`, `KitchenQual`, ...,
- `YearBuilt` correlato con `GarageYrBlt` per la maggior parte delle case.

L'OLS in presenza di multicollinearità ha **matrice $\mathbf{X}^\top \mathbf{X}$ quasi singolare** → invertirla amplifica il rumore. Ridge risolve aggiungendo $\alpha \mathbf{I}$ alla diagonale:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
$$

La matrice diventa sicuramente invertibile e i coefficienti si stabilizzano.

### 3.2 Tutte le feature contribuiscono

Diversamente da Lasso (che azzera alcuni coefficienti), Ridge **mantiene tutte le feature attive**, solo le restringe. Per Ames vogliamo questo comportamento: l'EDA mostra che molte feature minori contribuiscono cumulativamente, anche se nessuna è dominante.

### 3.3 Espansione OneHot crea sparsità "vera"

Dopo OneHotEncoding, molte colonne sono 0 per la maggior parte degli esempi (es. `Neighborhood_BrkSide` è 1 solo per le case in BrkSide). Lasso tenderebbe a zerare le categorie rare, perdendo segnale. Ridge le mantiene.

### 3.4 Quando usare invece Lasso

- Se il numero di feature è enorme e si vuole **selezione automatica** (modello sparso interpretabile).
- Se si è certi che molte feature siano effettivamente irrilevanti.
- In ML interpretabile: avere un modello con 10 feature attive è più semplice da spiegare di uno con 200.

## 4. Tuning di $\alpha$

L'iperparametro $\alpha$ va scelto via cross-validation. La griglia tipica per Ridge su Ames:

```python
alpha_grid = [0.1, 1.0, 5.0, 10.0, 30.0, 100.0]
```

Logiche per la scelta:

- **Scala logaritmica**: $\alpha$ entra moltiplicato → step lineari (1, 2, 3) coprono solo un ordine di grandezza, mentre (0.1, 1, 10, 100) ne coprono tre.
- **Range pratico**: con feature scalate ($\mu=0$, $\sigma=1$), valori di $\alpha$ &lt; 0.01 sono quasi-OLS, > 100 schiacciano tutto. Il punto ottimo è quasi sempre fra 1 e 100.

In Ames lo `α=10` ottimo è coerente con la letteratura (Pedro Marcelino su Kaggle ottiene risultati simili).

## 5. L'importanza dello scaling

Ridge **dipende dalla scala delle feature**: la penalità $\sum \beta_j^2$ tratta tutti i $\beta_j$ allo stesso modo, ma se una feature è in metri² (~1000) e un'altra in (0,1) per dummy, la prima riceve un coefficiente piccolo per costruzione → penalità trascurabile, mentre la seconda subisce shrinking eccessivo.

**Soluzione**: applicare `StandardScaler` (media 0, std 1) prima di Ridge. Nella nostra pipeline:

```python
Pipeline([
    ("preprocessor", column_transformer),  # imputazione + encoding
    ("scaler", StandardScaler(with_mean=False)),
    ("model", Ridge(alpha=10.0)),
])
```

`with_mean=False` evita di centrare le feature OneHot (sono sparse 0/1, centrarle distrugge la sparsità senza vantaggi).

## 6. Errori comuni

| Errore | Conseguenza | Fix |
|---|---|---|
| Scaling FUORI dalla pipeline (sul DataFrame intero prima dello split) | **Data leakage**: il test set influenza la media/std → metriche ottimistiche | Mettere `StandardScaler` come step interno della pipeline |
| Tuning $\alpha$ sul test set | Test set non più indipendente → metriche ottimistiche | Tuning solo via K-fold sul train; misurare su test holdout *una sola volta* |
| Confrontare Ridge ($\alpha$ scelto) con OLS senza scaling | Confronto ingiusto | Stessa pipeline, solo `alpha=0` per OLS |
| `Lasso` su feature OneHot senza min_frequency | Categorie rare azzerate ma il loro segnale era utile | Usare `OneHotEncoder(min_frequency=2)` per filtrare a monte |

## 7. Riferimenti

- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*, cap. 3.4.
- **Sklearn user guide**: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html).
- **Pedro Marcelino**, *Comprehensive data exploration with Python* (Kaggle Ames notebook classico).
