---
layout: default
title: Random Forest & Boosting
parent: Teoria
nav_order: 3
math: mathjax
description: >-
  Modelli tree-based per dati tabulari: Random Forest, Gradient Boosting,
  XGBoost. Bagging vs boosting, iperparametri chiave, perché dominano su tabular.
---

# Random Forest e Gradient Boosting
{: .no_toc }

> *"In data, the easy money is in trees."* — Leo Breiman (parafrasato dalla comunità ML).

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Perché i tree-based sono lo state-of-the-art su tabular

Su dati tabulari (righe = istanze, colonne = feature di tipo misto), i modelli **basati su alberi** (Random Forest, Gradient Boosting) battono quasi sempre le reti neurali in termini di accuracy/effort. Le ragioni:

1. **Invarianza alla scala**: nessun bisogno di normalizzare. Gli split sono basati su soglie, non su distanze.
2. **Interazioni catturate naturalmente**: un albero che split prima su `Neighborhood` e poi su `OverallQual` modella implicitamente l'interazione fra le due feature.
3. **Robustezza a outlier sulle feature**: un valore estremo viene confinato in una foglia separata, non distorce l'intera predizione.
4. **Gestione nativa di feature miste**: numeriche, categoriche encodate, binarie — tutte trattate uniformemente.

## 2. Decision Tree singolo

Un albero di regressione split ricorsivamente lo spazio delle feature in regioni rettangolari $R_1, R_2, ..., R_M$. La predizione in una regione è la media degli $y$ del training:

$$
\hat{y}(\mathbf{x}) = \sum_{m=1}^{M} \bar{y}_{R_m} \cdot \mathbb{1}\{\mathbf{x} \in R_m\}
$$

Lo split che divide la regione genitore in due figlie viene scelto **greedy**: per ogni feature $j$ e soglia $s$, valutiamo la riduzione della varianza:

$$
\Delta = \text{Var}(R_{\text{padre}}) - \frac{|R_{\text{sx}}|}{|R_{\text{padre}}|} \text{Var}(R_{\text{sx}}) - \frac{|R_{\text{dx}}|}{|R_{\text{padre}}|} \text{Var}(R_{\text{dx}})
$$

e scegliamo $(j^*, s^*) = \arg\max \Delta$. Si ferma quando la regione ha pochi campioni o la riduzione di varianza è trascurabile.

**Limite del singolo albero**: alta varianza. Bastano poche osservazioni in più nel training per cambiare radicalmente la struttura → predizioni molto sensibili al campione.

## 3. Random Forest: bagging + decorrelazione

L'idea di **Breiman (2001)** è semplice e potente:

1. **Bootstrap aggregating (bagging)**: addestrare $B$ alberi su $B$ sotto-campioni casuali (con rimpiazzo) del training set.
2. **Decorrelazione**: ad ogni split, considerare solo un sottoinsieme casuale di feature (di dimensione $\sqrt{p}$ tipicamente).
3. **Predizione finale**: media delle predizioni degli alberi.

Il risultato è un modello con **bias simile al singolo albero** ma **varianza drasticamente ridotta** ($\sim 1/B$ per alberi indipendenti, ma in pratica meno per via della correlazione residua).

### 3.1 Iperparametri principali

| Iperparametro       | Effetto                                     | Range tipico |
|---------------------|---------------------------------------------|--------------|
| `n_estimators`      | Numero di alberi                            | 200-2000 (più = meglio finché ROI > 0) |
| `max_depth`         | Profondità massima singolo albero           | None (cresci fino in fondo) o 10-30 |
| `min_samples_split` | Min campioni per fare uno split             | 2-20 |
| `max_features`      | Feature considerate per split               | `'sqrt'` (default) o 0.3-0.5 |
| `min_samples_leaf`  | Min campioni in una foglia (regolarizzazione) | 1-10 |

### 3.2 Punti deboli

- **Estrapolazione**: una RF non predice mai oltre il range del training (la sua predizione è sempre una media di valori visti). Se nel test ci sono case con `GrLivArea` superiore al massimo del training, la RF satura — un Ridge no.
- **Bias residuo**: gli alberi singoli sono "deboli"; se il segnale è quasi-lineare, una RF è sub-ottimale rispetto a Ridge.

## 4. Gradient Boosting: imparare sequenzialmente dagli errori

L'idea di **Friedman (2001)**: invece di aggregare alberi indipendenti, addestrarli **in sequenza**, ogni albero correggendo gli errori dei precedenti.

L'algoritmo a passi:

1. Inizializza $F_0(\mathbf{x}) = \bar{y}$ (media del training).
2. Per $t = 1, ..., T$:
   1. Calcola i residui (gradiente negativo della loss): $r_i^{(t)} = y_i - F_{t-1}(\mathbf{x}_i)$.
   2. Addestra un piccolo albero $h_t$ per predire $r_i^{(t)}$.
   3. Aggiorna $F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \eta \cdot h_t(\mathbf{x})$ con $\eta$ = learning rate.
3. Predizione finale: $F_T(\mathbf{x})$.

Il learning rate $\eta$ piccolo (~0.05) + tanti alberi ($T$ ~ 500-2000) è il classico trade-off: passi piccoli verso l'ottimo riducono il rischio di overfit ma richiedono più iterazioni.

### 4.1 XGBoost: gradient boosting industriale

**XGBoost** (Chen & Guestrin, 2016) è l'implementazione più usata. Aggiunge:

- **Approssimazione di secondo ordine** (Newton-Raphson) → split decisi più informativi.
- **Regolarizzazione esplicita** $\alpha L_1 + \lambda L_2$ sui pesi delle foglie.
- **Subsample stocastico** di righe e colonne (riduce overfit).
- **`tree_method='hist'`**: discretizzazione delle feature numeriche → CPU 5-10× più veloce su dataset di queste dimensioni.

### 4.2 Iperparametri chiave per XGBoost

| Iperparametro      | Effetto                                  | Range pratico |
|--------------------|------------------------------------------|---------------|
| `n_estimators`     | Numero alberi (boosting rounds)          | 500-2000 |
| `learning_rate`    | Step size                                | 0.01-0.1 |
| `max_depth`        | Profondità albero                        | 3-8 |
| `subsample`        | % righe usate per albero                 | 0.7-1.0 |
| `colsample_bytree` | % feature usate per albero               | 0.7-1.0 |
| `reg_alpha` (L1)   | Penalità L1 sui pesi foglia              | 0-1 |
| `reg_lambda` (L2)  | Penalità L2 sui pesi foglia              | 1-10 |

**Combinazione vincente tipica**: `lr=0.05, n_estimators=800, max_depth=5, subsample=0.8, colsample_bytree=0.8`.

## 5. Confronto su Ames Housing

| Modello       | Pro su Ames                                  | Contro su Ames |
|---------------|----------------------------------------------|----------------|
| **Ridge**     | Veloce, interpretabile, segnale lineare forte | Non cattura interazioni complesse |
| **RandomForest** | Buon equilibrio bias-varianza             | Sub-ottimale: il segnale è prevalentemente lineare |
| **XGBoost**   | Best accuracy quando ben tunato              | Richiede più tuning, meno interpretabile |

Risultati attesi sul holdout test (con tuning completo):
- Ridge: RMSE ≈ $20,000, R² ≈ 0.94
- RF:    RMSE ≈ $22,000, R² ≈ 0.92
- XGB:   RMSE ≈ $19,000, R² ≈ 0.95

Differenze nell'ordine del 5-10% — non drammatiche. Su Ames, **il preprocessing accurato pesa più della scelta del modello**.

## 6. Quando NON usare i tree-based

- **Dati ad alta dimensionalità sparsi** (es. testo, dummy esplose con $p \gg n$): Ridge/Lasso vincono.
- **Estrapolazione richiesta**: i tree non escono dal range del training.
- **Modelli interpretabili imposti da requisiti** (es. credit scoring soggetto a regulator): Ridge con coefficienti standardizzati è più trasparente.
- **Vincoli di latenza estremi**: 800 alberi XGBoost in inferenza sono ~50 ms; un Ridge è < 1 ms.

## 7. Riferimenti

- **Breiman, L.** (2001), *Random Forests*, Machine Learning 45(1).
- **Friedman, J.** (2001), *Greedy Function Approximation: A Gradient Boosting Machine*, Annals of Statistics.
- **Chen, T. & Guestrin, C.** (2016), *XGBoost: A Scalable Tree Boosting System*, KDD.
- **Hastie, Tibshirani, Friedman**, *The Elements of Statistical Learning*, cap. 10 (boosting) e 15 (random forests).
