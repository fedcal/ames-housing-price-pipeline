---
layout: default
title: Metriche di regressione
parent: Teoria
nav_order: 4
math: mathjax
description: >-
  RMSE, MAE, MAPE, R²: cosa misurano davvero, come leggerle, quando una
  preferire all'altra in scenari reali di regressione.
---

# Metriche di valutazione per la regressione
{: .no_toc }

## Indice
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 1. Le quattro metriche standard

In regressione, ogni predizione $\hat{y}_i$ produce un **residuo** $r_i = y_i - \hat{y}_i$. Le metriche aggregano i residui in un unico numero, ognuna con un'interpretazione specifica.

### 1.1 RMSE — Root Mean Squared Error

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

- **Unità**: stessa di $y$ (per Ames, dollari).
- **Penalizza errori grandi quadraticamente**: un errore di $50,000 pesa 100× quanto un errore di $5,000.
- **Sensibile agli outlier**: un singolo prezzo fortemente sbagliato può dominare la metrica.

### 1.2 MAE — Mean Absolute Error

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

- **Unità**: stessa di $y$.
- **Robusto agli outlier**: pesa tutti gli errori linearmente.
- **Comparabile direttamente con il "tipico errore" intuitivo del cliente**.

### 1.3 R² — Coefficient of Determination

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- **Adimensionale** (range $(-\infty, 1]$).
- **Interpretazione**: frazione della varianza di $y$ spiegata dal modello.
  - $R^2 = 1$: predizione perfetta.
  - $R^2 = 0$: il modello non fa meglio che predire la media.
  - $R^2 < 0$: il modello è peggiore della media (sì, è possibile).
- Su Ames, R² > 0.90 è il benchmark per modelli ben tunati.

### 1.4 RMSE-log

$$
\text{RMSE}_{\log} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \big(\log(1+y_i) - \log(1+\hat{y}_i)\big)^2}
$$

- **Adimensionale** (relativo).
- **Penalizza errori percentuali equamente** in tutte le fasce di prezzo.
- **Metrica della Kaggle Ames Competition** — usare per confrontarsi con la leaderboard.

### 1.5 MAPE — Mean Absolute Percentage Error

$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- **In percentuale**: comprensibile a stakeholder non tecnici.
- **Indefinito per $y_i = 0$**: in Ames non capita.
- Un MAPE del 7-8% significa "in media, sbaglio il prezzo del 7-8%".

## 2. Quale scegliere?

Dipende dal **costo dell'errore**.

| Scenario business | Metrica primaria | Perché |
|---|---|---|
| Stima per agenzie immobiliari (errori grandi sono catastrofici) | **RMSE** | Penalizza sbagli grossi |
| Reportistica per clienti (errore tipico interpretato come "media") | **MAE** | Robusto, comprensibile |
| Confronto fra modelli/leaderboard | **R²** o **RMSE_log** | Adimensionali |
| Comunicazione esecutiva | **MAPE** | "Sbagliamo del 7%" è più chiaro di "RMSE = $20,000" |
| Modelli per decisioni di investimento | **RMSE** sulla coda costosa | Errori sui prezzi alti hanno impatto $$ |

In progetti seri si **riportano TUTTE** e si commenta. Il nostro report (`reports/holdout_metrics.json`) le calcola tutte.

## 3. Errori sistematici vs casuali — leggere i residui

Le metriche aggregate non bastano: bisogna **diagnosticare** la distribuzione dei residui.

### 3.1 Plot predizioni vs reali

Atteso: punti vicini alla bisettrice $y = \hat{y}$. Pattern problematici:

- **Sotto-stima sistematica per case costose**: punti sopra la bisettrice nella zona alta. Indica che il modello "sottoadatta" la coda — tipico se non si è applicata la trasformazione log.
- **Curva**: il modello non cattura una non linearità.
- **Cluster separati**: forse manca una feature categorica importante (es. `Neighborhood`).

### 3.2 Istogramma dei residui

Atteso: distribuzione ~normale, centrata sullo 0, varianza costante. Pattern:

- **Skew**: residuali asimmetrici → trasformazione del target sbagliata.
- **Code lunghe**: outlier non gestiti (controlla se sono data entry o vendite anomale).
- **Eteroscedasticità** (varianza che cresce con $\hat{y}$): manca trasformazione log o servono feature non lineari.

### 3.3 Residuals vs predizioni

Plot $r_i$ vs $\hat{y}_i$: idealmente nessun pattern. Se vedi una "U" o una "campana", c'è non linearità non catturata.

## 4. Cross-validation vs holdout

Distinzione critica:

- **Cross-validation (K-fold)** sul TRAIN set: misura *stabilità* del modello — quanto la performance dipende dal campione. Riportiamo `mean ± std` su 5 fold.
- **Holdout test set** (non visto in CV né in tuning): misura *generalizzazione finale*. Riportato **una sola volta** alla fine.

Se le metriche CV e holdout divergono molto (es. CV RMSE 18k, holdout 25k), c'è un problema:
- Train e test hanno distribuzioni diverse (drift).
- Il tuning ha overfittato i fold (es. test set "incrociato" via leakage).

In Ames, con split casuale stratificato e niente leakage, divergenza CV ↔ holdout < 5% è normale.

## 5. Confronto fra modelli — significatività statistica

Quando confrontiamo due modelli con CV, non basta vedere chi ha la media più bassa: bisogna chiedersi se la differenza è **statisticamente significativa** dato il rumore di campionamento.

Approccio veloce: se l'intervallo di confidenza al 95% delle medie CV (≈ media ± 1.96·std/√k) si **sovrappone**, la differenza non è certa. Con K=5 questo è una stima grezza ma utile.

Approccio rigoroso: **paired t-test** o **Wilcoxon signed-rank** sui RMSE per fold.

In pratica, su Ames, differenze < 1% fra modelli sono nel rumore. Si sceglie il modello migliore considerando anche **interpretabilità**, **velocità di inferenza**, **manutenibilità**.

## 6. Riferimenti

- **Sklearn user guide**: [Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).
- **Kaggle Ames House Prices**: [scoring metric](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation) — RMSE-log.
- **Hyndman & Koehler** (2006), *Another look at measures of forecast accuracy*, IJF 22 — discussione critica di MAPE e alternative.
