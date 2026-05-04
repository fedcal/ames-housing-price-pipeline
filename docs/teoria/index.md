---
layout: default
title: Teoria
nav_order: 2
has_children: true
permalink: /teoria/
description: >-
  Fondamenti teorici della pipeline Ames Housing: regressione lineare,
  regolarizzazione, modelli tree-based, metriche di valutazione e prevenzione
  del data leakage.
---

# Teoria

I cinque articoli di questa sezione costruiscono progressivamente le basi
necessarie per leggere il progetto **Ames Housing Pipeline** e per ragionare
in modo critico sui risultati.

## Percorso consigliato di lettura

| Capitolo | Titolo | Concetti chiave |
|:--|:--|:--|
| 1 | [Regressione & target log](01_regressione_e_target_log/) | OLS, skewness, `log1p`, `TransformedTargetRegressor` |
| 2 | [Regolarizzazione](02_regolarizzazione/) | Ridge L2, Lasso L1, ElasticNet, bias-variance |
| 3 | [Random Forest & Boosting](03_random_forest_e_boosting/) | Ensemble, bagging, gradient boosting, XGBoost |
| 4 | [Metriche di regressione](04_metriche_regressione/) | RMSE, MAE, MAPE, R², quando usarle |
| 5 | [Pipeline & data leakage](05_pipeline_e_data_leakage/) | `Pipeline` sklearn, K-fold corretta, anti-pattern |

{: .note }
> Ogni capitolo è autocontenuto: leggi nell'ordine se vuoi una progressione
> didattica, oppure salta direttamente al capitolo che ti serve.

## Riferimenti trasversali

- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (2009).
- De Cock, D. (2011) — *Ames, Iowa: An Alternative to the Boston Housing Data
  as an End of Semester Regression Project*, JSE 19(3).
- Chen, T. & Guestrin, C. (2016) — *XGBoost: A Scalable Tree Boosting System*.
