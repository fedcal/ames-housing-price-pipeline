---
layout: default
title: Scelte tecniche
nav_order: 3
has_children: true
permalink: /scelte_tecniche/
description: >-
  Decisioni architetturali e di modellazione del progetto Ames Housing
  Pipeline, con trade-off espliciti e razionali documentati.
---

# Scelte tecniche

Questa sezione documenta **come** è costruito il progetto e **perché** ogni
componente è stata progettata in un certo modo. È pensata per chi vuole
estendere o adattare la pipeline a un dominio simile.

## Capitoli

| Capitolo | Titolo | Cosa contiene |
|:--|:--|:--|
| 1 | [Architettura](architettura/) | Moduli `src/`, flusso dati, CLI, dipendenze fra componenti. |
| 2 | [Scelte di modellazione](scelte_modello/) | Selezione famiglie di modelli, gestione del target, strategia di tuning, gestione del rischio. |

{: .tip }
> Per la teoria sottostante (algoritmi, metriche, regolarizzazione) consulta
> la sezione **[Teoria](../teoria/)**.
