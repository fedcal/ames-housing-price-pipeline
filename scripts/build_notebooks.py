"""Genera i 4 notebook didattici in `notebooks/`.

Lo script è la SORGENTE DI VERITÀ dei notebook: per modificarli si
edita questo file e si rilancia (`python scripts/build_notebooks.py`).
Vantaggi:
    - Sorgente in formato testuale → diff Git leggibili.
    - Riproducibilità (chiunque rigenera notebook identici).
    - Niente metadati casuali (kernel locale, output cache) committati.

I notebook vengono SCRITTI ma NON ESEGUITI dallo script: l'esecuzione
end-to-end è gestita separatamente da `scripts/run_notebooks.sh`.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write_notebook(name: str, cells: list[nbf.NotebookNode]) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    out = NB_DIR / name
    nbf.write(nb, out)
    print(f"[OK] {out.relative_to(ROOT)}  ({len(cells)} celle)")


# ============================================================================
# Notebook 01 — Esplorazione dei dati (EDA)
# ============================================================================
nb01 = [
    md(
        "# 01 — Esplorazione del dataset Ames Housing\n\n"
        "## Obiettivi didattici\n\n"
        "1. Comprendere la struttura del dataset (2930 osservazioni, 82 variabili).\n"
        "2. Distinguere tipi di variabili: numeriche, categoriche nominali, "
        "categoriche ordinali, identificatori.\n"
        "3. Esaminare la **distribuzione del target** `SalePrice` e identificare "
        "la skewness, motivando la trasformazione logaritmica.\n"
        "4. Analizzare i **valori mancanti** distinguendo fra missing strutturali "
        "(es. `PoolQC=NaN` ⇒ niente piscina) e missing veri.\n"
        "5. Identificare gli **outlier** raccomandati da De Cock (2011).\n\n"
        "## Riferimenti\n\n"
        "- De Cock, D. (2011). *Ames, Iowa: Alternative to the Boston Housing Data...* "
        "Journal of Statistics Education 19(3).\n"
        "- Documentazione delle 82 variabili: `data/raw/DataDocumentation.txt`.\n"
    ),
    code(
        "import sys\n"
        "sys.path.insert(0, '../src')\n"
        "\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "\n"
        "from ames_pipeline.data import load_raw\n"
        "from ames_pipeline.wrangling import (\n"
        "    fill_structural_missing,\n"
        "    remove_grliv_area_outliers,\n"
        "    NA_AS_NONE_CATEGORICAL,\n"
        "    NA_AS_ZERO_NUMERIC,\n"
        ")\n"
        "from ames_pipeline.preprocessing import infer_column_groups, ORDINAL_CATEGORIES_MAP\n"
    ),
    md(
        "## Caricamento\n\n"
        "Il dataset è il file originale `AmesHousing.txt` (tab-separated) "
        "scaricato dalla pagina ufficiale del Journal of Statistics Education. "
        "La funzione `load_raw()` esegue:\n\n"
        "1. download dal mirror JSE se assente,\n"
        "2. validazione SHA-256 (per riproducibilità),\n"
        "3. normalizzazione dei nomi colonna (rimuove spazi e `/`).\n"
    ),
    code(
        "df = load_raw()\n"
        "print(f'Shape: {df.shape}')\n"
        "print(f'Target: SalePrice  (min=${df.SalePrice.min():,}, max=${df.SalePrice.max():,})')\n"
        "df.head(3)"
    ),
    md(
        "## Composizione delle variabili\n\n"
        "Suddividiamo le colonne in tre famiglie (escludendo gli identificatori "
        "`Order`, `PID` che vanno droppati subito):\n\n"
        "- **Numeriche** continue/discrete (`LotArea`, `GrLivArea`, `OverallQual`, …)\n"
        "- **Ordinali** (qualità decrescente: Po<Fa<TA<Gd<Ex)\n"
        "- **Nominali** (`Neighborhood`, `HouseStyle`, …)\n\n"
        "*Nota didattica:* la classificazione qui è basata sulla documentazione "
        "del dataset, non su euristica automatica. Inferire 'numerica vs categorica' "
        "dal solo dtype porta errori (es. `OverallQual` è int ma logicamente ordinale).\n"
    ),
    code(
        "groups = infer_column_groups(df.drop(columns=['Order','PID','SalePrice']))\n"
        "for k, cols in groups.items():\n"
        "    print(f'{k:9s}: {len(cols):2d} colonne')\n"
        "print()\n"
        "print('Ordinali con ordine semantico predefinito:')\n"
        "for col in groups['ordinal']:\n"
        "    print(f'  {col:15s} → {ORDINAL_CATEGORIES_MAP.get(col)}')\n"
    ),
    md(
        "## Distribuzione del target `SalePrice`\n\n"
        "Il prezzo è **fortemente asimmetrico a destra** (long tail di case di lusso). "
        "Conseguenza: i modelli che minimizzano l'errore quadratico (RMSE) sono "
        "dominati dalle case costose. Trasformiamo il target con **log1p** per:\n\n"
        "- avvicinarlo a una gaussiana (assunzione comoda per Ridge),\n"
        "- rendere l'errore relativo più uniforme su tutte le fasce di prezzo,\n"
        "- usare RMSE-log come metrica (lo stesso scoring di Kaggle Ames Competition).\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].hist(df.SalePrice, bins=60, edgecolor='black')\n"
        "axes[0].set(title='SalePrice (scala originale)', xlabel='$', ylabel='count')\n"
        "axes[0].ticklabel_format(style='plain', axis='x')\n"
        "axes[1].hist(np.log1p(df.SalePrice), bins=60, edgecolor='black', color='C1')\n"
        "axes[1].set(title='log1p(SalePrice) — quasi-normale', xlabel='log $')\n"
        "fig.tight_layout(); plt.show()\n"
        "\n"
        "print(f'Skewness originale  : {df.SalePrice.skew():.3f}')\n"
        "print(f'Skewness log1p      : {np.log1p(df.SalePrice).skew():.3f}')\n"
    ),
    md(
        "## Analisi dei valori mancanti\n\n"
        "**Punto critico**: in Ames Housing, il `NaN` di una colonna come `PoolQC` "
        "non è un dato mancante — significa **\"casa senza piscina\"**. La "
        "documentazione di De Cock lo specifica per ~16 colonne. Imputare la "
        "moda/mediana qui sarebbe sbagliato: introdurrebbe rumore. Mappiamo invece "
        "questi NaN a una categoria 'None' (per categoriche) o `0` (per numeriche "
        "associate). Solo `LotFrontage` (e residui) hanno mancanti veri da imputare."
    ),
    code(
        "missing = df.isna().sum()\n"
        "missing = missing[missing > 0].sort_values(ascending=False)\n"
        "missing_df = pd.DataFrame({\n"
        "    'col': missing.index,\n"
        "    'missing': missing.values,\n"
        "    'pct': (missing.values / len(df) * 100).round(1),\n"
        "    'tipo': [\n"
        "        'strutturale (NA→None)' if c in NA_AS_NONE_CATEGORICAL else\n"
        "        'strutturale (NA→0)'    if c in NA_AS_ZERO_NUMERIC else\n"
        "        'vero (impute)'\n"
        "        for c in missing.index\n"
        "    ],\n"
        "})\n"
        "missing_df"
    ),
    code(
        "df_clean = fill_structural_missing(df)\n"
        "still_missing = df_clean.isna().sum()\n"
        "still_missing = still_missing[still_missing > 0]\n"
        "print(f'Dopo fill_structural_missing rimangono mancanti su:')\n"
        "print(still_missing.to_string())\n"
        "print('\\n→ Solo veri mancanti (verranno imputati con SimpleImputer dentro la pipeline).')\n"
    ),
    md(
        "## Outlier raccomandati da De Cock\n\n"
        "Il paper originale segnala 5 case con `GrLivArea > 4000 ft²` e prezzo "
        "anomalmente basso. Sono vendite particolari (parziali, fra parenti, "
        "liquidazioni) che rovinano i modelli lineari. Vanno rimosse PRIMA dello split."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "ax.scatter(df.GrLivArea, df.SalePrice, alpha=0.4, s=15)\n"
        "outliers = df[(df.GrLivArea > 4000) & (df.SalePrice < 300_000)]\n"
        "ax.scatter(outliers.GrLivArea, outliers.SalePrice, color='red', s=80,\n"
        "           label=f'outlier ({len(outliers)})', zorder=5)\n"
        "ax.set(xlabel='GrLivArea (ft²)', ylabel='SalePrice ($)',\n"
        "       title='Outlier GrLivArea segnalati da De Cock')\n"
        "ax.legend(); plt.show()\n"
    ),
    md(
        "## Top correlazioni con il target\n\n"
        "Le correlazioni di Pearson catturano relazioni lineari. Le più informative "
        "guideranno la baseline lineare; per i modelli non lineari (RF/XGB) tutte "
        "le feature contribuiscono."
    ),
    code(
        "df_num = df_clean.select_dtypes(include='number').drop(columns=['Order','PID'])\n"
        "corr = df_num.corr(numeric_only=True)['SalePrice'].sort_values(ascending=False)\n"
        "top10 = corr.head(11).iloc[1:]  # esclude SalePrice stesso\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "ax.barh(top10.index[::-1], top10.values[::-1])\n"
        "ax.set(title='Top-10 correlazioni di Pearson con SalePrice',\n"
        "       xlabel='Pearson r')\n"
        "for i, v in enumerate(top10.values[::-1]):\n"
        "    ax.text(v, i, f' {v:.3f}', va='center')\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md(
        "## Conclusioni dell'EDA e implicazioni per il modeling\n\n"
        "| Osservazione | Implicazione |\n"
        "|---|---|\n"
        "| Target skewed | Usare `log1p(SalePrice)` come variabile dipendente. |\n"
        "| 16+ colonne con NaN strutturali | Pre-fill **prima** dello split (no leakage: è semantica). |\n"
        "| `LotFrontage` ha 490 missing veri | Imputazione **dentro** la pipeline (mediana sul train). |\n"
        "| 5 outlier `GrLivArea` | Rimozione raccomandata da De Cock prima dello split. |\n"
        "| 23 ordinali con ordinamento Po<Fa<TA<Gd<Ex | `OrdinalEncoder` con `categories=...` esplicite. |\n"
        "| `OverallQual` r=0.80 con prezzo | Feature dominante; baseline lineare già forte. |\n"
        "| Cardinalità di `Neighborhood`=28 | OneHotEncoder con `min_frequency=2`. |\n\n"
        "→ Procedi al notebook **02_preprocessing_features**.\n"
    ),
]
write_notebook("01_eda.ipynb", nb01)


# ============================================================================
# Notebook 02 — Preprocessing & Feature Engineering
# ============================================================================
nb02 = [
    md(
        "# 02 — Preprocessing e Feature Engineering\n\n"
        "## Obiettivi didattici\n\n"
        "1. Costruire una pipeline `sklearn` **resistente al data leakage**.\n"
        "2. Distinguere encoding **ordinale** vs **one-hot** e applicare ognuno "
        "alle colonne corrette.\n"
        "3. Aggiungere feature derivate domain-specific (TotalSF, HouseAge, "
        "QualityScore, ...).\n"
        "4. Verificare la composizione del preprocessor (numero di feature in "
        "uscita, sparsità, ...).\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "from ames_pipeline.data import load_raw\n"
        "from ames_pipeline.wrangling import fill_structural_missing, remove_grliv_area_outliers\n"
        "from ames_pipeline.features import AmesFeatureEngineer\n"
        "from ames_pipeline.preprocessing import build_preprocessor, infer_column_groups\n"
    ),
    md(
        "## Step 1 — Wrangling (no leakage)\n\n"
        "Le operazioni che possono essere fatte fuori dalla pipeline sklearn "
        "sono SOLO quelle che non dipendono da statistiche del training set. "
        "In Ames sono due:\n\n"
        "- `fill_structural_missing`: mappatura semantica di NaN (no statistiche).\n"
        "- `remove_grliv_area_outliers`: rimozione di record fissi (no statistiche).\n\n"
        "Tutto il resto (imputazione mediana, scaling, encoding) **deve** stare "
        "dentro la pipeline."
    ),
    code(
        "df = load_raw()\n"
        "df = fill_structural_missing(df)\n"
        "df = remove_grliv_area_outliers(df)\n"
        "X = df.drop(columns=['Order','PID','SalePrice'])\n"
        "y = df['SalePrice']\n"
        "print(f'X.shape={X.shape}, y.shape={y.shape}')\n"
    ),
    md(
        "## Step 2 — Feature engineering custom\n\n"
        "Creiamo feature derivate **interpretabili**:\n\n"
        "| Nuova feature        | Definizione                                  | Razionale |\n"
        "|----------------------|----------------------------------------------|-----------|\n"
        "| `HouseAge`           | YrSold - YearBuilt                           | Età al momento della vendita. |\n"
        "| `YearsSinceRemodel`  | YrSold - YearRemodAdd                        | Modernità della ristrutturazione. |\n"
        "| `TotalSF`            | 1stFlrSF + 2ndFlrSF + TotalBsmtSF            | Area totale abitabile. |\n"
        "| `TotalBath`          | full + 0.5·half (sopra + sotto)              | Conteggio bagni pesato. |\n"
        "| `QualityScore`       | OverallQual × OverallCond                    | Interazione qualità×condizione. |\n"
        "| `HasGarage/Pool/...` | binarie da area > 0                           | Catturano presenza/assenza. |\n"
        "| `AreaPerRoom`        | GrLivArea / (TotRmsAbvGrd + 1)               | Densità abitativa. |\n\n"
        "Le feature derivate aiutano soprattutto i **modelli lineari**, che faticano "
        "a modellare interazioni e rapporti non lineari. RF/XGB le imparerebbero "
        "comunque, ma renderle esplicite migliora interpretabilità e velocità."
    ),
    code(
        "fe = AmesFeatureEngineer()\n"
        "X_fe = fe.fit_transform(X)\n"
        "new_cols = sorted(set(X_fe.columns) - set(X.columns))\n"
        "print(f'Feature aggiunte ({len(new_cols)}):')\n"
        "for c in new_cols:\n"
        "    print(f'  - {c}')\n"
    ),
    md(
        "## Step 3 — ColumnTransformer\n\n"
        "Il preprocessor è un `ColumnTransformer` con tre branch parallele:\n\n"
        "```\n"
        "   ┌── numeriche  → SimpleImputer(median)\n"
        "   ├── ordinali   → SimpleImputer('None') + OrdinalEncoder(categories=...)\n"
        "   └── nominali   → SimpleImputer('None') + OneHotEncoder(handle_unknown='ignore')\n"
        "```\n\n"
        "**Perché branch separate?** Ognuna richiede una strategia diversa di "
        "imputazione/encoding. Il `ColumnTransformer` le combina mantenendo la "
        "concatenazione corretta delle feature in uscita."
    ),
    code(
        "groups = infer_column_groups(X_fe)\n"
        "preprocessor = build_preprocessor(\n"
        "    numeric_cols=groups['numeric'],\n"
        "    ordinal_cols=groups['ordinal'],\n"
        "    nominal_cols=groups['nominal'],\n"
        ")\n"
        "preprocessor"
    ),
    md(
        "## Step 4 — Verifica forma di output\n\n"
        "Il `OneHotEncoder` espande le categoriche nominali in dummy. Vediamo "
        "il numero finale di feature e quante sono dummy."
    ),
    code(
        "X_transformed = preprocessor.fit_transform(X_fe)\n"
        "n_in = X_fe.shape[1]\n"
        "n_out = X_transformed.shape[1]\n"
        "print(f'Feature in:  {n_in}')\n"
        "print(f'Feature out: {n_out}  (espansione {(n_out / n_in - 1) * 100:.0f}% dovuta al OneHotEncoder)')\n"
        "print(f'Densità: {(X_transformed != 0).mean():.2%}')\n"
    ),
    md(
        "## Conclusione\n\n"
        "Il preprocessor è una funzione pura — `fit` su training, `transform` su "
        "qualsiasi nuovo dato. Lo passiamo al notebook successivo come step "
        "iniziale di tre pipeline candidate."
    ),
]
write_notebook("02_preprocessing_features.ipynb", nb02)


# ============================================================================
# Notebook 03 — Modeling, K-fold CV, Hyperparameter Tuning
# ============================================================================
nb03 = [
    md(
        "# 03 — Modeling e Hyperparameter Tuning\n\n"
        "## Obiettivi didattici\n\n"
        "1. Confrontare tre famiglie di modelli (lineare, ensemble bagging, gradient boosting).\n"
        "2. Applicare **K-fold cross-validation** con scoring corretto (RMSE).\n"
        "3. Eseguire **Grid/Randomized search** su iperparametri rilevanti.\n"
        "4. Avvolgere il target con `TransformedTargetRegressor(log1p, expm1)` "
        "per gestire automaticamente la trasformazione.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd\n"
        "\n"
        "from ames_pipeline.data import load_raw\n"
        "from ames_pipeline.wrangling import fill_structural_missing, remove_grliv_area_outliers\n"
        "from ames_pipeline.features import AmesFeatureEngineer\n"
        "from ames_pipeline.preprocessing import build_preprocessor, infer_column_groups\n"
        "from ames_pipeline.models import get_all_pipelines\n"
        "from ames_pipeline.tuning import tune_all_models, summarize_tuning, wrap_with_log_target\n"
        "from ames_pipeline.config import DEFAULT_CONFIG, PipelineConfig\n"
        "from sklearn.pipeline import Pipeline\n"
        "from sklearn.model_selection import train_test_split, cross_val_score, KFold\n"
    ),
    md(
        "## Setup: dati + pipeline candidate\n\n"
        "Riproduciamo brevemente il flusso dei notebook precedenti."
    ),
    code(
        "df = load_raw()\n"
        "df = fill_structural_missing(df)\n"
        "df = remove_grliv_area_outliers(df)\n"
        "X = df.drop(columns=['Order','PID','SalePrice'])\n"
        "y = df['SalePrice']\n"
        "\n"
        "# Stratificazione su quintili di prezzo per stabilità\n"
        "y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y_bins,\n"
        ")\n"
        "print(f'train={len(X_train)}, test={len(X_test)}')\n"
        "\n"
        "fe = AmesFeatureEngineer()\n"
        "X_train_fe = fe.fit_transform(X_train)\n"
        "groups = infer_column_groups(X_train_fe)\n"
        "preprocessor = build_preprocessor(\n"
        "    numeric_cols=groups['numeric'],\n"
        "    ordinal_cols=groups['ordinal'],\n"
        "    nominal_cols=groups['nominal'],\n"
        ")\n"
        "\n"
        "# Antepone feature_engineer alle pipeline candidate\n"
        "base = get_all_pipelines(preprocessor)\n"
        "candidates = {\n"
        "    name: Pipeline(steps=[('feature_engineer', AmesFeatureEngineer())] + list(p.steps))\n"
        "    for name, p in base.items()\n"
        "}\n"
        "list(candidates.keys())\n"
    ),
    md(
        "## Baseline cross-validation (no tuning)\n\n"
        "Misura le performance dei modelli con iperparametri di default. Serve "
        "come riferimento per stimare il guadagno del tuning successivo."
    ),
    code(
        "cv = KFold(n_splits=5, shuffle=True, random_state=42)\n"
        "baseline = []\n"
        "for name, pipe in candidates.items():\n"
        "    wrapped = wrap_with_log_target(pipe)\n"
        "    rmse = -cross_val_score(wrapped, X_train, y_train,\n"
        "                            scoring='neg_root_mean_squared_error',\n"
        "                            cv=cv, n_jobs=-1)\n"
        "    baseline.append({'model': name, 'rmse_mean': rmse.mean(), 'rmse_std': rmse.std()})\n"
        "pd.DataFrame(baseline).sort_values('rmse_mean')"
    ),
    md(
        "## Tuning iperparametri\n\n"
        "**Strategia per modello**:\n\n"
        "- **Ridge**: `GridSearchCV` su 6 valori di α (10⁻¹ → 10²) — grid piccola.\n"
        "- **RandomForest**: `GridSearchCV` su 24 combinazioni — esauriente.\n"
        "- **XGBoost**: `RandomizedSearchCV` 30 sample su grid combinatoria di "
        "  ~200 combinazioni — efficiente e nella pratica raggiunge il 95-99% "
        "  del best score di GridSearch completa.\n\n"
        "**Scoring**: `neg_root_mean_squared_error` su `log1p(target)` (gestito "
        "automaticamente da `TransformedTargetRegressor`).\n\n"
        "Tempo atteso ~5-10 minuti su laptop senza GPU. Per smoke-test ridurre "
        "le grid in `config.py`."
    ),
    code(
        "config = DEFAULT_CONFIG\n"
        "results = tune_all_models(\n"
        "    pipelines=candidates,\n"
        "    X=X_train, y=y_train,\n"
        "    config=config,\n"
        "    xgb_n_iter=30,\n"
        ")\n"
        "summary = summarize_tuning(results)\n"
        "summary"
    ),
    md(
        "## Discussione iperparametri ottimi\n\n"
        "- **Ridge α**: `α=10` tipico. Valori troppo piccoli sovra-adattano ai "
        "rumori di OneHot; troppo grandi schiacciano il segnale.\n"
        "- **RandomForest**: con `max_depth=None` (alberi profondi) e "
        "`max_features='sqrt'` ottiene il miglior bias-variance trade-off.\n"
        "- **XGBoost**: `learning_rate=0.05` + `n_estimators=800` è il classico "
        "binomio (lr basso compensato da più alberi). `max_depth=5` evita "
        "l'overfit, `subsample=0.8` introduce regolarizzazione stocastica.\n"
    ),
    md(
        "## Esportiamo le pipeline migliori per il notebook successivo"
    ),
    code(
        "import joblib\n"
        "from pathlib import Path\n"
        "out_dir = Path('../reports/models')\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name, r in results.items():\n"
        "    path = out_dir / f'{name.lower()}_best.joblib'\n"
        "    joblib.dump(r.best_estimator, path)\n"
        "    print(f'salvato: {path.relative_to(Path(\"..\"))}  RMSE_log_cv={r.best_score:.4f}')\n"
    ),
]
write_notebook("03_modeling_tuning.ipynb", nb03)


# ============================================================================
# Notebook 04 — Evaluation, Diagnostica, Inferenza
# ============================================================================
nb04 = [
    md(
        "# 04 — Valutazione, diagnostica e inferenza\n\n"
        "## Obiettivi didattici\n\n"
        "1. Valutare i modelli sul **test holdout** (non visto durante il tuning).\n"
        "2. Calcolare le metriche standard di regressione: RMSE, MAE, R², "
        "RMSE-log, MAPE.\n"
        "3. Diagnosticare il modello con i **plot dei residui** e "
        "**predizioni vs reali**.\n"
        "4. Esaminare la **feature importance** per modelli tree-based.\n"
        "5. Esporre il modello tramite la funzione `predict_price()`.\n"
    ),
    code(
        "import sys; sys.path.insert(0, '../src')\n"
        "import warnings; warnings.filterwarnings('ignore')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "import joblib\n"
        "from pathlib import Path\n"
        "\n"
        "from ames_pipeline.data import load_raw\n"
        "from ames_pipeline.wrangling import fill_structural_missing, remove_grliv_area_outliers\n"
        "from ames_pipeline.evaluation import (\n"
        "    evaluate_on_holdout, regression_metrics,\n"
        "    plot_predictions_vs_actual, plot_residuals,\n"
        "    get_top_feature_importance, plot_feature_importance,\n"
        ")\n"
        "from ames_pipeline.inference import predict_price, example_input\n"
        "from sklearn.model_selection import train_test_split\n"
    ),
    md("## Ricostruiamo lo stesso split del notebook 03"),
    code(
        "df = fill_structural_missing(load_raw())\n"
        "df = remove_grliv_area_outliers(df)\n"
        "X = df.drop(columns=['Order','PID','SalePrice'])\n"
        "y = df['SalePrice']\n"
        "y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y_bins,\n"
        ")\n"
        "print(f'test set: {len(X_test)} record')\n"
    ),
    md("## Caricamento modelli salvati"),
    code(
        "models_dir = Path('../reports/models')\n"
        "model_paths = {p.stem.replace('_best','').capitalize(): p\n"
        "               for p in models_dir.glob('*_best.joblib')}\n"
        "models = {n: joblib.load(p) for n, p in model_paths.items()}\n"
        "list(models.keys())\n"
    ),
    md(
        "## Metriche su holdout test\n\n"
        "| Metrica  | Significato                               | Buon valore (Ames) |\n"
        "|----------|-------------------------------------------|---------------------|\n"
        "| RMSE     | $ medio di errore (penalizza errori grandi) | < $25,000 |\n"
        "| MAE      | $ medio di errore robusto agli outlier    | < $17,000 |\n"
        "| R²       | varianza spiegata                          | > 0.90 |\n"
        "| RMSE-log | scoring Kaggle                             | < 0.13 |\n"
        "| MAPE     | errore % medio assoluto                    | < 10% |\n"
    ),
    code(
        "rows = []\n"
        "for name, model in models.items():\n"
        "    m = evaluate_on_holdout(model, X_test, y_test)\n"
        "    rows.append({'model': name, **m.as_dict()})\n"
        "results_df = pd.DataFrame(rows).sort_values('rmse').set_index('model')\n"
        "results_df.style.format({\n"
        "    'rmse': '${:,.0f}', 'mae': '${:,.0f}',\n"
        "    'r2': '{:.4f}', 'rmse_log': '{:.4f}', 'mape': '{:.2f}%',\n"
        "})\n"
    ),
    md(
        "## Diagnostica grafica del modello migliore\n\n"
        "Cerchiamo:\n\n"
        "- **Predizioni vs reali**: punti vicini alla bisettrice = buon fit. "
        "Sistematica sopra/sotto-stima di certe fasce di prezzo è un warning.\n"
        "- **Residui (y_true - y_pred)**: distribuzione attesa ~normale, "
        "centrata sullo zero, varianza costante. Code lunghe ⇒ outlier o "
        "non linearità non catturate.\n"
    ),
    code(
        "best_name = results_df.index[0]\n"
        "print(f'Miglior modello: {best_name}')\n"
        "best = models[best_name]\n"
        "y_pred = best.predict(X_test)\n"
        "fig1 = plot_predictions_vs_actual(y_test.values, y_pred,\n"
        "    title=f'{best_name}: predizioni vs reali (test)')\n"
        "plt.show()\n"
        "fig2 = plot_residuals(y_test.values, y_pred,\n"
        "    title=f'{best_name}: residui (test)')\n"
        "plt.show()\n"
    ),
    md(
        "## Feature importance (modelli tree-based)\n\n"
        "Solo `RandomForest` e `XGBoost` espongono `feature_importances_`. "
        "Per Ridge si possono usare i coefficienti standardizzati."
    ),
    code(
        "for name in ('Randomforest', 'Xgboost'):\n"
        "    if name not in models:\n"
        "        continue\n"
        "    model = models[name]\n"
        "    # Ricavo i nomi feature post-preprocessor\n"
        "    try:\n"
        "        ttr = model\n"
        "        from sklearn.compose import TransformedTargetRegressor\n"
        "        if isinstance(ttr, TransformedTargetRegressor):\n"
        "            ttr = ttr.regressor_\n"
        "        feature_names = ttr.named_steps['preprocessor'].get_feature_names_out()\n"
        "    except Exception:\n"
        "        feature_names = []\n"
        "    df_imp = get_top_feature_importance(model, feature_names, top_n=20)\n"
        "    print(f'\\n=== {name} top-20 ===')\n"
        "    print(df_imp.to_string(index=False))\n"
        "    plot_feature_importance(df_imp, title=f'{name}: top-20 feature importance')\n"
        "    plt.show()\n"
    ),
    md(
        "## Inferenza: la funzione `predict_price()`\n\n"
        "L'API è semplice: si passa un dizionario con le feature note (anche un "
        "sottoinsieme) e ritorna il prezzo predetto in dollari. Le colonne "
        "mancanti sono imputate automaticamente."
    ),
    code(
        "base = example_input()\n"
        "print(f\"Casa-base: ${predict_price(base):,.0f}\")\n"
        "\n"
        "luxury = {**base, 'OverallQual': 9, 'GrLivArea': 2400, '1stFlrSF': 1400,\n"
        "          '2ndFlrSF': 1000, 'TotalBsmtSF': 1400, 'Neighborhood': 'NridgHt'}\n"
        "print(f\"Casa-lusso: ${predict_price(luxury):,.0f}\")\n"
        "\n"
        "# Solo 4 feature: tutto il resto viene imputato\n"
        "minimal = {'OverallQual': 7, 'GrLivArea': 1800,\n"
        "           'Neighborhood': 'CollgCr', 'YearBuilt': 2005}\n"
        "print(f\"Casa-minimal (4 feature): ${predict_price(minimal):,.0f}\")\n"
    ),
    md(
        "## Conclusione\n\n"
        "Il pipeline è ora **production-ready** nel senso che:\n\n"
        "1. È serializzato con `joblib` → re-deployable su qualsiasi macchina con "
        "le stesse versioni delle librerie (vedi `requirements.txt`).\n"
        "2. La funzione `predict_price()` è l'unico punto di ingresso per "
        "l'inferenza, evitando duplicazione di logica.\n"
        "3. Tutte le metriche sono in dollari, comprensibili anche per "
        "stakeholder non tecnici.\n\n"
        "Per dettagli teorici sulle scelte fatte (perché log-target, perché "
        "Ridge vs Lasso, perché XGBoost batte RF, ...) vedi `docs/teoria/`."
    ),
]
write_notebook("04_evaluation_inference.ipynb", nb04)


print("\nTutti i 4 notebook generati in", NB_DIR)
