#!/usr/bin/env bash
# Pipeline completa: training full + esecuzione notebook + verifica.
# Uso: bash scripts/run_full.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d "venv" ]]; then
  echo "[setup] creo venv..."
  python3 -m venv venv
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install --upgrade pip --quiet
  pip install -e ".[notebooks]" --quiet
else
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[1/3] Training pipeline completa (~5-10 min)..."
ames-train

echo "[2/3] Rigenerazione + esecuzione notebook..."
python scripts/build_notebooks.py
for nb in notebooks/01_eda.ipynb \
          notebooks/02_preprocessing_features.ipynb \
          notebooks/03_modeling_tuning.ipynb \
          notebooks/04_evaluation_inference.ipynb; do
  echo "    → $nb"
  jupyter nbconvert --to notebook --execute --inplace \
      "$nb" --ExecutePreprocessor.timeout=1800 \
      --log-level=ERROR
done

echo "[3/3] Validazione: smoke-test predict_price()..."
python -c "
from ames_pipeline.inference import predict_price, example_input
p = predict_price(example_input())
print(f'  Prezzo casa-base: \${p:,.0f}')
assert 100_000 < p < 300_000, f'Predizione fuori range: \${p:,.0f}'
print('  [OK] inference smoke test passed')
"

echo
echo "Pipeline completata. Vedi reports/ per i risultati."
