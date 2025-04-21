#!/bin/bash

DATASETS=("raw" "h37rv" "nr")

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "🔁 Running SVR on $dataset"
  python run_svr_experiment.py --dataset_name $dataset
done

echo ""
echo "✅ All SVR experiments completed."