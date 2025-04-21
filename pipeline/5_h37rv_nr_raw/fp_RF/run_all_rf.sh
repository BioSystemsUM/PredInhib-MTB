#!/bin/bash

DATASETS=("raw" "h37rv" "nr")

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "🔁 Running Random Forest on $dataset"
  python run_rf_experiment.py --dataset_name $dataset
done

echo ""
echo "✅ All RF experiments completed."