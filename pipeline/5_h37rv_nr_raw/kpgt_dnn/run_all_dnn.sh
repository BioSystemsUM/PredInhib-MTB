#!/bin/bash

DATASETS=("h37rv" "raw" "nr")

for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "🔁 Running DNN on $dataset"
  python run_dnn_experiment.py --dataset_name $dataset
done

echo ""
echo "✅ All DNN experiments completed."