import os
import argparse
import pandas as pd
import numpy as np
import pickle
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from time import time
from tqdm import tqdm

BASE_DIR = "/home/malves/predinhib_mtb"
FOLD_ROOT_BASE = os.path.join(BASE_DIR, "data/cv/raw_h37rv_nr/folds")
FP_CACHE_TEMPLATE = os.path.join(FOLD_ROOT_BASE, "{dataset}/fingerprint_cache.pkl")
OUTPUT_BASE = os.path.join(BASE_DIR, "results/rf_preds/{dataset}")

def load_fingerprint_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def smiles_to_fp(smiles_list, fp_cache):
    return np.array([fp_cache[s] for s in smiles_list if s in fp_cache])

def run_rf_experiment(dataset, n_repeats=5, n_folds=5):
    fold_root = os.path.join(FOLD_ROOT_BASE, dataset)
    fp_cache_path = FP_CACHE_TEMPLATE.format(dataset=dataset)
    output_dir = OUTPUT_BASE.format(dataset=dataset)

    print(f"📂 Dataset: {dataset}")
    print(f"📁 Fold dir: {fold_root}")
    print(f"🔑 Fingerprint cache: {fp_cache_path}")
    print(f"📦 Output dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    fp_cache = load_fingerprint_cache(fp_cache_path)

    for rep in tqdm(range(n_repeats), desc="🔁 Repeats"):
        for fold in range(n_folds):
            out_path = os.path.join(output_dir, f"{dataset}_rep{rep}_fold{fold}_preds.csv")
            if os.path.exists(out_path):
                tqdm.write(f"⏩ Skipping rep {rep}, fold {fold} (already exists)")
                continue

            start = time()
            tqdm.write(f"🚀 rep{rep}, fold{fold}")

            test_path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{fold}.csv")
            test_df = pd.read_csv(test_path)

            # Training data
            train_dfs = []
            for f_ in range(n_folds):
                if f_ == fold: continue
                path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{f_}.csv")
                train_dfs.append(pd.read_csv(path))
            train_df = pd.concat(train_dfs).reset_index(drop=True)

            X_train = smiles_to_fp(train_df["smiles"], fp_cache)
            y_train = train_df["label"].values
            X_test = smiles_to_fp(test_df["smiles"], fp_cache)

            model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            out_df = test_df[["smiles", "label"]].copy()
            out_df["pred"] = y_pred
            out_df["rep"] = rep
            out_df["fold"] = fold
            out_df.to_csv(out_path, index=False)

            elapsed = time() - start
            tqdm.write(f"✅ rep{rep}-fold{fold} done in {elapsed:.1f}s → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["raw", "h37rv", "nr"])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    run_rf_experiment(args.dataset_name, args.n_repeats, args.n_folds)

# # Example usage:
# python run_rf_experiment.py --dataset_name raw