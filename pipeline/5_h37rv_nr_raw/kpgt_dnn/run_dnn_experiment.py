import os
import argparse
import pandas as pd
import numpy as np
import pickle
from glob import glob
from time import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from train_deep_model import train_and_predict  
import torch
import gc


# Paths
BASE_DIR = "/home/malves/predinhib_mtb"
FOLD_ROOT_BASE = os.path.join(BASE_DIR, "data/cv/raw_h37rv_nr/folds")
FP_CACHE_TEMPLATE = os.path.join(FOLD_ROOT_BASE, "{dataset}/kpgt_embeddings_cache.pkl")
OUTPUT_BASE = os.path.join(BASE_DIR, "results/dnn_preds/{dataset}")


# Load KPGT embedding cache
def load_kpgt_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Convert SMILES to embeddings
def smiles_to_embeddings(smiles_list, embedding_cache):
    missing = [s for s in smiles_list if s not in embedding_cache]
    if missing:
        raise ValueError(f"{len(missing)} SMILES missing from embedding cache")
    return np.array([embedding_cache[s] for s in smiles_list])

# Main CV loop
def run_dl_experiment(dataset, n_repeats=5, n_folds=5, gpu_id=6):
    fold_root = os.path.join(FOLD_ROOT_BASE, dataset)
    embedding_cache_path = FP_CACHE_TEMPLATE.format(dataset=dataset)
    output_dir = OUTPUT_BASE.format(dataset=dataset)

    print(f"📂 Dataset: {dataset}")
    print(f"📁 Fold dir: {fold_root}")
    print(f"🔑 Embedding cache: {embedding_cache_path}")
    print(f"📦 Output dir: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    embedding_cache = load_kpgt_cache(embedding_cache_path)

    for rep in tqdm(range(n_repeats), desc="🔁 Repeats"):
        for fold in range(n_folds):
            out_path = os.path.join(output_dir, f"{dataset}_rep{rep}_fold{fold}_preds.csv")
            if os.path.exists(out_path):
                tqdm.write(f"⏩ Skipping rep {rep}, fold {fold} (already exists)")
                continue

            start = time()
            tqdm.write(f"🚀 rep{rep}, fold{fold}")

            # Load data
            test_path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{fold}.csv")
            test_df = pd.read_csv(test_path)

            train_dfs = []
            for f_ in range(n_folds):
                if f_ == fold:
                    continue
                path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{f_}.csv")
                train_dfs.append(pd.read_csv(path))
            train_df = pd.concat(train_dfs).reset_index(drop=True)

            # Embedding + split
            X = smiles_to_embeddings(train_df["smiles"], embedding_cache)
            y = train_df["label"].values
            
            print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
            
            try:
                y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
                if len(np.unique(y_bins)) < 2:
                    raise ValueError("Not enough bins for stratification")
                stratify = y_bins
            except Exception as e:
                print(f"⚠️ Stratified split skipped for rep{rep}-fold{fold}: {e}")
                stratify = None

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify
            )
            
            print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
            print(f"Shape of X_val: {X_val.shape}, Shape of y_val: {y_val.shape}")
            

            X_test = smiles_to_embeddings(test_df["smiles"], embedding_cache)

            # Train and predict
            y_pred = train_and_predict(X_train, y_train, X_val, y_val, X_test, gpu=gpu_id)

            # Save results
            out_df = test_df[["smiles", "label"]].copy()
            out_df["pred"] = y_pred
            out_df["rep"] = rep
            out_df["fold"] = fold
            out_df.to_csv(out_path, index=False)

            elapsed = time() - start
            tqdm.write(f"✅ rep{rep}-fold{fold} done in {elapsed:.1f}s → {out_path}")

            # 🧹 Clean up memory (PyTorch + Python)
            del X_train, y_train, X_val, y_val, X_test, y_pred, train_df, test_df, out_df, train_dfs, y_bins
            torch.cuda.empty_cache()
            gc.collect()
        
# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["raw", "h37rv", "nr"])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=6)
    args = parser.parse_args()

    run_dl_experiment(args.dataset_name, args.n_repeats, args.n_folds, gpu_id=args.gpu)
