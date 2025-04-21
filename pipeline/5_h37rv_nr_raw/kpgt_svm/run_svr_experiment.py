import os
import argparse
import pandas as pd
import numpy as np
import pickle
from glob import glob
from sklearn.svm import SVR
from time import time
from tqdm import tqdm

# Paths
BASE_DIR = "/home/malves/predinhib_mtb"
FOLD_ROOT_BASE = os.path.join(BASE_DIR, "data/cv/raw_h37rv_nr/folds")
FP_CACHE_TEMPLATE = os.path.join(FOLD_ROOT_BASE, "{dataset}/kpgt_embeddings_cache.pkl")  # stores KPGT embeddings
OUTPUT_BASE = os.path.join(BASE_DIR, "results/svr_preds/{dataset}")  # new output path for SVR

# Load KPGT embedding cache: {smiles: vector}
def load_kpgt_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# Convert list of SMILES to embedding matrix
def smiles_to_embeddings(smiles_list, embedding_cache):
    missing = [s for s in smiles_list if s not in embedding_cache]
    if missing:
        raise ValueError(f"{len(missing)} SMILES missing from embedding cache")
    return np.array([embedding_cache[s] for s in smiles_list])

# Main experiment
def run_svr_experiment(dataset, n_repeats=5, n_folds=5):
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

            # Load test and training folds
            test_path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{fold}.csv")
            test_df = pd.read_csv(test_path)

            train_dfs = []
            for f_ in range(n_folds):
                if f_ == fold:
                    continue
                path = os.path.join(fold_root, f"{dataset}_rep{rep}_fold{f_}.csv")
                train_dfs.append(pd.read_csv(path))
            train_df = pd.concat(train_dfs).reset_index(drop=True)

            X_train = smiles_to_embeddings(train_df["smiles"], embedding_cache)
            y_train = train_df["label"].values
            X_test = smiles_to_embeddings(test_df["smiles"], embedding_cache)

            # Train SVR (use RBF kernel)
            model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Save predictions
            out_df = test_df[["smiles", "label"]].copy()
            out_df["pred"] = y_pred
            out_df["rep"] = rep
            out_df["fold"] = fold
            out_df.to_csv(out_path, index=False)

            elapsed = time() - start
            tqdm.write(f"✅ rep{rep}-fold{fold} done in {elapsed:.1f}s → {out_path}")

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["raw", "h37rv", "nr"])
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    run_svr_experiment(args.dataset_name, args.n_repeats, args.n_folds)

# # Example usage:
# python run_svr_experiment.py --dataset_name raw