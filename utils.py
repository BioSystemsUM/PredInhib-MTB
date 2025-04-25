import yaml
import pandas as pd
from pydantic import BaseModel, Field
from typing import Union

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np

from rdkit import Chem
import pandas as pd
import pickle
import numpy as np
import subprocess
import datetime
import os
from glob import glob


def load_chembl_datasets(folder_path):

    with open(f'{folder_path}id_name.yaml', 'r') as f:
        id_name = yaml.safe_load(f)

    dataframes = {}
    for id in id_name:
        dataframes[id] = pd.read_csv(
            f'{folder_path}{id}.csv',
            index_col=None,
            low_memory=False)

    return dataframes, id_name

# creates a new column to merge duplicate canonical_smiles MIC_uM using median
def merge_duplicates(df):
    # Group by canonical_smiles and calculate median, mean, std
    merged_df = df.groupby('canonical_smiles', as_index=False).agg({
        'MIC_uM': ['median', 'mean', 'std']
    })

    # Flatten column names
    merged_df.columns = ['canonical_smiles', 'MIC_uM_median', 'MIC_uM_mean', 'MIC_uM_std']

    # Merge stats back into original DataFrame
    df = pd.merge(df, merged_df, on='canonical_smiles')

    return df

#shuffle data
def add_log_column(df):
    df = df.copy() 
    df["mic_log"] = df["MIC_uM_median"].apply(lambda x: np.nan if x <= 0 else -np.log10(x))
    return df


def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def convert_to_uM(df, column='rdkit_smiles'):
    """
    Converts standard concentration units in the DataFrame to µM.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'standard_units', 'standard_value', and 'canonical_smiles' columns.

    Returns:
        List[float]: Values converted to µM.
    """
    uM_values = []

    for _, row in df.iterrows():
        if row['standard_units'] == 'nM':
            uM_values.append(row['standard_value'] / 1000)

        elif row['standard_units'] == 'ug.mL-1':
            molweight = Descriptors.ExactMolWt(Chem.MolFromSmiles(row[column]))
            uM_value = ((row['standard_value'] / molweight) * 1000)
            uM_values.append(uM_value)

        elif row['standard_units'] == 'uM':
            uM_values.append(row['standard_value'])

        else:
            raise KeyError('Standard Units not recognized')
    return uM_values


import os
import subprocess
import datetime
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Unique folder generator
def unique_dir_name():
    return datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

# ──────────────────────────────────────────────────────────────
# KPGT embedding function (replaces RDKit fingerprinting)
def smiles_to_embeddings(smiles, gpu, kpgt_root, env_python, model_path, config="base"):
    folder = unique_dir_name()
    datasets_dir = Path(kpgt_root) / "datasets"
    dataset_path = datasets_dir / folder
    dataset_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"Class": [0] * len(smiles), "smiles": smiles})
    csv_path = dataset_path / f"{folder}.csv"
    df.to_csv(csv_path, index=False)

    # Change to KPGT script dir
    original_path = Path.cwd()
    script_dir = Path(kpgt_root) / "scripts"
    os.chdir(script_dir)

    try:
        # Run preprocessing
        subprocess.run([
            env_python,
            str(script_dir / "preprocess_downstream_dataset.py"),
            "--data_path", str(datasets_dir),
            "--dataset", folder
        ], check=True)

        print("🧠 Extracting features...")

        # Run feature extraction
        subprocess.run([
            env_python,
            str(script_dir / "extract_features.py"),
            "--config", config,
            "--model_path", str(model_path),
            "--data_path", str(datasets_dir),
            "--gpu", str(gpu),
            "--dataset", folder
        ], check=True)

    finally:
        os.chdir(original_path)

    # Load embeddings
    npz_path = dataset_path / "kpgt_base.npz"
    data = np.load(npz_path)
    fps_array = data["fps"]

    # Cleanup
    import shutil
    shutil.rmtree(dataset_path)

    return fps_array

# ──────────────────────────────────────────────────────────────
# Main embedding computation for a dataset split
def compute_kpgt_embeddings_for_dataset(csv_paths, output_fp_cache_path, gpu, kpgt_root, env_python, model_path):
    all_smiles = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        all_smiles.update(df["smiles"])
    all_smiles = sorted(all_smiles)  # deterministic

    print(f"🧬 Total unique SMILES: {len(all_smiles)}")
    fps_array = smiles_to_embeddings(
        all_smiles,
        gpu=gpu,
        kpgt_root=kpgt_root,
        env_python=env_python,
        model_path=model_path
    )

    smiles_to_fp = {smi: fps_array[i] for i, smi in enumerate(all_smiles)}

    with open(output_fp_cache_path, "wb") as f:
        pickle.dump(smiles_to_fp, f)

    print(f"✅ Embeddings saved to: {output_fp_cache_path}")
