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



# ──────────────────────────────────────────────────────────────
# Unique folder generator for temporary dataset directories
def unique_dir_name():
    now = datetime.datetime.now()
    return str(now.strftime("%d-%m-%Y_%H-%M-%S"))

# ──────────────────────────────────────────────────────────────
# KPGT embedding function (replaces RDKit fingerprinting)
def smiles_to_embeddings(smiles, gpu):
    folder = unique_dir_name()
    dataset_path = f'/home/malves/predator/KPGT/datasets/{folder}/'
    os.makedirs(dataset_path)

    df = pd.DataFrame({'Class': [0]*len(smiles), 'smiles': smiles})
    df.to_csv(f'{dataset_path}{folder}.csv', index=False)

    original_path = os.getcwd()
    script_path = '/home/malves/predator/KPGT/scripts/preprocess_downstream_dataset.py'
    os.chdir(os.path.dirname(script_path))

    try:
        subprocess.run([
            '/home/malves/miniconda3/envs/KPGT/bin/python', script_path,
            '--data_path', '/home/malves/predator/KPGT/datasets',
            '--dataset', folder
        ])
        print('🧠 Extracting features...')
        subprocess.run([
            '/home/malves/miniconda3/envs/KPGT/bin/python',
            '/home/malves/predator/KPGT/scripts/extract_features.py',
            '--config', 'base',
            '--model_path', '/home/malves/predator/KPGT/models/pretrained/base/base.pth',
            '--data_path', '/home/malves/predator/KPGT/datasets/',
            '--gpu', str(gpu),
            '--dataset', folder
        ])
    finally:
        os.chdir(original_path)

    data = np.load(f'{dataset_path}/kpgt_base.npz')
    fps_array = data['fps']

    os.system(f'rm -r {dataset_path}')
    return fps_array

# ──────────────────────────────────────────────────────────────
# Main embedding computation for a dataset split
def compute_kpgt_embeddings_for_dataset(csv_paths, output_fp_cache_path, gpu=6):
    all_smiles = set()
    for path in csv_paths:
        df = pd.read_csv(path)
        all_smiles.update(df["smiles"])
    all_smiles = sorted(list(all_smiles))  # order for stable mapping

    print(f"🧬 Total unique SMILES: {len(all_smiles)}")
    fps_array = smiles_to_embeddings(all_smiles, gpu=gpu)
    smiles_to_fp = {smi: fps_array[i] for i, smi in enumerate(all_smiles)}

    with open(output_fp_cache_path, "wb") as f:
        pickle.dump(smiles_to_fp, f)

    print(f"✅ Embeddings saved to: {output_fp_cache_path}")