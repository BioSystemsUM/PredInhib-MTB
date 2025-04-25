# PREDINHIB_MTB

**PREDINHIB_MTB** is a machine learning pipeline designed to predict compound inhibition and MIC values against *Mycobacterium tuberculosis* (M. tuberculosis), leveraging curated assay descriptions using LLMs.

## 🔍 Overview

This repository supports two key tasks:
1. **MIC regression** using curated, biologically meaningful datasets.
2. **Multi-task classification** of compound activity across varying resistance profiles (Non-Resistant, Resistant, MDR).

We utilize Large Language Models (LLMs) to extract structured metadata from free-text assay descriptions in ChEMBL, enabling biologically consistent dataset construction and improved model performance.

## 📁 Repository Structure

```
.
├── data/
│   └── chemblapi/                          # ChEBML data downloaded for Mycobacterium tuberculosis
│   └── cv/                                 # Data required for repeated 5x5 CV (including folds)
│   └── llama/                              # Input and Output data of Llama processing
│   └── res_mdr_xdr/                        # Data for multi-task classification (NR, R, MDR)
|
├── figures                                 # Figures and plots generated during analysis
|
├── models/                                 # Trained model checkpoints and configs
|
├── pipeline/                          
│   └── 0_chembl_download.ipynb             # Data retrieval from ChEMBL
│   └── 1_data_exploration.ipynb            # Dataset inspection
│   └── 2_data_cleaning.ipynb               # Dataset filtering
│   └── 3_llama_processing/                 
│       └── 3_llama_processing              # Python Script to get metadata from descriptions using Llama3.3
│   └── 4_data_splitting/              
│       └── 4_1_llama_results_exploration.ipynb           # Exploration of LLama metadata extraction
│       └── 4_2_ml_preparation.ipynb        # preprocessing, and datasets creation for single-task
│   └── 5_h37rv_nr_raw/
│       └── 5_0_folds_creation.ipynb        # Stratified fold generation for cross-validation
│       └── fingerprints.ipynb              # Morgan fingerprints generation
│       └── kpgt_embeddings.ipynb           # KPGT embeddings generation
│       └── 5_2_kpgt_embeddings.ipynb       # KPGT embeddings generation
│       └── fp_RF/run_all_rf.sh             # Random Forest training and evaluation
│       └── fp_SVR/run_all_svr.sh           # SVR training and evaluation
│       └── fp_DNN/run_all_dnn.sh           # DNN training and evaluation
|
│   └── 6_multitask/
│       └── data_curation.ipynb             # Data curation for multi-task classification                     
│       └── model_run.ipynb                 # Multi-task model training and evaluation 
│       └── multitask_test_set.csv          # Test set for multi-task classification
├── results/                           # Model outputs from repeated 5x5 CV
├── utils.py                           # Utility functions
└── predinhib_env.yml                  # Conda environment specification
```

## 🧪 How to Reproduce

### 1. Setup

```bash
conda env create -f predinhib_env.yml
conda activate predinhib
```

### 2. Data Preparation

Follow these notebooks in order to get the most recent data (not applicable to reproducibility):
1. `0_chembl_download.ipynb`
2. `1_data_exploration.ipynb`
3. `2_data_cleaning.ipynb`

### 3. Assay Metadata Extraction

Use `3_llama_processing/3_llama_processing.py` to extract structured metadata (e.g., strain, resistance, mutant type) using LLaMA3.3 via LangChain and Ollama.

We recomment using the docker/podman version: 
```bash
docker pull ollama/ollama
```

### 4. Llama Results Exploration

Use `4_data_splitting/4_1_llama_results_exploration.ipynb` to explore the metadata extracted by LLaMA.

Use `4_data_splitting/4_2_ml_preparation.ipynb` to prepare the data for ML tasks.

### 5. Model Training for single-task and Evaluation
Stratified fold generation for cross-validation:

`5_0_folds_creation.ipynb`

Generate fingerprints and KPGT embeddings using: 

`5_1_fingerprints.ipynb` and `5_2_kpgt_embeddings.ipynb`, respectively.

(for KPGT embeddings, please clone https://github.com/lihan97/KPGT.git ans intall its environment)

Use `run_all_*model*.sh` scripts to train and evaluate models. The scripts are designed to be run in a bash shell.

- MIC prediction (regression)

### 6. Model Training for multi-task and Evaluation

Use `6_/multitask/model_run.ipynb` to train and evaluate models. .

- Resistance profile classification (multi-task)

## 📊 Benchmarks and Models

We evaluate:
- **Random Forest (Morgan fingerprints)**
- **SVR + KPGT embeddings**
- **DNN + KPGT embeddings (best performer)**

Tasks:
- MIC regression (R² up to 0.65, MAE 0.41)
- Multi-task binary classification (Non-Resistant, Resistant, MDR)

Post-hoc classification threshold: **10 µM**

## 📬 Contact

For questions, contact **Nuno Alves** via GitHub or email: `id10075@alunos.uminho.pt`.