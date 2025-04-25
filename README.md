# PREDINHIB_MTB

**PREDINHIB_MTB** is a machine learning pipeline for predicting compound inhibition and MIC values against *Mycobacterium tuberculosis* (M. tuberculosis), leveraging curated assay descriptions processed through Large Language Models (LLMs).

## 🔍 Overview

This repository supports two key tasks:
1. **MIC Regression** using curated, biologically meaningful datasets.
2. **Multi-task Classification** of compound activity across different resistance profiles (Non-Resistant, Resistant, MDR).

We use LLMs to extract structured metadata from free-text assay descriptions (e.g., from ChEMBL), enabling biologically consistent dataset construction and improved model performance.

## 📁 Repository Structure

```
.
├── data/
│   ├── chemblapi/           # ChEMBL data for Mycobacterium tuberculosis
│   ├── cv/                  # Repeated 5x5 cross-validation folds
│   ├── llama/               # Llama3.3 input/output
│   └── res_mdr_xdr/         # Data for multi-task classification
│
├── figures/                 # Generated plots and figures
│
├── models/                  # Trained models and configuration files
│
├── pipeline/
│   ├── 0_chembl_download.ipynb         # Download data from ChEMBL
│   ├── 1_data_exploration.ipynb        # Initial dataset inspection
│   ├── 2_data_cleaning.ipynb           # Filtering and cleaning
│   ├── 3_llama_processing/
│   │   └── 3_llama_processing.py       # Llama3.3 metadata extraction
│   ├── 4_data_splitting/
│   │   ├── 4_1_llama_results_exploration.ipynb  # Metadata exploration
│   │   └── 4_2_ml_preparation.ipynb             # ML dataset preparation
│   ├── 5_h37rv_nr_raw/
│   │   ├── 5_0_folds_creation.ipynb             # Stratified fold generation
│   │   ├── fingerprints.ipynb                   # Morgan fingerprint generation
│   │   ├── kpgt_embeddings.ipynb                # KPGT embedding generation
│   │   ├── 5_2_kpgt_embeddings.ipynb             # Alternative KPGT generation
│   │   ├── fp_RF/run_all_rf.sh                   # Random Forest training
│   │   ├── fp_SVR/run_all_svr.sh                  # SVR training
│   │   └── fp_DNN/run_all_dnn.sh                  # DNN training
│   └── 6_multitask/
│       ├── data_curation.ipynb                  # Multi-task dataset curation
│       ├── model_run.ipynb                      # Multi-task training/evaluation
│       └── multitask_test_set.csv               # Test set for classification
│
├── results/                    # Model predictions and CV results
├── utils.py                     # Utility functions
└── predinhib_env.yml            # Conda environment file
```

## 🧪 How to Reproduce

### 1. Environment Setup

```bash
conda env create -f predinhib_env.yml
conda activate predinhib
```

### 2. Data Preparation

(Optional) Download and process the latest ChEMBL data:
1. `0_chembl_download.ipynb`
2. `1_data_exploration.ipynb`
3. `2_data_cleaning.ipynb`

*Note: This step is not needed for pure reproducibility of current results.*

### 3. Assay Metadata Extraction (LLaMA)

Run `3_llama_processing/3_llama_processing.py` to extract structured metadata (strain, resistance, mutant type) from assay descriptions.

We recommend running Ollama inside a container for GPU acceleration:

```bash
podman pull ollama
podman run -d --name ollama_cuda --privileged --gpus all -v ollama_data:/root/.ollama -p 11434:11434 docker.io/ollama/ollama
podman exec -it ollama bash
ollama pull llama3.3
exit
```

### 4. Metadata Exploration and Dataset Preparation

- Explore extracted metadata: `4_data_splitting/4_1_llama_results_exploration.ipynb`
- Prepare datasets for ML: `4_data_splitting/4_2_ml_preparation.ipynb`

### 5. Single-task Model Training and Evaluation

- Generate folds: `5_h37rv_nr_raw/5_0_folds_creation.ipynb`
- Generate fingerprints: `5_h37rv_nr_raw/fingerprints.ipynb`
- Generate KPGT embeddings: `5_h37rv_nr_raw/kpgt_embeddings.ipynb` or `5_2_kpgt_embeddings.ipynb`

**Note on KPGT embeddings**:  
Clone [KPGT](https://github.com/lihan97/KPGT.git), install its environment, and download its pre-trained model from [figshare](https://figshare.com/s/d488f30c23946cf6898f?file=35369662).  
Update the paths to the KPGT repository, model, and environment as needed.

- Train and evaluate models using the bash scripts:
  - `run_all_rf.sh` (Random Forest)
  - `run_all_svr.sh` (SVR)
  - `run_all_dnn.sh` (DNN)

### 6. Multi-task Model Training and Evaluation

Train and evaluate multi-task models using:
- `6_multitask/model_run.ipynb`

This focuses on classifying compounds into Non-Resistant, Resistant, and MDR categories.

## 📊 Models and Benchmarks

We benchmark:
- **Random Forest** (Morgan fingerprints)
- **SVR** (KPGT embeddings)
- **DNN** (KPGT embeddings, best performance)

Task performance:
- MIC regression: **R² up to 0.65**, **MAE ≈ 0.41**
- Resistance classification (multi-task): Non-Resistant / Resistant / MDR **F1 scores of 0.84, 0.81, and 0.67, respectively**.
    Classification threshold: **10 µM**

## 📬 Contact

For questions or feedback, contact **Nuno Alves** via GitHub or email: `id10075@alunos.uminho.pt`.