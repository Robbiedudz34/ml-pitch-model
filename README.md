# ML Pitch Model

This repository contains the **process and pipeline code** I developed for machine-learning based **pitch type classification** using Statcast and Trackman data.

**Note:** The trained models (.txt) and encoder files (.pkl) are **not included** in this repository. The features lists and cleaned files were engineered in other scripts which are also **not included** in this repository. Included are the pipeline code, cleaned 2024 and 2025 statcast files, test files for the different models, and confusion matrices for the different stages of the pipeline flow.

---

## Overview

This project builds a multi-stage classification pipeline using **LightGBM** to identify pitch typees from baseball tracking data. Statcast and Trackman have different naming and labeling conventions for the pitch types, so I am strictly considering the 8 most prevalent pitch types:

**Fastball | Sinker | Cutter | Slider | Curveball | Sweeper | Changeup | Splitter**

The system is structured hierarchially:

**1. Broad Grouping:** Separates pitches into 3 categories - **Fastball & Sinker | Changeup & Splitter | Cutter, Slider, Curveball, & Sweeper**

**2. Specialized Models:**
- v3a: Changeup vs. Splitter
- v3b: Fastball vs. Sinker
- v3c: All Breaking Pitches, subdivided between **Hard** and **Slow** types
- v3c1: Hard - Cutter vs. Slider
- v3c2: Slow - Curveball vs. Sweeper

**3. Unified Report:** Predictions are re-combined into a complete 8 pitch type classification.

This hierarchial design improved accuracy to an average test of **94%**, reducing error relative to the best single multiclass model by approximately **50%**.

---

## Repository Structure

### Core Script
- `v3_pipeline.py` — Main pipeline script (loads encoders, runs evaluation)

### Example Datasets
- `statcast_2024_clean.parquet` — Example preprocessed Statcast dataset
- `statcast_2025_clean.parquet` — Example preprocessed Statcast dataset

### Test Feature Files

_These parquet files serve as demo test sets so the pipeline can be run without access to the full private training corpus_

- `test_features_v3_groups.parquet` — Broad group model
- `test_features_v3a.parquet` — Changeup vs Splitter
- `test_features_v3b.parquet` — Fastball vs Sinker
- `test_features_v3c.parquet` — Breaking (hard vs slow)
- `test_features_v3c1.parquet` — Cutter vs Slider
- `test_features_v3c2.parquet` — Curveball vs Sweeper

### Figures (Confusion Matrices)
- `cm_v3.png` — Full 8-class model
- `cm_v3_groups.png` — Broad groups
- `cm_v3a.png` — Changeup vs Splitter
- `cm_v3b.png` — Fastball vs Sinker
- `cm_v3c.png` — Breaking (hard vs slow)
- `cm_v3c1.png` — Cutter vs Slider
- `cm_v3c2.png` — Curveball vs Sweeper

### Repo Management
- `.gitignore` — Ensures large model/encoder files are not tracked  
- `README.md` — Project documentation

## What is not included, but required for full functionality to run the pipeline or create the models used:

This repository is focused on the **pipeline process and evaluation framework**.  
Several supporting files and artifacts are **not included**, either because they are too large, private, or specific to my local training environment:

### Feature Engineering Scripts
- `build_training_features.py` – prepares encoded training/testing sets from raw data  
- `new_columns.py` – constructs advanced physics features (spin ratios, deltas, movement metrics)  
- `create_parquet_files.py` – converts raw Statcast/Trackman CSVs into processed parquet files  

### Training Data
- Concatenated multi-season Statcast training datasets (2016–2023)  
- Cleaned season datasets used for model fitting  

### Model Training & Optimization
- Grid search script for hyperparameter tuning with LightGBM  
- Prior experimental training scripts and intermediate test versions  

### Private Model Artifacts
- Trained model weights (`.txt` files)  
- Encoders and pipeline objects (`.pkl` files)  

### Why these are excluded
- Large file sizes (GitHub’s 100 MB file size limit)  
- Proprietary artifacts from model training  
- To emphasize **process over product**: this repo demonstrates the *pipeline design*, not a ready-to-use pretrained model

---

## Features

- Handles both **Statcast** and **Trackman** data, for both inconsistencies and data processing needs
- Robust feature engineering on relevant pitching metrics
- Custom LabelEncoders for categorical variables
- Full evaluation reports _(precision, recall, f1, support)_ per stage
- Confusion matrix outputs for deeper inspection
- Support for processing small test files

---

## Example Output

After processing a test dataset, the pipeline produces

- Broad group classification report
- Sub-model reports for the dissection of the broader group of data
- Final unified evaluation across all 8 pitch types
- Normalized confusion matrix plot

<img width="1000" height="800" alt="cm_v3" src="https://github.com/user-attachments/assets/60735bd1-4eb0-4bab-ab4c-9f6d27f827e8" />

---

## Disclaimer

This repository displays the process. The actual model weights, training construction, encoders, and necessary pieces to the pipeline are not included. Results in the small scale will depend on data availablity and preprocessing consistency. A necessary step for small scale interpretation is including the pitch types associated with a pitcher and a strong base of known pitch types to train on for that pitcher.

This is a research and analytics tool, not an official MLB or Statcast product. See LICENSE for details or inquiry.

---

**Author - Robbie Dudzinski | Sports Analytics and Data Science Consultant | Former Professional Pitcher**
