# 🫀 Hybrid CNN-Transformer for ECG Arrhythmia Detection

## Overview

This project develops a deep learning system to classify ECG heartbeats using a combination of Convolutional Neural Networks (CNNs) and Transformer architectures.

We aim to compare:

* CNN baseline (local pattern learning)
* Transformer baseline (global sequence modeling)
* Hybrid CNN-Transformer model

The goal is to understand when and why Transformers outperform CNNs for ECG signal classification.

---

## Dataset

We use the MIT-BIH Arrhythmia Database, which contains:

* Raw ECG signals
* Annotated heartbeat locations (R-peaks)
* Beat-level labels

Each heartbeat is extracted into a fixed-length window centered around the R-peak.

---

## Preprocessing Pipeline

The preprocessing script:

* Loads ECG signals and annotations using `wfdb`
* Extracts heartbeat windows (250 samples per beat)
* Removes non-beat annotations (e.g., '+')
* Maps raw labels into 4 classes:

  * N: Normal
  * S: Supraventricular
  * V: Ventricular
  * Q: Other
* Handles class imbalance

To run preprocessing:

```bash
python src/data/preprocess.py
```

This generates:

* `X.npy` → heartbeat signals
* `y.npy` → labels

---

## Project Structure

```
.
├── Data/
│   ├── mit-bih-arrhythmia-database-1.0.0/  # Raw MIT-BIH dataset (not tracked)
│   ├── processed/                          # Cleaned X.npy and y.npy (not tracked)
├── notebooks/                              # EDA and noise inspection
│   ├── 01_inspect_data.ipynb
│   └── eda.ipynb
├── src/
│   ├── data/                               # Data loading and preprocessing
│   │   └── preprocess.py
│   ├── models/                             # Model architectures
│   ├── training/                           # Training loops and scripts
│   └── evaluation/                         # Performance metrics
├── .gitignore                              # Prevents large data/junk from being pushed
└── README.md                               # Project documentation
```

---

## Models

We will implement:

* CNN baseline
* Transformer baseline
* Hybrid CNN + Transformer model

---

## Evaluation

* Accuracy
* F1-score (important due to class imbalance)
* Confusion Matrix

---

## 🔍 Research Question

When does a Transformer outperform a CNN in ECG arrhythmia classification?

---


## Team

* Laurent Julia Calac
* Matthew Hakim
* Yi-Ting Chin
* Mohammad Yassin
