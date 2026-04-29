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

The data pipeline consists of two stages:

1. Extraction & Cleaning:
* Loads ECG signals and annotations using `wfdb`
* Extracts heartbeat windows (250 samples per beat)
* Removes non-beat annotations (e.g., '+')
* Maps raw labels into 4 classes:
  * N: Normal
  * S: Supraventricular
  * V: Ventricular
  * Q: Other

To run preprocessing:

```bash
python src/data/preprocess.py
```

This generates:

* `X.npy` → heartbeat signals
* `y.npy` → labels

2. Stratified Splitting:
* Splits the cleaned data into Training (70%), Validation (15%), and Testing (15%).
* Uses stratification to ensure class proportions are preserved across all sets, which is critical for handling the dataset's class imbalance.

To run the split.py file:

```bash
python src/data/split.py
```

This pipeline generates the following files in Data/processed/:
* X_train.npy, y_train.npy
* X_val.npy, y_val.npy
* X_test.npy, y_test.npy


---

## Project Structure

```
.
├── checkpoints/                            # Saved model checkpoints and training logs
│   ├── best_model.pt
│   ├── last_model.pt
│   └── training_log.csv
│
├── checkpoints(colab)/                     # Checkpoints generated from Google Colab runs
│
├── Data/                                   # Raw and processed ECG datasets
│   ├── mit-bih-arrhythmia-database-1.0.0/  # Original MIT-BIH dataset
│   └── processed/                          # Cleaned / transformed data files
│
├── models/                                 # Trained models, reports, and visual outputs
│   ├── cnn_baseline.pt
│   ├── transformer_baseline.pt
│   ├── cnn_baseline_confusion.png
│   ├── transformer_baseline_confusion.png
│   ├── transformer_attention.png
│   ├── tuning_results.csv
│   ├── transformer_tuning_results.csv
│   ├── tuning_heatmap.png
│   ├── transformer_tuning_heatmap.png
│   ├── smote_before_after.png
│   └── classification_report_table.png
│
├── notebooks/                              # EDA, experimentation, and model training notebooks
│   ├── 01_inspect_data.ipynb
│   ├── cnn_model.ipynb
│   ├── eda.ipynb
│   ├── hybrid_cnn_transformer.ipynb
│   ├── hybrid_train.ipynb
│   ├── hybrid_training_colab.ipynb
│   ├── smote.ipynb
│   └── transformer_model.ipynb
│
├── src/
│   └── data/                               # Data preprocessing pipeline scripts
│       ├── preprocess.py
│       ├── smote.py
│       └── split.py
│
├── Archive.zip                             # Archived project files
├── pipeline.html                           # Exported project pipeline visualization
├── .gitignore
└── README.md
```

---

## Models

We will implement:

* CNN baseline
- CNN baseline model served as the starting point for classification because convolutional neural networks are a natural fit for image-based inputs. Instead of using raw ECG waveforms directly, each heartbeat was transformed into a spectrogram, a 2D time-frequency representation that captures how signal frequencies change over time. This allows the ECG beat to be treated like an image, where important cardiac patterns become visually distinguishable.

* Transformer baseline
* Hybrid CNN + Transformer model

---

## Evaluation

* Accuracy
* F1-score / Marco F1-score (important due to class imbalance)
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
