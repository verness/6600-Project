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

### 1. CNN Baseline

The CNN served as our starting point because convolutional networks are a natural fit for image-based inputs. Each heartbeat was converted into a **spectrogram** — a 2D time-frequency representation generated via the Short-Time Fourier Transform (STFT) — where clinically important patterns (QRS energy at 10–40 Hz, P/T waves below 10 Hz) become spatially separable. A secondary metadata branch processes extracted beat-level features (RR intervals, heart rate, amplitude), and both branches are fused before classification.

**Architecture:** 3× Conv2d blocks (32→64→128 filters, kernel sizes 7→5→3) + AdaptiveAvgPool2d → metadata MLP (6→32) → concatenation (160-dim) → dense classifier → 4 classes

**Key design choices:**
- Decreasing kernel size captures broad QRS patterns first, then fine morphological detail
- AdaptiveAvgPool2d produces a fixed-length feature vector regardless of input resolution
- Hyperparameters selected via random search over 15 configurations (LR, batch size, dropout, weight decay)

### 2. Transformer Baseline

The Transformer operates directly on **raw waveform sequences** (250 samples), bypassing the spectrogram conversion entirely. Self-attention enables every time sample to attend to every other, allowing the model to learn long-range dependencies — such as the relationship between P-wave morphology and QRS timing — that local convolutional kernels cannot easily capture.

**Architecture:** Positional encoding → multi-head self-attention encoder (4 layers, 8 heads) → [CLS] token classification head → 4 classes

**Key design choices:**
- Raw waveform input avoids STFT computational overhead
- Attention maps provide interpretability (which beat regions drive classification)
- Z-score normalization removes patient-level baseline drift

### 3. Hybrid CNN-Transformer

The hybrid model combines the CNN's spatial feature extraction with the Transformer's global reasoning. An **EfficientNet-B0** backbone (pretrained on ImageNet, first 4 stages frozen) extracts spatial tokens from spectrograms, which are then processed by a Transformer encoder alongside a metadata token and a learnable [CLS] token.

**Architecture:** EfficientNet-B0 backbone → patch token projection (256-dim) + metadata MLP token + [CLS] token → Transformer encoder (4 layers, 8 heads) → classification head → 4 classes

**Key design choices:**
- Pretrained backbone provides transferable low-level features (edges, textures) from natural images
- Differential learning rates (lower for CNN, higher for Transformer)
- Focal Loss (γ=2.0) to focus training on hard-to-classify minority beats
- Linear warmup + cosine decay learning rate schedule


---

## Evaluation

* Accuracy
* F1-score / Marco F1-score (important due to class imbalance)
* Confusion Matrix

---

## 🔍 Research Question

When does a Transformer outperform a CNN in ECG arrhythmia classification?

---

## References

- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- Khan, F., Yu, X., Yuan, Z., & Rehman, A. U. (2023). ECG classification using 1-D convolutional deep residual neural network. *PLoS ONE*, 18(4), e0284791.
- Mukhoti, J., Kulharia, V., Sanyal, A., Golodetz, S., Torr, P. H. S., & Dokania, P. K. (2020). Calibrating deep neural networks using focal loss. *NeurIPS 2020*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT 2019*.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML 2019*.


## Team

* Laurent Julia Calac
* Matthew Hakim
* Yi-Ting Chin
* Mohammad Yassin
