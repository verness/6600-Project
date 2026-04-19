"""
ECG PREPROCESSING PIPELINE: MIT-BIH Arrhythmia Database
------------------------------------------------------
This script extracts individual heartbeats from raw MIT-BIH ECG records, 
performs label cleaning/mapping, and saves the result as NumPy arrays.

Key Specifications:
- Signal Extraction: Uses Lead II (first channel) from .dat records.
- Beat Windowing: Extracts 250 samples per beat (100 before R-peak, 150 after).
- Class Mapping:
    - Class 0: Normal/Bundle Branch Block (N, L, R)
    - Class 1: Supraventricular/Atrial (A, a, J)
    - Class 2: Ventricular (V, E)
    - Class 3: Fusion (F)
- Output: Saves X.npy (features) and y.npy (labels) to Data/processed/.

Requirements: wfdb, numpy
"""

import wfdb
import numpy as np
import os
from collections import Counter

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../Data/mit-bih-arrhythmia-database-1.0.0/"))

print("Resolved DATA_PATH:", DATA_PATH)
print("Exists:", os.path.exists(DATA_PATH))


# --- EXTRACT BEATS FROM ONE RECORD ---
def extract_beats(record_id):
    record_path = os.path.join(DATA_PATH, record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    signal = record.p_signal[:, 0]  # use 1 channel

    window_before = 100
    window_after = 150

    beats = []
    labels = []

    for i, r_peak in enumerate(annotation.sample):
        label = annotation.symbol[i]

        # skip non-beat labels
        if label == '+':
            continue

        if r_peak - window_before < 0 or r_peak + window_after >= len(signal):
            continue

        beat = signal[r_peak - window_before : r_peak + window_after]

        beats.append(beat)
        labels.append(label)

    return beats, labels


# --- GET ALL RECORD IDS ---
def get_all_records():
    return [f.split('.')[0] for f in os.listdir(DATA_PATH) if f.endswith('.dat')]


# --- BUILD FULL DATASET ---
def build_dataset():
    all_beats = []
    all_labels = []

    records = get_all_records()
    print(f"Found {len(records)} records")

    for record_id in records:
        try:
            beats, labels = extract_beats(record_id)
            all_beats.extend(beats)
            all_labels.extend(labels)
            print(f"Processed {record_id} | Beats: {len(beats)}")
        except Exception as e:
            print(f"Skipped {record_id}: {e}")

    return np.array(all_beats), np.array(all_labels)


# --- LABEL CLEANING ---
label_map = {
    'N': 0, 'L': 0, 'R': 0,   # Normal
    'A': 1, 'a': 1, 'J': 1,   # Atrial
    'V': 2, 'E': 2,           # Ventricular
    'F': 3                    # Fusion
}


def clean_dataset(X, y):
    clean_X = []
    clean_y = []

    for beat, label in zip(X, y):
        if label in label_map:
            clean_X.append(beat)
            clean_y.append(label_map[label])

    return np.array(clean_X), np.array(clean_y)


# --- MAIN ---
if __name__ == "__main__":
    print("\nBuilding dataset...\n")

    X, y = build_dataset()

    print("\nBefore cleaning:")
    print("Shape:", X.shape)
    print("Labels:", set(y))

    X, y = clean_dataset(X, y)

    print("\nAfter cleaning:")
    print("Shape:", X.shape)
    print("Label distribution:", Counter(y))

    # Define the directory path
    PROCESSED_DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../Data/processed/"))

    # Create the folder if it's missing
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    # Save the files
    np.save(os.path.join(PROCESSED_DATA_PATH, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y.npy"), y)

    print(f"\nSuccess! Files saved inside: {PROCESSED_DATA_PATH}")
