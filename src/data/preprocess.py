"""
ECG PREPROCESSING PIPELINE: MIT-BIH Arrhythmia Database
---------------------------------------------------------
Extracts individual heartbeats from raw MIT-BIH ECG records,
parses patient metadata from .hea comment lines, derives
signal-level features per beat, and saves everything as NumPy arrays.

Outputs (saved to Data/processed/):
    X.npy          – Beat waveforms,     shape (N, 250)
    y.npy          – Integer labels,     shape (N,)
    meta.npy       – Metadata matrix,    shape (N, 6)   float32
    meta_info.json – Column names + encoding key for meta.npy

Metadata columns (in order):
    0  age          – Patient age in years (NaN → imputed with dataset median)
    1  sex          – 0 = male, 1 = female, 0.5 = unknown
    2  n_meds       – Count of medications listed in header
    3  rr_pre       – RR interval BEFORE this beat (samples @ 360 Hz)
    4  rr_post      – RR interval AFTER  this beat (samples @ 360 Hz)
    5  rr_ratio     – rr_pre / rr_post  (ectopy indicator; 1.0 = regular)

--- What the .hea comment lines contain ---
MIT-BIH .hea files store patient info in free-text comment lines starting
with '#'.  The convention (not formally standardised) is:

    Line 1:  # <age> <sex> <gain_ch1> <gain_ch2> <x_factor>
             e.g.  "# 69 M 1085 1629 x1"
    Line 2+: # <medication1>, <medication2>, ...
             e.g.  "# Aldomet, Inderal"

Age and sex are the only reliably structured fields across all 48 records;
the numeric columns after sex are signal-quality parameters, not clinical data.

Requirements: wfdb, numpy
"""

import wfdb
import numpy as np
import os
import json
import re
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../Data/mit-bih-arrhythmia-database-1.0.0/")
)
PROCESSED_DATA_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../Data/processed/")
)

print("Resolved DATA_PATH:", DATA_PATH)
print("Exists:            ", os.path.exists(DATA_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "N": 0, "L": 0, "R": 0,   # Normal / Bundle Branch Block
    "A": 1, "a": 1, "J": 1,   # Supraventricular / Atrial
    "V": 2, "E": 2,            # Ventricular
    "F": 3,                    # Fusion
}

WINDOW_BEFORE = 100
WINDOW_AFTER  = 150


# ─────────────────────────────────────────────────────────────────────────────
# HEADER METADATA PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_header_metadata(record_path: str) -> dict:
    """
    Read the .hea file for a record and extract patient metadata from
    comment lines (lines beginning with '#').

    MIT-BIH comment convention
    ──────────────────────────
    First comment line:  # <age> <sex> <...signal params...>
        age  – integer, or '?' if unknown
        sex  – 'M' / 'F' / other

    Subsequent comment lines: free-text medication lists.

    Returns a dict with keys: age (float|NaN), sex (float), n_meds (int),
    med_names (list[str]).
    """
    hea_path = record_path + ".hea"
    age   = np.nan
    sex   = 0.5        # 0=M, 1=F, 0.5=unknown
    n_meds    = 0
    med_names = []

    if not os.path.exists(hea_path):
        return dict(age=age, sex=sex, n_meds=n_meds, med_names=med_names)

    with open(hea_path, "r") as f:
        comment_lines = [
            ln.strip().lstrip("#").strip()
            for ln in f
            if ln.strip().startswith("#")
        ]

    if not comment_lines:
        return dict(age=age, sex=sex, n_meds=n_meds, med_names=med_names)

    # ── First comment line: age + sex ────────────────────────────────────────
    # Format:  "69 M 1085 1629 x1"   or   "? F ..."   or just "? ?"
    first = comment_lines[0]
    tokens = first.split()

    if len(tokens) >= 1:
        try:
            age = float(tokens[0])
        except ValueError:
            age = np.nan   # '?' or any non-numeric

    if len(tokens) >= 2:
        sex_token = tokens[1].upper()
        if sex_token == "M":
            sex = 0.0
        elif sex_token == "F":
            sex = 1.0
        # else stays 0.5 (unknown)

    # ── Subsequent comment lines: medications ────────────────────────────────
    # Lines may be comma-separated ("Aldomet, Inderal") or single entries.
    # Some records have "None" or are empty.
    for line in comment_lines[1:]:
        if not line or line.lower() in ("none", "n/a", "-"):
            continue
        meds = [m.strip() for m in re.split(r"[,;]", line) if m.strip()]
        med_names.extend(meds)

    n_meds = len(med_names)

    return dict(age=age, sex=sex, n_meds=n_meds, med_names=med_names)


# ─────────────────────────────────────────────────────────────────────────────
# BEAT EXTRACTION (signal + per-beat RR features)
# ─────────────────────────────────────────────────────────────────────────────

def extract_beats(record_id: str) -> tuple[list, list, list]:
    """
    Extract windowed beats, their labels, and per-beat RR features
    from a single MIT-BIH record.

    RR features derived here (not from .hea) because they require the
    annotation sample positions:
        rr_pre   – samples between this R-peak and the previous one
        rr_post  – samples between this R-peak and the next one
        rr_ratio – rr_pre / rr_post  (deviates from 1 during ectopy)

    Edge beats (first / last, where one neighbour is unavailable) are
    assigned rr_pre = rr_post = median RR for that record, ratio = 1.0.

    Returns (beats, labels, rr_features) — three parallel lists.
    """
    record_path = os.path.join(DATA_PATH, record_id)
    record      = wfdb.rdrecord(record_path)
    annotation  = wfdb.rdann(record_path, "atr")

    signal      = record.p_signal[:, 0]   # Lead II (first channel)
    r_peaks     = annotation.sample
    symbols     = annotation.symbol

    # Compute all RR intervals up front
    rr_intervals = np.diff(r_peaks).astype(float)   # length = len(r_peaks) - 1
    median_rr    = float(np.median(rr_intervals)) if len(rr_intervals) > 0 else 360.0

    beats       = []
    labels      = []
    rr_features = []   # each entry: [rr_pre, rr_post, rr_ratio]

    for i, (r_peak, label) in enumerate(zip(r_peaks, symbols)):
        # Skip non-beat annotations (rhythm/noise markers)
        if label not in LABEL_MAP:
            continue

        # Skip beats too close to signal boundaries
        if r_peak - WINDOW_BEFORE < 0 or r_peak + WINDOW_AFTER >= len(signal):
            continue

        # RR intervals — fall back to median at record edges
        rr_pre  = float(r_peaks[i] - r_peaks[i - 1]) if i > 0 else median_rr
        rr_post = float(r_peaks[i + 1] - r_peaks[i]) if i < len(r_peaks) - 1 else median_rr
        rr_ratio = rr_pre / rr_post if rr_post > 0 else 1.0

        beat = signal[r_peak - WINDOW_BEFORE : r_peak + WINDOW_AFTER]
        beats.append(beat)
        labels.append(label)
        rr_features.append([rr_pre, rr_post, rr_ratio])

    return beats, labels, rr_features


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def get_all_records() -> list[str]:
    return sorted({
        f.split(".")[0]
        for f in os.listdir(DATA_PATH)
        if f.endswith(".dat")
    })


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Iterate over all records, extract beats + metadata, and return four
    parallel arrays / lists:

        X          (N, 250)  – raw beat waveforms
        y_raw      (N,)      – string symbol labels (before integer mapping)
        meta_raw   (N, 6)    – [age, sex, n_meds, rr_pre, rr_post, rr_ratio]
        med_names  (N,)      – medication list strings per beat (for reference)
    """
    all_beats     = []
    all_labels    = []
    all_meta      = []
    all_med_names = []

    records = get_all_records()
    print(f"Found {len(records)} records\n")

    for record_id in records:
        record_path = os.path.join(DATA_PATH, record_id)

        try:
            # --- Signal-derived features ---
            beats, labels, rr_feats = extract_beats(record_id)

            # --- Header-derived metadata ---
            hdr = parse_header_metadata(record_path)
            age    = hdr["age"]
            sex    = hdr["sex"]
            n_meds = hdr["n_meds"]
            meds_str = ", ".join(hdr["med_names"]) if hdr["med_names"] else "none"

            for beat, label, rr in zip(beats, labels, rr_feats):
                all_beats.append(beat)
                all_labels.append(label)
                # Combine header features with per-beat RR features
                all_meta.append([age, sex, float(n_meds)] + rr)
                all_med_names.append(meds_str)

            print(
                f"  {record_id}  |  beats: {len(beats):4d}  |  "
                f"age: {age if not np.isnan(age) else '?':>4}  "
                f"sex: {'M' if sex == 0.0 else 'F' if sex == 1.0 else '?'}  "
                f"meds: {meds_str}"
            )

        except Exception as e:
            print(f"  Skipped {record_id}: {e}")

    X        = np.array(all_beats,  dtype=np.float32)
    y_raw    = np.array(all_labels, dtype=object)
    meta_raw = np.array(all_meta,   dtype=np.float32)

    return X, y_raw, meta_raw, all_med_names


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING & IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def clean_and_map_labels(
    X: np.ndarray,
    y_raw: np.ndarray,
    meta_raw: np.ndarray,
    med_names: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    1. Keep only beats whose symbol is in LABEL_MAP.
    2. Map string symbols → integer class labels.
    """
    mask  = np.array([lbl in LABEL_MAP for lbl in y_raw])
    X     = X[mask]
    y_int = np.array([LABEL_MAP[lbl] for lbl in y_raw[mask]], dtype=np.int64)
    meta  = meta_raw[mask]
    meds  = [med_names[i] for i, m in enumerate(mask) if m]
    return X, y_int, meta, meds


def impute_metadata(meta: np.ndarray) -> np.ndarray:
    """
    Impute missing values (NaN) column-wise using the column median.
    Primarily relevant for the age column, which may be '?' in some records.
    Returns a copy with no NaN values.
    """
    meta = meta.copy()
    for col_idx in range(meta.shape[1]):
        col = meta[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = float(np.nanmedian(col))
            col[nan_mask] = median_val
            meta[:, col_idx] = col
            print(
                f"  Imputed {nan_mask.sum()} NaN(s) in column {col_idx} "
                f"with median={median_val:.2f}"
            )
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Building dataset ===\n")
    X, y_raw, meta_raw, med_names = build_dataset()

    print(f"\nBefore cleaning:")
    print(f"  Beats shape : {X.shape}")
    print(f"  Unique symbols: {set(y_raw)}")

    print("\n=== Cleaning labels ===\n")
    X, y, meta, med_names = clean_and_map_labels(X, y_raw, meta_raw, med_names)

    print(f"\nAfter cleaning:")
    print(f"  Beats shape   : {X.shape}")
    print(f"  Metadata shape: {meta.shape}")
    print(f"  Label distribution: {Counter(y.tolist())}")

    print("\n=== Imputing metadata ===\n")
    meta = impute_metadata(meta)

    # Confirm no NaNs remain
    assert not np.isnan(meta).any(), "NaN values remain after imputation!"
    print("  No NaN values in metadata — imputation complete.")

    # ── Save outputs ─────────────────────────────────────────────────────────
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    np.save(os.path.join(PROCESSED_DATA_PATH, "X.npy"),    X)
    np.save(os.path.join(PROCESSED_DATA_PATH, "y.npy"),    y)
    np.save(os.path.join(PROCESSED_DATA_PATH, "meta.npy"), meta)

    # Save a human-readable description of the metadata columns
    meta_info = {
        "columns": [
            {"index": 0, "name": "age",      "unit": "years",   "notes": "patient age; NaN imputed with dataset median"},
            {"index": 1, "name": "sex",       "unit": "encoded", "notes": "0=male, 1=female, 0.5=unknown"},
            {"index": 2, "name": "n_meds",    "unit": "count",   "notes": "number of medications listed in .hea header"},
            {"index": 3, "name": "rr_pre",    "unit": "samples", "notes": "RR interval before this beat at 360 Hz"},
            {"index": 4, "name": "rr_post",   "unit": "samples", "notes": "RR interval after this beat at 360 Hz"},
            {"index": 5, "name": "rr_ratio",  "unit": "ratio",   "notes": "rr_pre/rr_post; deviates from 1.0 during ectopy"},
        ],
        "label_map": {v: k for k, v in LABEL_MAP.items()},
        "class_names": {
            "0": "Normal/Bundle Branch Block (N, L, R)",
            "1": "Supraventricular/Atrial (A, a, J)",
            "2": "Ventricular (V, E)",
            "3": "Fusion (F)",
        },
        "sample_rate_hz": 360,
        "beat_window": {
            "samples_before_rpeak": WINDOW_BEFORE,
            "samples_after_rpeak":  WINDOW_AFTER,
            "total_samples":        WINDOW_BEFORE + WINDOW_AFTER,
        },
    }

    with open(os.path.join(PROCESSED_DATA_PATH, "meta_info.json"), "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"\n=== Saved to: {PROCESSED_DATA_PATH} ===")
    print(f"  X.npy          {X.shape}      dtype={X.dtype}")
    print(f"  y.npy          {y.shape}        dtype={y.dtype}")
    print(f"  meta.npy       {meta.shape}   dtype={meta.dtype}")
    print(f"  meta_info.json (column descriptions)")
    print("\nDone.")