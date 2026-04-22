
# Applies SMOTE (Synthetic Minority Oversampling TEchnique) to ONLY the
# training set to balance the heavy class imbalance in the MIT-BIH dataset.

# CRITICAL: SMOTE is applied AFTER the train/val/test split and ONLY to the
# training set. Applying it before the split, or to val/test, causes data
# leakage and inflates evaluation metrics.

# Input  : X_train.npy, y_train.npy  (from split.py)
# Output : X_train_smote.npy, y_train_smote.npy

import os
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

# --- PATH SETUP ---
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../Data/processed"))

LABEL_NAMES = {
    0: "N - Normal",
    1: "S - Supraventricular",
    2: "V - Ventricular",
    3: "Q - Other"
}


def main():
    print("=" * 60)
    print("  SMOTE Oversampling (training set only)")
    print("=" * 60)

    # --- Load training data ---
    try:
        X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    except FileNotFoundError:
        print(f"ERROR: X_train.npy / y_train.npy not found in {PROCESSED_DIR}")
        print("Run preprocess.py and split.py first.")
        return

    print(f"\n  Loaded:  X_train={X_train.shape}, y_train={y_train.shape}")

    # --- Distribution BEFORE SMOTE ---
    before = Counter(y_train.tolist())
    print("\n  Before SMOTE:")
    for c in sorted(before):
        print(f"    {LABEL_NAMES[c]:<28} {before[c]:>7,}")

    # --- Apply SMOTE ---
    # k_neighbors=5 is the default; reduce if the smallest class has <6 samples
    smallest_class_count = min(before.values())
    k_neighbors = min(5, smallest_class_count - 1)

    print(f"\n  Applying SMOTE (k_neighbors={k_neighbors})...")
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

    # --- Distribution AFTER SMOTE ---
    after = Counter(y_resampled.tolist())
    print("\n  After SMOTE:")
    for c in sorted(after):
        print(f"    {LABEL_NAMES[c]:<28} {after[c]:>7,}")

    print(f"\n  New training shape: X={X_resampled.shape}, y={y_resampled.shape}")

    # --- Save ---
    np.save(os.path.join(PROCESSED_DIR, "X_train_smote.npy"),
            X_resampled.astype(np.float32))
    np.save(os.path.join(PROCESSED_DIR, "y_train_smote.npy"),
            y_resampled.astype(np.int64))

    print("\n  Saved to:")
    print(f"    {PROCESSED_DIR}/X_train_smote.npy")
    print(f"    {PROCESSED_DIR}/y_train_smote.npy")
    print("\n  Validation/test sets are UNCHANGED (use original X_val / X_test)")
    print("=" * 60)

if __name__ == "__main__":
    main()
