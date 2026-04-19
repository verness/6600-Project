"""
DATA SPLITTING SCRIPT
---------------------
This script loads the processed MIT-BIH Arrhythmia dataset and performs a 
stratified split into Training (70%), Validation (15%), and Testing (15%) sets.

Stratification is used to maintain the ratio of arrhythmia classes across all 
splits, which is critical due to the inherent class imbalance in the dataset.
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- DYNAMIC PATH SETUP ---
# BASE_DIR is 'src/data'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PROCESSED_DIR points to 'Project_Root/Data/processed'
PROCESSED_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../Data/processed"))

def main():
    print(f"--- Starting Data Split ---")
    
    # 1. Load the cleaned dataset
    try:
        X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
        y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
        print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    except FileNotFoundError:
        print(f"ERROR: Could not find X.npy/y.npy in {PROCESSED_DIR}")
        return

    # 2. First split: Separate Training (70%) and Temporary (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 3. Second split: Split the Temporary 30% into Val (15%) and Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # 4. Define and Save splits
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test
    }

    print(f"Saving files to {PROCESSED_DIR}...")
    for name, data in splits.items():
        np.save(os.path.join(PROCESSED_DIR, f"{name}.npy"), data)
        print(f" - Saved {name}.npy | Shape: {data.shape}")

    print("--- Data Split Complete ---")

if __name__ == "__main__":
    main()