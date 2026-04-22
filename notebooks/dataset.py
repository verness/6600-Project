"""
ECG DATASET & AUGMENTATION PIPELINE
-------------------------------------
Loads the processed MIT-BIH splits (X, y, meta), applies SMOTE to the
training set to oversample minority arrhythmia classes, and exposes a
PyTorch Dataset that applies online signal augmentation during training.

Usage
-----
    from dataset import build_dataloaders

    train_loader, val_loader, test_loader = build_dataloaders(
        processed_dir = "../../Data/processed",
        spectrogram_h = 64,
        spectrogram_w = 64,
        batch_size    = 32,
    )

    for spectrograms, metadata, labels in train_loader:
        logits = model(spectrograms, metadata)

Design decisions
----------------
SMOTE strategy
    Raw 1-D waveforms (250 samples) and metadata (6 features) are
    concatenated into a single flat vector before SMOTE so that
    synthetic neighbours are consistent across both modalities.
    They are split back apart afterwards.  SMOTE is applied only to
    the training split — val/test are never touched.

    Target: oversample every minority class up to 50 % of the majority
    class count.  This avoids extreme over-representation of tiny
    classes while still giving the model meaningful exposure to them.
    Adjust `smote_sampling_strategy` to taste.

Spectrogram conversion
    The 1-D beat waveform is converted to a 2-D spectrogram on-the-fly
    inside the Dataset using a short-time Fourier transform (STFT).
    This keeps the stored .npy files small and lets you tune STFT
    parameters without re-running preprocessing.

Online augmentation (training only)
    Four lightweight transforms are applied randomly to the raw waveform
    BEFORE the STFT so that augmented samples look like real physiology:

        1. Gaussian noise   – models electrode/amplifier noise
        2. Baseline wander  – low-frequency sinusoidal drift (common in
                              ambulatory recordings)
        3. Amplitude scale  – ±20 % gain variation
        4. Random time-crop – shifts the waveform window by ±10 samples
                              (recentres the R-peak slightly)

    Metadata is never augmented — demographic/RR values are fixed
    properties of the recording.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from collections import Counter
from typing import Literal


# ─────────────────────────────────────────────────────────────────────────────
# 1. Signal augmentation helpers  (operate on raw 1-D numpy waveforms)
# ─────────────────────────────────────────────────────────────────────────────

def augment_waveform(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply a random combination of four physiologically-motivated
    augmentations to a single beat waveform.

    Args:
        wave : (L,) float32 — raw beat samples
        rng  : numpy random Generator (passed in so the Dataset controls seeding)

    Returns:
        wave : (L,) float32 — augmented copy (original is not mutated)
    """
    wave = wave.copy()
    L    = len(wave)

    # 1. Gaussian noise  (σ = 0.5–2 % of signal range)
    if rng.random() < 0.6:
        sigma = rng.uniform(0.005, 0.02) * (wave.max() - wave.min() + 1e-8)
        wave += rng.normal(0, sigma, size=L).astype(np.float32)

    # 2. Baseline wander  (low-frequency sinusoid, 0.05–0.5 Hz at 360 Hz SR)
    if rng.random() < 0.5:
        freq      = rng.uniform(0.05, 0.5)
        amplitude = rng.uniform(0.01, 0.05) * (wave.max() - wave.min() + 1e-8)
        phase     = rng.uniform(0, 2 * np.pi)
        t         = np.arange(L, dtype=np.float32) / 360.0
        wave     += (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)

    # 3. Amplitude scaling  (±20 %)
    if rng.random() < 0.5:
        scale = rng.uniform(0.80, 1.20)
        wave  = (wave * scale).astype(np.float32)

    # 4. Random time-crop / shift  (±10 samples, zero-pad edges)
    if rng.random() < 0.4:
        shift = int(rng.integers(-10, 11))   # inclusive on both ends
        wave  = np.roll(wave, shift).astype(np.float32)
        # Zero out the wrapped-around edge rather than leaving artefacts
        if shift > 0:
            wave[:shift] = 0.0
        elif shift < 0:
            wave[shift:] = 0.0

    return wave


# ─────────────────────────────────────────────────────────────────────────────
# 2. 1-D waveform → 2-D spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def waveform_to_spectrogram(
    wave: np.ndarray,
    target_h: int = 64,
    target_w: int = 64,
    n_fft: int = 64,
    hop_length: int = 4,
) -> np.ndarray:
    """
    Convert a 1-D beat waveform to a log-magnitude spectrogram and
    resize it to (target_h, target_w).

    The STFT output is (n_fft//2 + 1, frames); we crop/pad both
    dimensions to exactly (target_h, target_w) so that the CNN always
    receives a fixed-size image regardless of beat length.

    Args:
        wave       : (L,) float32
        target_h   : Frequency-axis height of the output image
        target_w   : Time-axis width of the output image
        n_fft      : FFT window size
        hop_length : Hop between windows

    Returns:
        spec : (target_h, target_w) float32 — log-magnitude spectrogram,
               normalised to [0, 1]
    """
    # Hann window
    window = np.hanning(n_fft).astype(np.float32)

    # Pad wave if shorter than n_fft
    if len(wave) < n_fft:
        wave = np.pad(wave, (0, n_fft - len(wave)))

    # STFT via strided sliding windows
    n_frames = 1 + (len(wave) - n_fft) // hop_length
    frames   = np.stack(
        [wave[i * hop_length : i * hop_length + n_fft] * window
         for i in range(n_frames)],
        axis=1,
    )                                         # (n_fft, n_frames)
    spectrum  = np.fft.rfft(frames, axis=0)   # (n_fft//2+1, n_frames)
    magnitude = np.abs(spectrum).astype(np.float32)
    log_mag   = np.log1p(magnitude)           # log1p for numerical stability

    # Resize to (target_h, target_w) by cropping / zero-padding
    freq_bins, time_frames = log_mag.shape

    # Frequency axis
    if freq_bins >= target_h:
        log_mag = log_mag[:target_h, :]
    else:
        log_mag = np.pad(log_mag, ((0, target_h - freq_bins), (0, 0)))

    # Time axis
    if time_frames >= target_w:
        log_mag = log_mag[:, :target_w]
    else:
        log_mag = np.pad(log_mag, ((0, 0), (0, target_w - time_frames)))

    # Normalise to [0, 1]
    vmin, vmax = log_mag.min(), log_mag.max()
    if vmax > vmin:
        log_mag = (log_mag - vmin) / (vmax - vmin)

    return log_mag   # (target_h, target_w)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SMOTE oversampling  (training split only)
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    meta_train: np.ndarray,
    y_train: np.ndarray,
    smote_sampling_strategy: dict | str = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Oversample minority arrhythmia classes on the training set using SMOTE.

    SMOTE interpolates in the combined waveform+metadata feature space so
    that synthetic samples are coherent across both modalities.

    Args:
        X_train     : (N, 250)  — raw beat waveforms
        meta_train  : (N, 6)    — patient/beat metadata
        y_train     : (N,)      — integer class labels
        smote_sampling_strategy:
            dict  → {class_label: target_count} for explicit control
            'auto'→ oversample all minorities up to the majority count
            Example for 50 % of majority:
                counts = Counter(y_train)
                majority = max(counts.values())
                strategy = {c: max(n, majority // 2)
                            for c, n in counts.items()
                            if n < majority}
        k_neighbors : Number of nearest neighbours for SMOTE interpolation.
                      Reduce to 3 if any minority class has fewer than 6 samples.
        random_state: Reproducibility seed.

    Returns:
        X_res    : (N', 250)
        meta_res : (N', 6)
        y_res    : (N',)
    """
    print("\n=== SMOTE Oversampling ===")
    print(f"  Before — {Counter(y_train.tolist())}")

    # Concatenate waveform + metadata into one feature matrix for SMOTE
    combined = np.concatenate([X_train, meta_train], axis=1)  # (N, 256)

    # Guard: k_neighbors must be < the smallest minority class count
    min_class_count = min(Counter(y_train.tolist()).values())
    k = min(k_neighbors, min_class_count - 1)
    if k < 1:
        print("  WARNING: A minority class has only 1 sample — skipping SMOTE.")
        return X_train, meta_train, y_train

    smote = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        k_neighbors=k,
        random_state=random_state,
    )
    combined_res, y_res = smote.fit_resample(combined, y_train)

    # Split back into waveforms and metadata
    X_res    = combined_res[:, :X_train.shape[1]].astype(np.float32)
    meta_res = combined_res[:, X_train.shape[1]:].astype(np.float32)

    print(f"  After  — {Counter(y_res.tolist())}")
    print(f"  Added {len(y_res) - len(y_train):,} synthetic samples")
    return X_res, meta_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# 4. PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG beat classification.

    Each item returns:
        spectrogram : (1, H, W) float32 tensor  — log-magnitude STFT image
        metadata    : (M,)      float32 tensor  — patient/beat features
        label       : ()        int64 tensor    — arrhythmia class index

    Args:
        X            : (N, L)  — raw beat waveforms (float32)
        meta         : (N, M)  — metadata matrix (float32)
        y            : (N,)    — integer labels (int64)
        spectrogram_h: Height of output spectrogram image
        spectrogram_w: Width  of output spectrogram image
        n_fft        : STFT window size
        hop_length   : STFT hop size
        augment      : If True, apply online waveform augmentation.
                       Should be True only for the training split.
        seed         : Base seed for the augmentation RNG.
    """

    def __init__(
        self,
        X: np.ndarray,
        meta: np.ndarray,
        y: np.ndarray,
        spectrogram_h: int = 64,
        spectrogram_w: int = 64,
        n_fft: int = 64,
        hop_length: int = 4,
        augment: bool = False,
        seed: int = 0,
    ):
        self.X             = X.astype(np.float32)
        self.meta          = meta.astype(np.float32)
        self.y             = y.astype(np.int64)
        self.spec_h        = spectrogram_h
        self.spec_w        = spectrogram_w
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.augment       = augment
        self.rng           = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        wave = self.X[idx]      # (L,)
        meta = self.meta[idx]   # (M,)
        label = self.y[idx]     # scalar

        # Online augmentation (training only)
        if self.augment:
            wave = augment_waveform(wave, self.rng)

        # Convert waveform to spectrogram image
        spec = waveform_to_spectrogram(
            wave,
            target_h=self.spec_h,
            target_w=self.spec_w,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )                                                # (H, W)

        spec_tensor  = torch.from_numpy(spec).unsqueeze(0)  # (1, H, W)
        meta_tensor  = torch.from_numpy(meta)                # (M,)
        label_tensor = torch.tensor(label, dtype=torch.long) # scalar

        return spec_tensor, meta_tensor, label_tensor


# ─────────────────────────────────────────────────────────────────────────────
# 5. Convenience builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str,
    spectrogram_h: int = 64,
    spectrogram_w: int = 64,
    n_fft: int = 64,
    hop_length: int = 4,
    batch_size: int = 32,
    num_workers: int = 0,
    smote: bool = True,
    smote_target_ratio: float = 0.5,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
):
    """
    Load the processed splits, optionally apply SMOTE to training data,
    and return (train_loader, val_loader, test_loader).

    Args:
        processed_dir      : Path to the folder containing the .npy split files.
        spectrogram_h/w    : Spectrogram image dimensions fed into the CNN.
        n_fft / hop_length : STFT parameters.
        batch_size         : Mini-batch size.
        num_workers        : DataLoader worker processes.
        smote              : Whether to apply SMOTE to the training split.
        smote_target_ratio : Minority classes are oversampled up to this
                             fraction of the majority class count (0.5 = 50 %).
                             Set to 1.0 to fully balance all classes.
        smote_k_neighbors  : SMOTE k parameter.
        random_state       : Seed for SMOTE and dataset RNGs.

    Returns:
        train_loader, val_loader, test_loader
    """
    # ── Load splits ──────────────────────────────────────────────────────────
    def load(name):
        return np.load(os.path.join(processed_dir, f"{name}.npy"))

    X_train, y_train, meta_train = load("X_train"), load("y_train"), load("meta_train")
    X_val,   y_val,   meta_val   = load("X_val"),   load("y_val"),   load("meta_val")
    X_test,  y_test,  meta_test  = load("X_test"),  load("y_test"),  load("meta_test")

    print(f"Loaded splits:")
    print(f"  Train : X={X_train.shape}  meta={meta_train.shape}  y={Counter(y_train.tolist())}")
    print(f"  Val   : X={X_val.shape}    meta={meta_val.shape}    y={Counter(y_val.tolist())}")
    print(f"  Test  : X={X_test.shape}   meta={meta_test.shape}   y={Counter(y_test.tolist())}")

    # ── SMOTE on training split ───────────────────────────────────────────────
    if smote:
        counts   = Counter(y_train.tolist())
        majority = max(counts.values())
        target   = {
            cls: max(n, int(majority * smote_target_ratio))
            for cls, n in counts.items()
            if n < majority
        }
        # Only pass a strategy dict if there's anything to oversample
        strategy = target if target else "auto"
        X_train, meta_train, y_train = apply_smote(
            X_train, meta_train, y_train,
            smote_sampling_strategy=strategy,
            k_neighbors=smote_k_neighbors,
            random_state=random_state,
        )

    # ── Build Datasets ───────────────────────────────────────────────────────
    shared_spec_kwargs = dict(
        spectrogram_h=spectrogram_h,
        spectrogram_w=spectrogram_w,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    train_ds = ECGDataset(X_train, meta_train, y_train,
                          augment=True, seed=random_state, **shared_spec_kwargs)
    val_ds   = ECGDataset(X_val,   meta_val,   y_val,
                          augment=False, **shared_spec_kwargs)
    test_ds  = ECGDataset(X_test,  meta_test,  y_test,
                          augment=False, **shared_spec_kwargs)

    # ── Build DataLoaders ────────────────────────────────────────────────────
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    print(f"\nDataLoaders ready:")
    print(f"  Train batches : {len(train_loader)}  ({len(train_ds):,} samples)")
    print(f"  Val   batches : {len(val_loader)}   ({len(val_ds):,} samples)")
    print(f"  Test  batches : {len(test_loader)}   ({len(test_ds):,} samples)")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== Sanity check (synthetic data) ===\n")

    # Generate fake splits that mimic the real class imbalance
    rng = np.random.default_rng(0)
    N   = 1000
    # Class distribution roughly matching MIT-BIH: ~70% N, 10% A, 10% V, 10% F
    y_fake = np.array(
        [0] * 700 + [1] * 100 + [2] * 150 + [3] * 50, dtype=np.int64
    )
    rng.shuffle(y_fake)
    X_fake    = rng.standard_normal((N, 250)).astype(np.float32)
    meta_fake = rng.standard_normal((N, 6)).astype(np.float32)

    # Save to a temp directory so build_dataloaders can load them
    with tempfile.TemporaryDirectory() as tmpdir:
        for split, (xi, mi, yi) in zip(
            ["train", "val", "test"],
            [
                (X_fake[:700],  meta_fake[:700],  y_fake[:700]),
                (X_fake[700:850], meta_fake[700:850], y_fake[700:850]),
                (X_fake[850:],  meta_fake[850:],  y_fake[850:]),
            ],
        ):
            np.save(os.path.join(tmpdir, f"X_{split}.npy"),    xi)
            np.save(os.path.join(tmpdir, f"meta_{split}.npy"), mi)
            np.save(os.path.join(tmpdir, f"y_{split}.npy"),    yi)

        train_loader, val_loader, test_loader = build_dataloaders(
            processed_dir=tmpdir,
            spectrogram_h=64,
            spectrogram_w=64,
            batch_size=16,
            smote=True,
            smote_target_ratio=0.5,
        )

        # Check one batch
        specs, metas, labels = next(iter(train_loader))
        print(f"\nFirst training batch:")
        print(f"  spectrogram : {tuple(specs.shape)}   dtype={specs.dtype}")
        print(f"  metadata    : {tuple(metas.shape)}   dtype={metas.dtype}")
        print(f"  labels      : {tuple(labels.shape)}  dtype={labels.dtype}")
        print(f"  label values: {labels.tolist()}")
        assert specs.shape[1:] == (1, 64, 64), "Unexpected spectrogram shape"
        assert metas.shape[1]  == 6,           "Unexpected metadata dim"
        print("\nAll checks passed.")
