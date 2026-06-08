
"""
Normalization, augmentation, and array construction for BraTS slices.

Key fixes vs. original code
----------------------------
* Data-leakage fix: augmentation is applied AFTER train/test split,
  so augmented copies of a training slice cannot appear in the test set.
* Spatially-consistent augmentation via albumentations additional_targets.
* EMPTY_KEEP_PROB reduced to 0.2 for a better tumor/background ratio.
* ElasticTransform called without the removed alpha_affine kwarg
  (compatible with albumentations ≥ 2.0).
"""

import numpy as np
import cv2
import h5py
from tqdm import tqdm
import albumentations as A

from configs.config import (
    IMG_SIZE, TARGET_SAMPLES, EMPTY_KEEP_PROB,
    MODALITY_INDICES, RANDOM_SEED,
)

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_zscore(img: np.ndarray) -> np.ndarray:
    """
    Z-score normalization per modality slice, then rescale to [0, 1].

    MRI intensities have no fixed physical scale, so min-max rescaling
    is dominated by outlier hot pixels / scanner artefacts.
    We clamp at the 1st/99th percentile *before* standardizing.

    Steps
    -----
    1. Cast to float32.
    2. Clip to [p1, p99] to suppress outliers.
    3. Subtract mean, divide by std.
    4. Rescale result to [0, 1] for neural-network numerical stability.
    """
    img = img.astype(np.float32)
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip(img, p1, p99)
    mean = np.mean(img)
    std  = np.std(img) + 1e-8
    img  = (img - mean) / std
    img  = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# ─────────────────────────────────────────────
# AUGMENTATION PIPELINE
# Applied only during the training-set augmentation pass (after split).
# All spatial transforms are applied identically to every channel and
# the mask via albumentations' additional_targets mechanism.
# ─────────────────────────────────────────────

_augmentor = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
        # alpha_affine removed — not supported in albumentations ≥ 2.0
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.GaussNoise(var_limit=(0.001, 0.005), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.3
        ),
    ],
    additional_targets={
        "T1CE":  "image",
        "T2":    "image",
        "FLAIR": "image",
        "mask":  "mask",
    },
)


def augment_sample(
    channels_dict: dict[str, np.ndarray],
    mask_2d: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Apply spatially-consistent augmentation to all 4 modality channels
    and the binary mask.

    Parameters
    ----------
    channels_dict : dict
        Keys "T1", "T1CE", "T2", "FLAIR" → float32 arrays of shape (H, W).
    mask_2d : np.ndarray
        Binary float32 mask of shape (H, W).

    Returns
    -------
    aug_channels : dict  (same keys, augmented)
    aug_mask     : np.ndarray  float32 (H, W)
    """
    result = _augmentor(
        image  = channels_dict["T1"],
        T1CE   = channels_dict["T1CE"],
        T2     = channels_dict["T2"],
        FLAIR  = channels_dict["FLAIR"],
        mask   = mask_2d.astype(np.uint8),
    )
    aug_channels = {
        "T1":    result["image"],
        "T1CE":  result["T1CE"],
        "T2":    result["T2"],
        "FLAIR": result["FLAIR"],
    }
    aug_mask = result["mask"].astype(np.float32)
    return aug_channels, aug_mask


# ─────────────────────────────────────────────
# RAW SLICE EXTRACTION  (no augmentation yet)
# ─────────────────────────────────────────────

def _extract_slice(file_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read one BraTS .h5 file, resize to IMG_SIZE, and return the
    stacked 4-channel image and binary mask.

    Returns None if the file cannot be read.
    """
    try:
        with h5py.File(file_path, "r") as f:
            image = f["image"][:]   # (240, 240, 4)
            mask  = f["mask"][:]    # (240, 240) or (240, 240, N)

        # Collapse multi-class BraTS mask to binary
        # Labels: 0=background, 1=necrotic core, 2=edema, 4=enhancing tumor
        if mask.ndim == 3:
            mask = np.max(mask, axis=-1)
        mask = (mask > 0).astype(np.float32)

        channels: dict[str, np.ndarray] = {}
        for name, idx in MODALITY_INDICES.items():
            ch = normalize_zscore(image[:, :, idx])
            ch = cv2.resize(ch, (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA)
            channels[name] = ch

        mask_resized = cv2.resize(
            mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST
        )

        input_img = np.stack(
            [channels["T1"], channels["T1CE"],
             channels["T2"], channels["FLAIR"]],
            axis=-1,
        ).astype(np.float32)                              # (H, W, 4)

        mask_final = np.expand_dims(mask_resized, axis=-1)  # (H, W, 1)
        return input_img, mask_final

    except Exception:
        return None


def build_raw_arrays(
    h5_files: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterate over .h5 files and build the original (non-augmented)
    X / Y arrays with balanced tumor/empty sampling.

    Augmentation is intentionally NOT applied here — it is applied
    later, only on the training split, to prevent data leakage.

    Parameters
    ----------
    h5_files : list[str]
        Paths collected by data.loader.collect_h5_files().

    Returns
    -------
    X_array : np.ndarray  float32  (N, IMG_SIZE, IMG_SIZE, 4)
    Y_array : np.ndarray  float32  (N, IMG_SIZE, IMG_SIZE, 1)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    X_list: list[np.ndarray] = []
    Y_list: list[np.ndarray] = []
    tumor_count = 0
    empty_count = 0

    print("\nBuilding raw (pre-split) arrays — no augmentation yet...\n")

    for file_path in tqdm(h5_files):
        if len(X_list) >= TARGET_SAMPLES:
            break

        result = _extract_slice(file_path)
        if result is None:
            continue

        input_img, mask_final = result
        has_tumor = np.sum(mask_final) > 0

        if has_tumor:
            X_list.append(input_img)
            Y_list.append(mask_final)
            tumor_count += 1
        elif rng.random() < EMPTY_KEEP_PROB:
            X_list.append(input_img)
            Y_list.append(mask_final)
            empty_count += 1

    X_array = np.array(X_list, dtype=np.float32)
    Y_array = np.array(Y_list, dtype=np.float32)

    print("\n✅ Raw arrays built")
    print(f"   Total samples   : {len(X_array)}")
    print(f"   Tumor slices    : {tumor_count}")
    print(f"   Empty slices    : {empty_count}")
    print(f"   X shape: {X_array.shape}  |  Y shape: {Y_array.shape}")
    _print_channel_stats(X_array)

    return X_array, Y_array


def _print_channel_stats(X: np.ndarray) -> None:
    ch_names = list(MODALITY_INDICES.keys())
    print("\nPer-channel value ranges:")
    for i, name in enumerate(ch_names):
        ch = X[:, :, :, i]
        print(f"  {name:5s} — min: {ch.min():.4f}  "
              f"max: {ch.max():.4f}  mean: {ch.mean():.4f}")
    print(f"\nMask unique values   : {np.unique(X)[:5]} …")


# ─────────────────────────────────────────────
# POST-SPLIT AUGMENTATION  (training set only)
# ─────────────────────────────────────────────

def augment_training_set(
    X_train: np.ndarray,
    Y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Double the tumor-positive training samples by creating one
    augmented copy of each tumor slice.

    Called AFTER train_test_split so augmented samples never
    contaminate the test set.

    Parameters
    ----------
    X_train, Y_train : np.ndarray

    Returns
    -------
    X_aug, Y_aug : np.ndarray  (original + augmented tumor slices)
    """
    X_aug: list[np.ndarray] = list(X_train)
    Y_aug: list[np.ndarray] = list(Y_train)
    aug_count = 0

    print("\nAugmenting training set (tumor slices only)...")

    for img, mask in zip(X_train, Y_train):
        if np.sum(mask) == 0:
            continue   # skip empty slices

        channels_dict = {
            "T1":    img[:, :, 0],
            "T1CE":  img[:, :, 1],
            "T2":    img[:, :, 2],
            "FLAIR": img[:, :, 3],
        }
        aug_ch, aug_mask = augment_sample(channels_dict, mask.squeeze(-1))

        aug_img = np.stack(
            [aug_ch["T1"], aug_ch["T1CE"],
             aug_ch["T2"], aug_ch["FLAIR"]],
            axis=-1,
        ).astype(np.float32)

        X_aug.append(aug_img)
        Y_aug.append(np.expand_dims(aug_mask, axis=-1))
        aug_count += 1

    X_out = np.array(X_aug, dtype=np.float32)
    Y_out = np.array(Y_aug, dtype=np.float32)

    print(f"   Augmented copies added : {aug_count}")
    print(f"   Training set size      : {len(X_out)}")
    return X_out, Y_out
```
