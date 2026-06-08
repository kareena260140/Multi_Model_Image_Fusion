
"""
Downloads the BraTS 2020 dataset via kagglehub and collects
all .h5 file paths up to MAX_FILES.
"""

import os
import kagglehub
from configs.config import KAGGLE_DATASET, MAX_FILES


def download_dataset() -> str:
    """
    Downloads (or retrieves cached) BraTS 2020 dataset.

    Returns
    -------
    str
        Local path to the dataset root directory.
    """
    print(f"Downloading dataset: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Dataset path: {path}")
    return path


def collect_h5_files(dataset_path: str) -> list[str]:
    """
    Recursively collects all .h5 files under ``dataset_path``,
    sorts them, and caps the list at MAX_FILES.

    Parameters
    ----------
    dataset_path : str
        Root directory returned by ``download_dataset()``.

    Returns
    -------
    list[str]
        Sorted list of absolute .h5 file paths.
    """
    h5_files = []
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            if fname.endswith(".h5"):
                h5_files.append(os.path.join(root, fname))

    h5_files = sorted(h5_files)[:MAX_FILES]
    print(f"Total .h5 files found: {len(h5_files)}")
    return h5_files
```
