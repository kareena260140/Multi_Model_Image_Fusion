
"""
End-to-end BraTS segmentation pipeline.

Run
---
    python main.py

Steps
-----
1.  Download BraTS 2020 dataset via kagglehub
2.  Collect .h5 file paths
3.  Build raw (pre-split) X / Y arrays  — no augmentation yet
4.  Save raw arrays to disk
5.  Train / test split  ← split happens BEFORE augmentation
6.  Augment training set only  ← prevents data leakage into test set
7.  Loss / metric sanity check
8.  Build model
9.  Compile model
10. Train
11. Evaluate (Keras metrics + HD95)
12. Plot training curves + sample predictions
"""

import numpy as np
from sklearn.model_selection import train_test_split

from configs.config import (
    X_ARRAY_PATH, Y_ARRAY_PATH, TEST_SIZE, RANDOM_SEED,
)
from data.loader       import download_dataset, collect_h5_files
from data.preprocessing import build_raw_arrays, augment_training_set
from models.unet        import build_multistream_attention_unet
from training.losses    import run_loss_sanity_check
from training.trainer   import compile_model, train_model, evaluate_model
from utils.visualize    import plot_sample, plot_training_curves, plot_predictions


def main() -> None:

    # ── 1. Data ──────────────────────────────────────────────────────
    dataset_path = download_dataset()
    h5_files     = collect_h5_files(dataset_path)

    # ── 2. Build raw arrays (no augmentation) ────────────────────────
    X_array, Y_array = build_raw_arrays(h5_files)

    np.save(X_ARRAY_PATH, X_array)
    np.save(Y_ARRAY_PATH, Y_array)
    print(f"\nRaw arrays saved → {X_ARRAY_PATH}, {Y_ARRAY_PATH}")

    # ── 3. Sanity visualization (optional, comment out if headless) ──
    plot_sample(X_array, Y_array)

    # ── 4. Train / test split  ← BEFORE augmentation ─────────────────
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_array, Y_array,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    print(f"\nTrain : {X_train.shape}  |  Test : {X_test.shape}")

    # ── 5. Augment training set only ──────────────────────────────────
    X_train, Y_train = augment_training_set(X_train, Y_train)

    # ── 6. Loss / metric sanity check ────────────────────────────────
    run_loss_sanity_check()

    # ── 7. Build + compile model ──────────────────────────────────────
    model = build_multistream_attention_unet()
    model.summary()
    compile_model(model)

    # ── 8. Train ──────────────────────────────────────────────────────
    history = train_model(model, X_train, Y_train, X_test, Y_test)

    # ── 9. Evaluate ───────────────────────────────────────────────────
    evaluate_model(model, X_test, Y_test)

    # ── 10. Visualize ─────────────────────────────────────────────────
    plot_training_curves(history)
    plot_predictions(model, X_test, Y_test, num_samples=4)


if __name__ == "__main__":
    main()
```
