# config.py
```python
# configsconfig.py

Central configuration for the BraTS segmentation pipeline.
Edit this file to change any hyperparameter or path before running main.py.


import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
SAVE_DIR        = content                          # root output directory
CHECKPOINT_PATH = os.path.join(SAVE_DIR, best_model.keras)
CSV_LOG_PATH    = os.path.join(SAVE_DIR, training_log.csv)
X_ARRAY_PATH    = os.path.join(SAVE_DIR, X_array.npy)
Y_ARRAY_PATH    = os.path.join(SAVE_DIR, Y_array.npy)
CURVES_PATH     = os.path.join(SAVE_DIR, training_curves.png)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
KAGGLE_DATASET  = awsaf49brats2020-training-data
MAX_FILES       = 3000     # cap on H5 files to scan
TARGET_SAMPLES  = 1800     # total samples after aug; increase if RAM allows
EMPTY_KEEP_PROB = 0.2      # fraction of tumor-free slices to keep (reduced from 0.4)
TEST_SIZE       = 0.2      # traintest split ratio
RANDOM_SEED     = 42

# ─────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────
IMG_SIZE        = 128      # height = width (square resize)
NUM_CHANNELS    = 4        # T1, T1CE, T2, FLAIR

# BraTS HDF5 channel indices  (image shape 240×240×4)
MODALITY_INDICES = {
    T1    0,
    T1CE  1,
    T2    2,
    FLAIR 3,
}

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
CBAM_REDUCTION  = 8        # channel reduction ratio inside CBAM
DROPOUT_RATE    = 0.1      # dropout between double-conv layers

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS          = 80
BATCH_SIZE      = 16
TARGET_LR       = 5e-4
WARMUP_EPOCHS   = 5

# ReduceLROnPlateau
RLROP_FACTOR    = 0.5
RLROP_PATIENCE  = 4
RLROP_MIN_LR    = 1e-7
RLROP_COOLDOWN  = 2

# EarlyStopping
ES_PATIENCE     = 12
ES_MIN_DELTA    = 0.001

# ─────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────
FOCAL_TVERSKY_WEIGHT = 0.8
BINARY_FOCAL_WEIGHT  = 0.2

# Tversky αβ  (α=FP penalty, β=FN penalty)
# β  α missing a tumor is worse than a false alarm
TVERSKY_ALPHA   = 0.3
TVERSKY_BETA    = 0.7
TVERSKY_GAMMA   = 0.75     # focal exponent

# Binary focal loss
FOCAL_GAMMA     = 2.0
FOCAL_ALPHA     = 0.25

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
HD95_SAMPLES    = 200      # number of test samples for HD95 computation
```
