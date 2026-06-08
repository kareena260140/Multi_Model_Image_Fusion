
"""
Model compilation, training loop, and evaluation.

Key fixes vs. original code
----------------------------
* sample_weight computed AFTER augmentation on the final training set.
* Validation uses a separate sample_weight array so val_loss is
  computed on the same per-pixel scale as train_loss (comparable curves).
* Best weights are loaded explicitly after fit() in case EarlyStopping
  did not trigger (training finished naturally).
"""

import numpy as np
import tensorflow as tf

from configs.config import (
    TARGET_LR, EPOCHS, BATCH_SIZE, CHECKPOINT_PATH, HD95_SAMPLES,
)
from training.losses import (
    combined_loss, dice_coef, iou_metric,
    precision_metric, recall_metric, specificity_metric,
    compute_hd95_dataset,
)
from training.callbacks import build_callbacks


# ─────────────────────────────────────────────
# COMPILE
# ─────────────────────────────────────────────

def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    Compile the model with Adam + combined loss + all metrics.

    Adam configuration
    ------------------
    clipnorm=1.0  prevents exploding gradients in the deep multi-stream
                  architecture, especially during the warmup phase.
    epsilon=1e-7  slightly above Keras default for numerical stability.

    Parameters
    ----------
    model : uncompiled Keras model from build_multistream_attention_unet()

    Returns
    -------
    Compiled model (same object, mutated in place; returned for chaining).
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=TARGET_LR,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,
        ),
        loss=combined_loss,
        metrics=[
            dice_coef,
            iou_metric,
            precision_metric,
            recall_metric,
            specificity_metric,
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    # One-pass verification on a tiny dummy batch
    print("Running compile verification on dummy batch...")
    _x = tf.random.uniform((2, 128, 128, 4))
    _y = tf.cast(tf.random.uniform((2, 128, 128, 1)) > 0.5, tf.float32)
    _vals = model.train_on_batch(_x, _y)

    print("\n✅ Compile verified. Dummy batch results:")
    for name, val in zip(model.metrics_names, _vals):
        print(f"  {name:30s}: {val:.4f}")

    print(f"\nInput  shape : {model.input_shape}")
    print(f"Output shape : {model.output_shape}")
    print(f"Parameters   : {model.count_params():,}")
    return model


# ─────────────────────────────────────────────
# SAMPLE WEIGHTS HELPER
# ─────────────────────────────────────────────

def _build_sample_weights(Y: np.ndarray) -> np.ndarray:
    """
    Build a per-pixel sample-weight array to compensate for the
    tumor/background class imbalance inside each slice.

    Formula
    -------
        weight_i = total_pixels / (n_classes × count_i)

    Returns
    -------
    np.ndarray  float32, shape (N, H, W)  — one weight per pixel
    """
    tumor_px  = np.sum(Y == 1.0)
    bg_px     = np.sum(Y == 0.0)
    total_px  = tumor_px + bg_px

    w0 = total_px / (2.0 * bg_px)      # background weight
    w1 = total_px / (2.0 * tumor_px)   # tumor weight

    print(f"  Background weight : {w0:.4f}")
    print(f"  Tumor weight      : {w1:.4f}")
    print(f"  Tumor is weighted {w1 / w0:.1f}× more than background")

    weights = np.where(
        Y.squeeze(-1) == 1.0, w1, w0
    ).astype(np.float32)               # (N, H, W)
    return weights


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> tf.keras.callbacks.History:
    """
    Run the full training loop.

    Parameters
    ----------
    model   : compiled Keras model
    X_train : (N_train, H, W, 4) float32
    Y_train : (N_train, H, W, 1) float32 binary masks
    X_test  : (N_test,  H, W, 4) float32
    Y_test  : (N_test,  H, W, 1) float32 binary masks

    Returns
    -------
    tf.keras.callbacks.History
    """
    print("\nComputing per-pixel class weights...")
    train_weights = _build_sample_weights(Y_train)
    val_weights   = _build_sample_weights(Y_test)    # ← fix: val also weighted

    callbacks = build_callbacks()

    print("\n" + "=" * 55)
    print("Starting training")
    print(f"  Epochs      : {EPOCHS}  (early stopping @ patience=12)")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Train shape : {X_train.shape}")
    print(f"  Val   shape : {X_test.shape}")
    print("=" * 55 + "\n")

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test, val_weights),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        sample_weight=train_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Load the best checkpoint explicitly in case training finished
    # naturally (EarlyStopping.restore_best_weights only triggers on stop).
    print("\nLoading best saved weights from checkpoint...")
    model.load_weights(CHECKPOINT_PATH)
    print("✅ Best weights restored.\n")

    # Print best epoch summary
    best_epoch = int(np.argmax(history.history["val_dice_coef"]))
    print("=" * 55)
    print(f"Best epoch     : {best_epoch + 1}")
    print(f"Val Dice       : {history.history['val_dice_coef'][best_epoch]:.4f}")
    print(f"Val IoU        : {history.history['val_iou_metric'][best_epoch]:.4f}")
    print(f"Val Precision  : {history.history['val_precision_metric'][best_epoch]:.4f}")
    print(f"Val Recall     : {history.history['val_recall_metric'][best_epoch]:.4f}")
    print(f"Val Loss       : {history.history['val_loss'][best_epoch]:.4f}")
    print("=" * 55)

    return history


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> None:
    """
    Run full test-set evaluation: Keras metrics + HD95.

    Parameters
    ----------
    model  : trained Keras model (best weights already loaded)
    X_test : (N, H, W, 4)
    Y_test : (N, H, W, 1)
    """
    print("\nRunning full test set evaluation...\n")
    results = model.evaluate(X_test, Y_test,
                             batch_size=BATCH_SIZE, verbose=1)

    print("\nTest set results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name:25s}: {value:.4f}")

    compute_hd95_dataset(model, X_test, Y_test,
                         num_samples=HD95_SAMPLES)
```
