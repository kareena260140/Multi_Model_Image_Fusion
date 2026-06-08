
"""
Keras callbacks used during training.

Included
--------
WarmupLRScheduler  : linearly ramps LR from near-zero to TARGET_LR
                     over the first WARMUP_EPOCHS epochs
build_callbacks    : returns the full list of callbacks ready for model.fit()
"""

import tensorflow as tf
from configs.config import (
    CHECKPOINT_PATH, CSV_LOG_PATH,
    TARGET_LR, WARMUP_EPOCHS,
    RLROP_FACTOR, RLROP_PATIENCE, RLROP_MIN_LR, RLROP_COOLDOWN,
    ES_PATIENCE, ES_MIN_DELTA,
)


class WarmupLRScheduler(tf.keras.callbacks.Callback):
    """
    Linear learning-rate warmup.

    For the first ``warmup_epochs`` epochs the LR is ramped from
    (target_lr / warmup_epochs) up to target_lr.  After that,
    ReduceLROnPlateau takes over.

    Why warm up?
    ------------
    The multi-stream + CBAM + attention-gate architecture has many
    parameters.  Starting at full LR from epoch 0 risks an unstable
    loss landscape in the first few steps.  Warmup gives BatchNorm
    statistics time to stabilize before the full gradient signal hits.

    Parameters
    ----------
    warmup_epochs : int   number of ramp-up epochs
    target_lr     : float learning rate to reach at the end of warmup
    """

    def __init__(self, warmup_epochs: int = WARMUP_EPOCHS,
                 target_lr: float = TARGET_LR) -> None:
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr     = target_lr

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        if epoch < self.warmup_epochs:
            warmup_lr = self.target_lr * ((epoch + 1) / self.warmup_epochs)
            self.model.optimizer.learning_rate.assign(warmup_lr)
            print(
                f"\n  [Warmup] Epoch {epoch + 1}/{self.warmup_epochs}"
                f" — LR set to {warmup_lr:.2e}"
            )


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """
    Build and return the standard callback list for model.fit().

    Callbacks
    ---------
    1. ModelCheckpoint  — saves best weights based on val_dice_coef
    2. ReduceLROnPlateau— halves LR after 4 stagnant epochs
    3. EarlyStopping    — stops after 12 stagnant epochs
    4. CSVLogger        — logs every epoch to disk
    5. WarmupLRScheduler— linear LR ramp for first WARMUP_EPOCHS

    Returns
    -------
    list[tf.keras.callbacks.Callback]
    """
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath       = CHECKPOINT_PATH,
        monitor        = "val_dice_coef",
        mode           = "max",
        save_best_only = True,
        verbose        = 1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_dice_coef",
        mode     = "max",
        factor   = RLROP_FACTOR,
        patience = RLROP_PATIENCE,
        min_lr   = RLROP_MIN_LR,
        cooldown = RLROP_COOLDOWN,
        verbose  = 1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor              = "val_dice_coef",
        mode                 = "max",
        patience             = ES_PATIENCE,
        min_delta            = ES_MIN_DELTA,
        restore_best_weights = True,
        verbose              = 1,
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        CSV_LOG_PATH,
        append=False,
    )

    warmup = WarmupLRScheduler(
        warmup_epochs = WARMUP_EPOCHS,
        target_lr     = TARGET_LR,
    )

    print("✅ Callbacks ready.")
    print(f"   Checkpoint → {CHECKPOINT_PATH}")
    print(f"   CSV log    → {CSV_LOG_PATH}")

    return [checkpoint, reduce_lr, early_stop, csv_logger, warmup]
```
