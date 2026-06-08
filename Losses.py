
"""
Loss functions and segmentation metrics for the BraTS pipeline.

Losses (used during training)
------------------------------
dice_loss               : Sørensen–Dice loss
tversky_loss            : Generalised Dice; independently weights FP and FN
focal_tversky_loss      : Tversky loss scaled by a focusing exponent γ
binary_focal_loss       : Per-pixel focal cross-entropy
combined_loss           : 0.8 × FocalTversky + 0.2 × BinaryFocal  ← primary loss

Metrics (monitored during training)
-------------------------------------
dice_coef               : Hard-threshold Dice (F1)
iou_metric              : Jaccard index
precision_metric        : TP / (TP + FP)
recall_metric           : TP / (TP + FN)  — most important for medical seg.
specificity_metric      : TN / (TN + FP)

Post-training evaluation
-------------------------
compute_hd95_single     : HD95 for one mask pair
compute_hd95_dataset    : HD95 statistics over the test set
"""

import numpy as np
import tensorflow as tf

from configs.config import (
    FOCAL_TVERSKY_WEIGHT, BINARY_FOCAL_WEIGHT,
    TVERSKY_ALPHA, TVERSKY_BETA, TVERSKY_GAMMA,
    FOCAL_GAMMA, FOCAL_ALPHA,
)


# ─────────────────────────────────────────────
# LOSSES
# ─────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    """
    Sørensen–Dice Loss.

    Math:
        Dice = (2 · |X ∩ Y|) / (|X| + |Y|)
        Loss = 1 − Dice
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (denom + smooth)


def tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = TVERSKY_ALPHA,
    beta: float  = TVERSKY_BETA,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Tversky Loss — generalization of Dice.

    Math:
        TI   = TP / (TP + α·FP + β·FN)
        Loss = 1 − TI

    Parameters
    ----------
    alpha : FP penalty weight  (lower → more tolerant of false alarms)
    beta  : FN penalty weight  (higher → heavily penalise missed tumors)

    When α = β = 0.5  →  Tversky reduces to Dice.

    Reference: Salehi et al., MICCAI 2017.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FP = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    ti = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1.0 - ti


def focal_tversky_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float  = TVERSKY_ALPHA,
    beta: float   = TVERSKY_BETA,
    gamma: float  = TVERSKY_GAMMA,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Focal Tversky Loss.

    Math:
        FTL = (1 − TI)^γ

    γ < 1 : more gradient from easy examples
    γ = 1 : standard Tversky Loss
    γ > 1 : gradient dominated by hard examples (small tumors, fuzzy edges)

    We use γ = 0.75 so the model focuses on hard-to-segment regions
    without completely ignoring easy ones.

    Reference: Abraham & Khan, ISBI 2019.
    """
    tv = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta, smooth=smooth)
    return tf.pow(tv, gamma)


def binary_focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    gamma: float    = FOCAL_GAMMA,
    alpha_fl: float = FOCAL_ALPHA,
) -> tf.Tensor:
    """
    Binary Focal Loss.

    Math:
        FL = −α · (1 − p_t)^γ · log(p_t)

    Down-weights confident correct predictions so the gradient focuses
    on uncertain, hard pixels near the boundary.

    Reference: Lin et al., "Focal Loss for Dense Object Detection",
               ICCV 2017.
    """
    y_true   = tf.cast(y_true, tf.float32)
    y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    p_t      = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
    alpha_t  = y_true * alpha_fl + (1.0 - y_true) * (1.0 - alpha_fl)
    focal_w  = alpha_t * tf.pow(1.0 - p_t, gamma)
    return tf.reduce_mean(-focal_w * tf.math.log(p_t))


def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Primary training loss:
        L = 0.8 · FocalTverskyLoss + 0.2 · BinaryFocalLoss

    FocalTversky handles region-level FP/FN balance (global overlap).
    BinaryFocal sharpens boundaries and focuses on hard individual pixels.
    The 80/20 split keeps overall region quality as the primary objective.
    """
    ftl = focal_tversky_loss(y_true, y_pred)
    bfl = binary_focal_loss(y_true, y_pred)
    return FOCAL_TVERSKY_WEIGHT * ftl + BINARY_FOCAL_WEIGHT * bfl


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def dice_coef(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Dice Coefficient (hard threshold @ 0.5).

    Math: Dice = (2·TP) / (2·TP + FP + FN)
    """
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def iou_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Intersection over Union (Jaccard Index).

    Math: IoU = TP / (TP + FP + FN)
    Always ≤ Dice for the same prediction.
    """
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union        = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Precision = TP / (TP + FP)
    High precision → few false alarms.
    """
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FP = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (TP + smooth) / (TP + FP + smooth)


def recall_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Recall (Sensitivity) = TP / (TP + FN)
    High recall → few missed tumors.

    This is the most clinically important metric — missing a tumor
    is far more dangerous than a false alarm.
    The Tversky loss uses β=0.7 > α=0.3 for the same reason.
    """
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    return (TP + smooth) / (TP + FN + smooth)


def specificity_metric(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    smooth: float = 1e-6,
) -> tf.Tensor:
    """
    Specificity = TN / (TN + FP)
    High specificity → correctly labels most background pixels.
    """
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f   = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    TN = tf.reduce_sum((1.0 - y_true_f) * (1.0 - y_pred_f))
    FP = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (TN + smooth) / (TN + FP + smooth)


# ─────────────────────────────────────────────
# HD95  (post-training only — not differentiable)
# ─────────────────────────────────────────────

def compute_hd95_single(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
) -> float:
    """
    95th-percentile Hausdorff Distance for a single 2D mask pair.

    HD95 measures the 95th percentile of bidirectional surface distances
    between the predicted and ground-truth boundaries.

    Units : pixels (multiply by voxel spacing in mm for clinical output)
    Returns 0.0 if either mask is empty (HD95 is undefined).

    Parameters
    ----------
    pred_mask : 2D numpy array, float or bool
    true_mask : 2D numpy array, float or bool
    """
    import scipy.ndimage as ndi
    from scipy.ndimage import distance_transform_edt

    pred_bin = (pred_mask > 0.5).astype(bool)
    true_bin = (true_mask > 0.5).astype(bool)

    if not pred_bin.any() or not true_bin.any():
        return 0.0

    pred_border = pred_bin ^ ndi.binary_erosion(pred_bin)
    true_border = true_bin ^ ndi.binary_erosion(true_bin)

    d_pred_to_true = distance_transform_edt(~true_border)[pred_border]
    d_true_to_pred = distance_transform_edt(~pred_border)[true_border]

    all_distances = np.concatenate([d_pred_to_true, d_true_to_pred])
    return float(np.percentile(all_distances, 95))


def compute_hd95_dataset(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_samples: int = 200,
) -> np.ndarray:
    """
    Compute and print HD95 statistics over ``num_samples`` test images.

    Parameters
    ----------
    model       : trained Keras model
    X_test      : (N, H, W, 4) float32 array
    Y_test      : (N, H, W, 1) float32 binary masks
    num_samples : number of test samples to evaluate

    Returns
    -------
    np.ndarray  HD95 values, shape (num_samples,)
    """
    preds = model.predict(X_test[:num_samples], verbose=0)
    hd95_scores = np.array([
        compute_hd95_single(preds[i].squeeze(), Y_test[i].squeeze())
        for i in range(num_samples)
    ])

    print(f"\n📐 Hausdorff Distance 95 (HD95) — {num_samples} test samples:")
    print(f"   Mean   : {hd95_scores.mean():.4f} px")
    print(f"   Median : {np.median(hd95_scores):.4f} px")
    print(f"   Std    : {hd95_scores.std():.4f} px")
    print(f"   Best   : {hd95_scores.min():.4f} px")
    print(f"   Worst  : {hd95_scores.max():.4f} px")
    return hd95_scores


# ─────────────────────────────────────────────
# SANITY CHECK
# ─────────────────────────────────────────────

def run_loss_sanity_check() -> None:
    """
    Verify all losses and metrics execute on a random batch.
    Call this before model.compile() to catch import or shape errors early.
    """
    print("Running loss/metric sanity check...")
    _y_true = tf.cast(
        tf.random.uniform((4, 128, 128, 1), minval=0, maxval=2, dtype=tf.int32),
        tf.float32,
    )
    _y_pred = tf.random.uniform((4, 128, 128, 1))

    checks = {
        "dice_loss"          : dice_loss,
        "tversky_loss"       : tversky_loss,
        "focal_tversky_loss" : focal_tversky_loss,
        "binary_focal_loss"  : binary_focal_loss,
        "combined_loss"      : combined_loss,
        "dice_coef"          : dice_coef,
        "iou_metric"         : iou_metric,
        "precision_metric"   : precision_metric,
        "recall_metric"      : recall_metric,
        "specificity_metric" : specificity_metric,
    }

    for name, fn in checks.items():
        val = fn(_y_true, _y_pred).numpy()
        print(f"  {name:25s}: {val:.4f}")

    print("\n✅ All losses and metrics verified.")
```
