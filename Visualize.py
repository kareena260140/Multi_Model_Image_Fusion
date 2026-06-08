
"""
Visualization helpers: sanity plots and training curves.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from configs.config import CURVES_PATH


def plot_sample(
    X_array: np.ndarray,
    Y_array: np.ndarray,
    idx: int | None = None,
) -> None:
    """
    Plot all 4 modalities + ground-truth mask for one sample.

    Parameters
    ----------
    X_array : (N, H, W, 4) float32
    Y_array : (N, H, W, 1) float32
    idx     : sample index; defaults to the middle tumor slice
    """
    if idx is None:
        tumor_indices = [i for i in range(len(Y_array))
                         if np.sum(Y_array[i]) > 0]
        idx = tumor_indices[len(tumor_indices) // 2] if tumor_indices else 0

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    ch_names = ["T1", "T1CE", "T2", "FLAIR", "Mask"]

    for j in range(4):
        axes[j].imshow(X_array[idx, :, :, j], cmap="gray")
        axes[j].set_title(ch_names[j])
        axes[j].axis("off")

    axes[4].imshow(Y_array[idx].squeeze(), cmap="hot")
    axes[4].set_title("Mask")
    axes[4].axis("off")

    plt.suptitle(f"Sample {idx} — all 4 modalities + ground-truth mask",
                 fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    history: tf.keras.callbacks.History,
    save_path: str = CURVES_PATH,
) -> None:
    """
    Plot and save training vs. validation curves for key metrics.

    Parameters
    ----------
    history   : History object returned by model.fit()
    save_path : file path to save the figure (PNG)
    """
    metrics_to_plot = [
        ("dice_coef",        "val_dice_coef",       "Dice Coefficient"),
        ("iou_metric",       "val_iou_metric",       "IoU"),
        ("precision_metric", "val_precision_metric", "Precision"),
        ("recall_metric",    "val_recall_metric",    "Recall"),
        ("loss",             "val_loss",             "Loss"),
    ]

    available = [
        m for m in metrics_to_plot
        if m[0] in history.history and m[1] in history.history
    ]

    fig, axes = plt.subplots(1, len(available),
                             figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    best_epoch = int(np.argmax(history.history["val_dice_coef"]))

    for ax, (train_key, val_key, title) in zip(axes, available):
        train_vals = history.history[train_key]
        val_vals   = history.history[val_key]
        epochs_ran = range(1, len(train_vals) + 1)

        ax.plot(epochs_ran, train_vals, label="Train", linewidth=1.8)
        ax.plot(epochs_ran, val_vals,   label="Val",
                linewidth=1.8, linestyle="--")
        ax.axvline(x=best_epoch + 1, color="red",
                   linestyle=":", linewidth=1.2,
                   label=f"Best (ep {best_epoch + 1})")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Training History — Multi-Stream Attention U-Net", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Training curves saved to {save_path}")


def plot_predictions(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_samples: int = 4,
    threshold: float = 0.5,
) -> None:
    """
    Plot model predictions alongside ground-truth masks.

    Shows T1CE (channel 1) as the background since it best highlights
    the enhancing tumor core.

    Parameters
    ----------
    model       : trained Keras model
    X_test      : (N, H, W, 4)
    Y_test      : (N, H, W, 1)
    num_samples : number of samples to visualize
    threshold   : binary threshold for predicted probability map
    """
    preds = model.predict(X_test[:num_samples], verbose=0)

    fig, axes = plt.subplots(num_samples, 3,
                             figsize=(9, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        t1ce   = X_test[i, :, :, 1]        # T1CE channel
        gt     = Y_test[i].squeeze()
        pred_b = (preds[i].squeeze() > threshold).astype(float)

        axes[i][0].imshow(t1ce, cmap="gray")
        axes[i][0].set_title("T1CE")

        axes[i][1].imshow(gt, cmap="hot")
        axes[i][1].set_title("Ground Truth")

        axes[i][2].imshow(pred_b, cmap="hot")
        axes[i][2].set_title("Prediction")

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("Model Predictions vs Ground Truth", fontsize=13)
    plt.tight_layout()
    plt.show()
```
