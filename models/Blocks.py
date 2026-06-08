
"""
Reusable Keras building blocks for the Multi-Stream Attention U-Net.

Blocks
------
conv_block        : double Conv2D + BN + ReLU + Dropout
cbam_block        : Channel + Spatial attention (CBAM, Woo et al. ECCV 2018)
attention_gate    : Soft attention on skip connections (Oktay et al. 2018)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
    Reshape, Dense, Multiply, Add, Concatenate,
    Dropout, Lambda,
)
import tensorflow.keras.backend as K

from configs.config import DROPOUT_RATE, CBAM_REDUCTION


# ─────────────────────────────────────────────
# DOUBLE-CONV BLOCK
# ─────────────────────────────────────────────

def conv_block(
    x: tf.Tensor,
    filters: int,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.Tensor:
    """
    Standard double-conv block: Conv → BN → ReLU → Dropout → Conv → BN → ReLU.

    He initialization is used because the network uses ReLU activations
    throughout; He init keeps variance stable across layers.

    Parameters
    ----------
    x           : input tensor
    filters     : number of Conv2D output filters
    dropout_rate: spatial dropout applied between the two convolutions

    Returns
    -------
    tf.Tensor of shape (B, H, W, filters)
    """
    x = Conv2D(filters, 3, padding="same",
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, 3, padding="same",
               kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


# ─────────────────────────────────────────────
# CBAM BLOCK
# ─────────────────────────────────────────────

def cbam_block(
    x: tf.Tensor,
    reduction_ratio: int = CBAM_REDUCTION,
) -> tf.Tensor:
    """
    Convolutional Block Attention Module (CBAM).

    Applies channel attention then spatial attention sequentially.

    Channel Attention
    -----------------
    Given feature map F (H × W × C):
        Mc = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
    Tells the network *which* channels to emphasize.

    Spatial Attention
    -----------------
    Given channel-refined F':
        Ms = σ( Conv7×7( [AvgPool_c(F'), MaxPool_c(F')] ) )
    Tells the network *where* to look.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module",
               ECCV 2018.

    Parameters
    ----------
    x               : input tensor (B, H, W, C)
    reduction_ratio : channel reduction factor inside the MLP

    Returns
    -------
    tf.Tensor of shape (B, H, W, C)
    """
    channels = x.shape[-1]

    # ── Channel Attention ──────────────────────────────────
    avg_pool = GlobalAveragePooling2D()(x)          # (B, C)
    max_pool = GlobalMaxPooling2D()(x)              # (B, C)

    avg_pool = Reshape((1, 1, channels))(avg_pool)
    max_pool = Reshape((1, 1, channels))(max_pool)

    # Shared two-layer MLP: C → C//r → C
    bottleneck = max(1, channels // reduction_ratio)
    dense1 = Dense(bottleneck, activation="relu",
                   kernel_initializer="he_normal", use_bias=False)
    dense2 = Dense(channels,
                   kernel_initializer="he_normal", use_bias=False)

    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))

    channel_att = Activation("sigmoid")(Add()([avg_out, max_out]))  # (B,1,1,C)
    x = Multiply()([x, channel_att])

    # ── Spatial Attention ──────────────────────────────────
    avg_spatial = Lambda(
        lambda t: K.mean(t, axis=-1, keepdims=True)
    )(x)                                             # (B, H, W, 1)
    max_spatial = Lambda(
        lambda t: K.max(t, axis=-1, keepdims=True)
    )(x)                                             # (B, H, W, 1)

    spatial_cat = Concatenate()([avg_spatial, max_spatial])  # (B, H, W, 2)
    spatial_att = Conv2D(
        1, kernel_size=7, padding="same",
        activation="sigmoid",
        kernel_initializer="glorot_uniform",
        use_bias=False,
    )(spatial_cat)                                   # (B, H, W, 1)

    x = Multiply()([x, spatial_att])
    return x


# ─────────────────────────────────────────────
# ATTENTION GATE
# ─────────────────────────────────────────────

def attention_gate(
    g: tf.Tensor,
    x: tf.Tensor,
    filters: int,
) -> tf.Tensor:
    """
    Attention Gate for U-Net skip connections.

    Produces a soft spatial attention map α ∈ (0, 1) from the
    gating signal ``g`` (decoder) and skip tensor ``x`` (encoder),
    then returns the attended skip features ``x ⊙ α``.

    Math
    ----
        q_g = W_g · g + b_g
        q_x = W_x · x + b_x
        ψ   = σ( W_ψ · ReLU(q_g + q_x) + b_ψ )   ← attention map
        out = x ⊙ ψ

    Reference: Oktay et al., "Attention U-Net: Learning Where to Look
               for the Pancreas", MIDL 2018.

    Parameters
    ----------
    g       : gating signal from decoder  (B, H, W, C_g)
    x       : skip connection from encoder (B, H, W, C_x)
    filters : intermediate filter count for the projection layers

    Returns
    -------
    tf.Tensor  attended skip features, same shape as ``x``
    """
    g_proj = Conv2D(filters, 1, padding="same",
                    kernel_initializer="he_normal")(g)
    g_proj = BatchNormalization()(g_proj)

    x_proj = Conv2D(filters, 1, padding="same",
                    kernel_initializer="he_normal")(x)
    x_proj = BatchNormalization()(x_proj)

    combined = Add()([g_proj, x_proj])
    combined = Activation("relu")(combined)

    psi = Conv2D(1, 1, padding="same",
                 activation="sigmoid",
                 kernel_initializer="glorot_uniform")(combined)  # (B, H, W, 1)

    return Multiply()([x, psi])
```
