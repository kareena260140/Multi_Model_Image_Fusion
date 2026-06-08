
"""
Multi-Stream Attention U-Net with CBAM.

Architecture overview
---------------------
Input  (B, 128, 128, 4) — T1 / T1CE / T2 / FLAIR stacked on channel axis

1. Shared Encoder (3 levels)
   Each level: conv_block → CBAM → MaxPool
   Produces skip connections c1 (64×64×32), c2 (32×32×64), c3 (16×16×128)

2. Per-Modality Mini-Encoders (×4)
   Each modality channel gets its own lightweight 3-level CNN
   → feature maps at (16×16×64)

3. Spatially-Adaptive Late Fusion  ← fixed from original scalar weighting
   Concatenate all 4 modality maps → (16×16×256)
   Apply 1×1 Conv (sigmoid) to produce a spatial attention map per modality
   Fused = weighted sum over modalities, spatially adaptive per pixel

4. Bottleneck
   Concat(shared_encoder_output, fused) → conv_block + CBAM

5. Decoder (3 levels)
   Each level: ConvTranspose → AttentionGate(skip) → Concat → conv_block + CBAM

6. Output  Conv2D(1, sigmoid) → (B, 128, 128, 1)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Conv2DTranspose, Concatenate,
    GlobalAveragePooling2D, Lambda,
)
from tensorflow.keras.models import Model

from configs.config import IMG_SIZE, NUM_CHANNELS
from models.blocks import conv_block, cbam_block, attention_gate


def _mini_encoder(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """
    Lightweight 3-level encoder for a single modality channel.

    Downsamples the (128×128×1) modality input to (16×16, filters*4).
    Each level doubles the filter count.

    Parameters
    ----------
    x       : single-channel input  (B, 128, 128, 1)
    filters : base filter count (output is filters*4)
    name    : name prefix for all layers

    Returns
    -------
    tf.Tensor  (B, 16, 16, filters*4)
    """
    x = Conv2D(filters, 3, padding="same",
               kernel_initializer="he_normal", name=f"{name}_c1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(name=f"{name}_p1")(x)           # → 64×64

    x = Conv2D(filters * 2, 3, padding="same",
               kernel_initializer="he_normal", name=f"{name}_c2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(name=f"{name}_p2")(x)           # → 32×32

    x = Conv2D(filters * 4, 3, padding="same",
               kernel_initializer="he_normal", name=f"{name}_c3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D(name=f"{name}_p3")(x)           # → 16×16

    return x  # (B, 16, 16, filters*4)


def build_multistream_attention_unet(
    img_size: int = IMG_SIZE,
    num_channels: int = NUM_CHANNELS,
) -> Model:
    """
    Build and return the Multi-Stream Attention U-Net.

    Parameters
    ----------
    img_size     : spatial size of the square input (default 128)
    num_channels : number of input MRI modalities (default 4)

    Returns
    -------
    tf.keras.Model
        Compiled-ready model. Call model.compile() separately.
    """
    inputs = Input((img_size, img_size, num_channels), name="input")

    # ──────────────────────────────────────────────────────
    # SPLIT INTO MODALITY CHANNELS
    # ──────────────────────────────────────────────────────
    T1    = Lambda(lambda t: t[:, :, :, 0:1], name="split_T1"   )(inputs)
    T1CE  = Lambda(lambda t: t[:, :, :, 1:2], name="split_T1CE" )(inputs)
    T2    = Lambda(lambda t: t[:, :, :, 2:3], name="split_T2"   )(inputs)
    FLAIR = Lambda(lambda t: t[:, :, :, 3:4], name="split_FLAIR")(inputs)

    # ──────────────────────────────────────────────────────
    # SHARED ENCODER  (operates on all 4 channels jointly)
    # Produces skip connections used in the decoder.
    # ──────────────────────────────────────────────────────
    # Level 1  →  128×128×32
    c1 = conv_block(inputs, 32)
    c1 = cbam_block(c1, reduction_ratio=4)
    p1 = MaxPooling2D()(c1)                          # 64×64×32

    # Level 2  →  64×64×64
    c2 = conv_block(p1, 64)
    c2 = cbam_block(c2, reduction_ratio=8)
    p2 = MaxPooling2D()(c2)                          # 32×32×64

    # Level 3  →  32×32×128
    c3 = conv_block(p2, 128)
    c3 = cbam_block(c3, reduction_ratio=8)
    p3 = MaxPooling2D()(c3)                          # 16×16×128

    # ──────────────────────────────────────────────────────
    # PER-MODALITY MINI ENCODERS
    # Each modality has its own 3-level encoder so the network
    # can learn complementary low-level representations.
    #   base filters = 16  →  output channels = 64 per modality
    # ──────────────────────────────────────────────────────
    feat_T1    = _mini_encoder(T1,    16, "T1_stream")    # (B,16,16,64)
    feat_T1CE  = _mini_encoder(T1CE,  16, "T1CE_stream")  # (B,16,16,64)
    feat_T2    = _mini_encoder(T2,    16, "T2_stream")    # (B,16,16,64)
    feat_FLAIR = _mini_encoder(FLAIR, 16, "FLAIR_stream") # (B,16,16,64)

    # ──────────────────────────────────────────────────────
    # SPATIALLY-ADAPTIVE LATE FUSION  ← key fix
    #
    # Original code used a scalar GlobalAvgPool → Dense(4) → softmax,
    # which produced the SAME weight for every spatial location.
    # A tumour in the upper-left corner got the same T1CE weight as
    # a background region in the lower-right — defeating the purpose.
    #
    # Fix: use a 1×1 Conv to produce a spatial attention map (H×W×1)
    # for each modality independently.  This lets the network learn
    # "T1CE matters at position (r,c)" locally, not globally.
    #
    # Math (for one modality m):
    #   α_m = sigmoid( W_m * concat_all_modalities )  ∈ (0,1)^{H×W}
    #   fused = Σ_m  α_m ⊙ feat_m
    # ──────────────────────────────────────────────────────
    modal_concat = Concatenate(name="modal_concat")(
        [feat_T1, feat_T1CE, feat_T2, feat_FLAIR]
    )                                                    # (B,16,16,256)

    # One spatial attention map per modality via 1×1 Conv + sigmoid
    att_T1    = Conv2D(1, 1, padding="same", activation="sigmoid",
                       name="att_T1"   )(modal_concat)   # (B,16,16,1)
    att_T1CE  = Conv2D(1, 1, padding="same", activation="sigmoid",
                       name="att_T1CE" )(modal_concat)
    att_T2    = Conv2D(1, 1, padding="same", activation="sigmoid",
                       name="att_T2"   )(modal_concat)
    att_FLAIR = Conv2D(1, 1, padding="same", activation="sigmoid",
                       name="att_FLAIR")(modal_concat)

    # Weighted sum of modality features  →  (B,16,16,64)
    from tensorflow.keras.layers import Multiply, Add
    fused = Add(name="weighted_fusion")([
        Multiply()([feat_T1,    att_T1]),
        Multiply()([feat_T1CE,  att_T1CE]),
        Multiply()([feat_T2,    att_T2]),
        Multiply()([feat_FLAIR, att_FLAIR]),
    ])

    # ──────────────────────────────────────────────────────
    # BOTTLENECK
    # Concatenate shared encoder output (16×16×128) with
    # fused modality features (16×16×64) → (16×16×192)
    # ──────────────────────────────────────────────────────
    bottleneck_in = Concatenate(name="bottleneck_concat")([p3, fused])
    b1 = conv_block(bottleneck_in, 256)
    b1 = cbam_block(b1, reduction_ratio=16)              # (B,16,16,256)

    # ──────────────────────────────────────────────────────
    # DECODER  (3 up-sampling levels)
    # Each level: ConvTranspose → AttentionGate(skip) → Concat → conv + CBAM
    # ──────────────────────────────────────────────────────

    # Level 1: 16→32
    u1 = Conv2DTranspose(128, 2, strides=2, padding="same")(b1)  # (B,32,32,128)
    c3_att = attention_gate(g=u1, x=c3, filters=64)
    u1 = Concatenate()([u1, c3_att])
    d1 = conv_block(u1, 128)
    d1 = cbam_block(d1, reduction_ratio=8)                        # (B,32,32,128)

    # Level 2: 32→64
    u2 = Conv2DTranspose(64, 2, strides=2, padding="same")(d1)   # (B,64,64,64)
    c2_att = attention_gate(g=u2, x=c2, filters=32)
    u2 = Concatenate()([u2, c2_att])
    d2 = conv_block(u2, 64)
    d2 = cbam_block(d2, reduction_ratio=8)                        # (B,64,64,64)

    # Level 3: 64→128
    u3 = Conv2DTranspose(32, 2, strides=2, padding="same")(d2)   # (B,128,128,32)
    c1_att = attention_gate(g=u3, x=c1, filters=16)
    u3 = Concatenate()([u3, c1_att])
    d3 = conv_block(u3, 32)
    d3 = cbam_block(d3, reduction_ratio=4)                        # (B,128,128,32)

    # ──────────────────────────────────────────────────────
    # OUTPUT
    # ──────────────────────────────────────────────────────
    outputs = Conv2D(1, 1, activation="sigmoid", name="output")(d3)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="MultiStream_Attention_UNet_CBAM",
    )
    return model
```
