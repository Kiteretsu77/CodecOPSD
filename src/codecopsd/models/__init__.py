"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .fp8_linear import Fp8Linear, text_encoder_to_fp8
from .transformers import (
    Cosmos3CodecOPSDTransformer, H3FlashCodecOPSDTransformer3DModel, MiniMaxH3ImageCodecOPSDTransformer2DModel,
    MiniMaxH3CodecOPSDTransformer3DModel,
)
