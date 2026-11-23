"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .transformer_cosmos3 import Cosmos3CodecOPSDTransformer
from .transformer_h3_flash import H3FlashCodecOPSDTransformer3DModel
from .transformer_minimax_h3 import MiniMaxH3ImageCodecOPSDTransformer2DModel, MiniMaxH3CodecOPSDTransformer3DModel
