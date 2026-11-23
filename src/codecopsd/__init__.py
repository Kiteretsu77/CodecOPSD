"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .models import (
    Cosmos3CodecOPSDTransformer, MiniMaxH3ImageCodecOPSDTransformer2DModel, MiniMaxH3CodecOPSDTransformer3DModel,
)
from .pipelines import (
    Cosmos3CodecOPSDPipeline, MiniMaxH3ImageCodecOPSDPipeline, MiniMaxH3CodecOPSDPipeline, MiniMaxH3CodecOPSDPipelineOutput,
)
