"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .cosmos3 import Cosmos3CodecOPSDPipeline
from .minimax_h3_image import MiniMaxH3ImageCodecOPSDPipeline
from .minimax_h3_ref2va import MiniMaxH3CodecOPSDPipeline, MiniMaxH3CodecOPSDPipelineOutput
