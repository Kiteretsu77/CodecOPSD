"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .pipeline_minimax_h3_ref2va import (
    MiniMaxH3CodecOPSDPipeline, add_audio_track, load_reference, map_references, stage_times,
)
from .pipeline_output import MiniMaxH3CodecOPSDPipelineOutput
