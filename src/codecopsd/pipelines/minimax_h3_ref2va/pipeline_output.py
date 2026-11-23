"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from dataclasses import dataclass

import numpy as np
import torch
from diffusers.utils import BaseOutput


@dataclass
class MiniMaxH3CodecOPSDPipelineOutput(BaseOutput):
    """frames: the generated videos ([B, T, H, W, 3] in [0, 1] for `output_type="np"`) or their latents. With
    `audio=True`: audio, the soundtrack (float32 [2, samples] in [-1, 1], stereo; the audio rows for "latent"), and
    sampling_rate (32000); None otherwise."""

    frames: torch.Tensor | np.ndarray | list
    audio: np.ndarray | torch.Tensor | None = None
    sampling_rate: int | None = None
