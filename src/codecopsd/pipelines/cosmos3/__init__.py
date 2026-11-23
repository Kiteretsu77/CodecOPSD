"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.
"""

from .pipeline_cosmos3 import (CONTROLS, TASKS, Cosmos3CodecOPSDPipeline, droid_canvas, fit_frames, negative_text,
                                    prompt_text, read_frames)
