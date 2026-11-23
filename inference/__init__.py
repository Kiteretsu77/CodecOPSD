"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Inference from the command line, run from the repository root.

    python inference/generate.py minimax_h3_ref2va --prompt-file prompt.txt --reference a.png --reference b.mp4

    generate.py     the command line of every model
    load.py         the models, building a pipeline, offload (shared)
    utils.py        the dropped-token count, the run summary lines (shared)
    cli/            one module per model: its flags and one run
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # same output; the cache stays near the memory in use
