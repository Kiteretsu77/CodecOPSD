"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Inference: the command line and the Gradio demos, run from the repository root.

    python inference/generate.py minimax_h3_ref2va      # command line: a bundled example or your own inputs
    python app.py ref2va                                # a demo page: ref2va, ref2va_extreme or ref2image

    generate.py     the command line of every model
    ../app.py       the demo pages (at the repository root)
    load.py         the models, building a pipeline, offload (shared)
    extreme.py      the 12 to 32 GB path only (HyperFlow H3 Flash): GPU memory plans, its setup
    hardware.py     the machine: GPU, CPU memory, kernels, warnings (python -m inference.hardware)
    vlm_fp8.py      save the FP8 VLM once from the bf16 release (python -m inference.vlm_fp8; a fallback)
    examples.py     the bundled examples: their folders, example.json and flags (shared)
    utils.py        the dropped-token count, the run summary lines (shared)
    cli/            one module per model: its flags and one run
    demo/           the Gradio demo, one module per part (demo/__init__.py lists them)
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # same output; the cache stays near the memory in use
