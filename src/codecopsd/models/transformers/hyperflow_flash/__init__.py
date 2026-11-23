"""HyperFlow H3 Flash transformer code, vendored unchanged from https://github.com/Video-Rebirth/hyperflow-h3-flash
(commit dc67ea77de7795a570a41f98dc97e48f66a5828b, `src/hyperflow_h3_flash/transformer.py` and `int8.py`), licensed
under the Apache License, Version 2.0 (`LICENSE` in this folder). The same two files ship with the weights
(`VideoRebirthAI/hyperflow-h3-flash`, folder `ref2va_int8_convrot/`).

HyperFlow H3 Flash is Video Rebirth's compressed, few-step version of the MiniMax-H3 transformer: block 9 of the
Ref2VA transformer removed, 114 linears factored into low-rank pairs, 180 linears in W8A8 INT8 behind a Hadamard
rotation (Triton or cuBLASLt kernels), a 16-dimensional AdaLN, and two-time (t, r) conditioning for 4, 6 or 8 steps
without guidance. 13.9B parameters, 16.3 GB on disk.
"""

from .transformer import (
    DroppedBlock,
    HyperFlowH3FlashTransformer3DModel,
    LowRankLinear,
    set_ff_chunk,
    set_lowrank_rank_multiple,
)
from .int8 import Int8RotLinear
