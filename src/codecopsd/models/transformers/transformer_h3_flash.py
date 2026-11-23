"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

HyperFlow H3 Flash (Video Rebirth's compressed, few-step MiniMax-H3 Ref2VA transformer, `hyperflow_flash/`) with the
CodecOPSD condition LoRA on the reference rows."""

import torch

from .hyperflow_flash import HyperFlowH3FlashTransformer3DModel, set_ff_chunk, set_lowrank_rank_multiple
from .hyperflow_flash.int8 import set_kernel_options
from .transformer_minimax_h3 import ConditionLoRAMixin


def int8_gemm_choice(capability):
    """The INT8 GEMM kernel CodecOPSD sets for Flash on a GPU of compute capability `capability` ((major, minor)), or
    None to keep Flash's own choice (`hyperflow_flash.int8.resolve_int8_gemm`):

        "triton"   RTX Blackwell (12: RTX 50 series, RTX PRO), where Flash would pick cuBLASLt
        "cublas"   RTX 30 and RTX 40 series (8.6, 8.9), where Flash would pick its Triton kernel

    On these GPUs the chosen kernel gives the same output as Flash's choice and runs faster."""
    if capability[0] == 12:
        return "triton"
    if tuple(capability) in ((8, 6), (8, 9)):
        return "cublas"
    return None


class H3FlashCodecOPSDTransformer3DModel(ConditionLoRAMixin, HyperFlowH3FlashTransformer3DModel):
    r"""HyperFlow H3 Flash (Ref2VA) that runs a CodecOPSD condition LoRA on the reference rows.

    The CodecOPSD LoRA was trained on the released MiniMax-H3 and is used as it is: it wraps the same projections of every
    block Flash keeps (bf16 low-rank, INT8 or bf16 dense linears alike) and skips block 9, which Flash removed (290 of
    the 296 wrapped projections). Every forward needs `endpoint_timestep` (Flash's two-time conditioning); the packed
    layout is the CodecOPSD one, `[text | target audio | target video | references]`, with the LoRA on rows
    `reference_start:`.
    """

    def lora_dtype(self):
        return torch.bfloat16  # the LoRA runs in bf16 next to Flash's INT8 and bf16 low-rank layers

    @classmethod
    def from_flash(cls, folder, torch_dtype=torch.bfloat16, rank_multiple=8):
        """The transformer from a downloaded Flash folder (`ref2va_int8_convrot/` of `VideoRebirthAI/hyperflow-h3-flash`).

        As Flash's own pipeline loads it, with one change: the feed-forwards run over the whole sequence instead of
        16,384-token chunks, so that the LoRA's row indices (positions in the whole sequence) stay exact. Low-rank
        ranks are padded to multiples of `rank_multiple` (Flash's default 8; zero padding, the same function).

        The INT8 GEMM kernel follows `int8_gemm_choice`: Flash's Triton kernel on RTX Blackwell (bit-identical output,
        1.03x the denoising speed and 0.7 GB less peak memory on an RTX PRO 6000) and cuBLAS (`torch._int_mm`) on the
        RTX 30 and RTX 40 series (bit-identical output, about 1.04x the denoising speed on an RTX 3090 / 4090)."""
        transformer = cls.from_pretrained(folder, torch_dtype=torch_dtype)
        set_lowrank_rank_multiple(transformer, rank_multiple)
        set_ff_chunk(transformer, 0)
        choice = int8_gemm_choice(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        if choice is not None:
            set_kernel_options(int8_gemm=choice)
        return transformer.eval()
