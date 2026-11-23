"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

FP8 linears for the text encoder of MiniMax-H3 / MiniMax-H3-Image (51 decoder layers: 50.3 GB in bf16).

Every linear of the decoder layers is stored float8_e4m3fn with one scale per output row (about 26 GB). Two ways to run it:

- `matmul="fp8"`: the input is cast to FP8 too (one scale per token) and the product runs on the FP8 tensor cores
  (`torch._scaled_mm`, accumulation in fp32, output in the input dtype). Needs FP8 tensor cores (Ada, Hopper, Blackwell);
  falls back to `"bf16"` elsewhere.
- `matmul="bf16"`: storage only, the weight is cast back to the input dtype for a bf16 product (any GPU).

The vision encoder, the embeddings and the norms stay bf16; the language-model head, never read by MiniMax-H3, is dropped.
"""

import torch
import torch.nn.functional as F
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs: the PyTorch cast below
    triton = None

FP8 = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8).max
USE_TRITON = triton is not None  # one fused kernel for the per-token cast; the PyTorch ops take ~10x longer

if triton is not None:

    @triton.jit
    def _quantize_rows_kernel(x_ptr, q_ptr, s_ptr, K, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.div_rn(tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12), 448.0)
        tl.store(q_ptr + row * K + offs, tl.div_rn(x, scale).to(tl.float8e4nv), mask=mask)
        tl.store(s_ptr + row, scale)


def quantize_rows(x: torch.Tensor):
    """[M, K] -> (float8_e4m3fn [M, K], float32 scale [M, 1]): one scale per row, the row's largest value maps to 448."""
    if USE_TRITON and x.is_cuda and x.stride(-1) == 1:
        M, K = x.shape
        q = torch.empty((M, K), device=x.device, dtype=FP8)
        s = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        if M:
            _quantize_rows_kernel[(M,)](x, q, s, K, x.stride(0), BLOCK=triton.next_power_of_2(K), num_warps=16)
        return q, s
    s = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / FP8_MAX
    return (x.float() / s).to(FP8), s


class Fp8Linear(nn.Module):
    """`nn.Linear` with its weight stored in FP8, one scale per output row."""

    rowwise = None  # whether torch._scaled_mm takes per-row / per-token scales on this GPU (probed on first use)

    def __init__(self, linear: nn.Linear, matmul: str = "fp8"):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else linear.weight.device
        w = linear.weight.detach().to(device, torch.float32)
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / FP8_MAX
        home = linear.weight.device
        self.register_buffer("weight_fp8", (w / scale).to(FP8).to(home))  # [out, in]
        self.register_buffer("scale", scale.t().contiguous().to(home))  # [1, out], float32
        self.bias = linear.bias
        self.matmul = matmul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.matmul != "fp8" or not x.is_cuda or torch.cuda.get_device_capability(x.device) < (8, 9):
            return F.linear(x, self.weight_fp8.to(x.dtype) * self.scale.t().to(x.dtype), self.bias)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x_fp8, x_scale = quantize_rows(x)  # one scale per token
        out = self._mm(x_fp8, x_scale, x.dtype)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], out.shape[-1])

    def _mm(self, x_fp8, x_scale, dtype):
        w = self.weight_fp8.t()  # [in, out], column-major as _scaled_mm wants
        if Fp8Linear.rowwise is not False:
            try:
                out = torch._scaled_mm(x_fp8, w, scale_a=x_scale, scale_b=self.scale, out_dtype=dtype)
                Fp8Linear.rowwise = True
                return out
            except RuntimeError:
                if Fp8Linear.rowwise:
                    raise
                Fp8Linear.rowwise = False  # per-tensor scales only: unit scales in the product, the real ones after it
        one = torch.ones((), device=x_fp8.device)
        out = torch._scaled_mm(x_fp8, w, scale_a=one, scale_b=one, out_dtype=dtype)
        return out.mul_(x_scale.to(dtype)).mul_(self.scale.to(dtype))


def text_encoder_to_fp8(text_encoder, matmul: str = "fp8") -> int:
    """Swap every linear of the text encoder's decoder layers for an `Fp8Linear`, one layer at a time (the bf16 copy of a layer
    is freed before the next is converted), and drop the unused language-model head. Returns the number of linears."""
    if matmul not in ("fp8", "bf16"):
        raise ValueError(f"matmul must be 'fp8' or 'bf16', got {matmul!r}")
    text_encoder.lm_head = nn.Identity()
    count = 0
    for layer in text_encoder.model.language_model.layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Fp8Linear(child, matmul))
                    count += 1
    torch.cuda.empty_cache()
    return count
