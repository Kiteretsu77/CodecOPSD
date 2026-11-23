"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Attention kernel of the diffusion transformer (README: Acceleration).

Every choice computes exact attention; they differ in speed and in the GPUs that run them. The VAE stays on PyTorch's
SDPA: diffusers makes the last backend set the default of every model, and the VAE attends in float32, which the
FlashAttention kernels do not accept.
"""

BACKENDS = {
    "auto": None,  # default: cudnn on Hopper / Blackwell (H100, H200, B200), sdpa on older GPUs (A100, where cudnn is slower)
    "sdpa": "native",  # PyTorch scaled_dot_product_attention with its own kernel choice; every GPU
    "cudnn": "_native_cudnn",  # PyTorch SDPA, the cuDNN kernel; every GPU with a recent cuDNN
    "flash3": "_flash_3",  # FlashAttention 3 (Hopper: H100 / H200), needs the flash_attn_interface package
    "flash3-hub": "_flash_3_hub",  # FlashAttention 3 from the Hugging Face kernels hub (Hopper), needs `kernels`
}


def auto_backend():
    """The backend "auto" picks on the first GPU: "cudnn" from compute capability 9 (Hopper, Blackwell), "sdpa" below
    (A100, RTX 3090 / 4090, where cuDNN attention is slower) and without a GPU."""
    import torch
    return "cudnn" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9 else "sdpa"


def set_attention(pipe, name):
    """Run `pipe.transformer`'s attention with backend `name` (a key of `BACKENDS`); returns the backend used."""
    if name not in BACKENDS:
        raise ValueError(f"attention must be one of {sorted(BACKENDS)}, got {name!r}")
    if name == "auto":
        name = auto_backend()
    pipe.transformer.set_attention_backend(BACKENDS[name])
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "set_attention_backend"):
        vae.set_attention_backend("native")
    return name
