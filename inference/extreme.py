"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The extreme path only: HyperFlow H3 Flash (Video Rebirth's compressed MiniMax H3 transformer: 13.9B parameters, INT8
and low rank, 16 GB, its own 6 steps) with the CodecOPSD LoRA on a 12, 16, 24 or 32 GB GPU. Both entry points of that
path call `setup` (`python -m inference.hardware` reads the plans too); the normal path never imports this file.

    python app.py ref2va_extreme                         # the demo page (the page itself: app.py)
    python inference/generate.py h3_flash_ref2va         # the command line (--sampler flash)

    GPU_PLANS, gpu_plan, memory_plan    the GPU memory plans (README: Consumer GPUs)
    PINNED_RAM_GB                       the CPU memory the pinned plans keep
    PIN_VLM_RAM_GB                      the CPU memory from which `--pin-vlm auto` page-locks an FP8 VLM folder
    flash_weights                       the default HyperFlow H3 Flash folder
    setup                               the process capped, the plan picked and printed with the hardware report
    taeh3_weights, use_taeh3_encoder    the references encoded by TAEH3 instead of the released VAE encoder

Kept elsewhere on purpose: the INT8 GEMM kernel of Flash is picked in src (`transformer_h3_flash.int8_gemm_choice`,
reported by inference/hardware.py). The demo's Audio switch (Settings, inference/demo/ui.py) is shown for Flash and for
`--sampler base`, the samplers that step the soundtrack (`samplers.audio_schedule`); it is off by default, as the
CodecOPSD LoRA was trained with silent audio rows.
"""

from codecopsd.download import model_dir

from .hardware import announce, available_ram_gb, cap_gpu_memory, detect, vlm_files_gb

# GPU memory plans of HyperFlow H3 Flash, by the GPU's memory in GB: the first plan whose floor the GPU reaches. Every
# plan gives the same output; they differ only in what waits in CPU memory. Measured at 480p (README).
GPU_PLANS = (
    (64, dict(name="64 GB+", offload="none", stream_blocks=0,
              text="every model resident")),
    (30, dict(name="32 GB", offload="pinned", stream_blocks=0,
              text="one model on the GPU at a time, the text encoder streamed layer by layer")),
    (22, dict(name="24 GB", offload="pinned", stream_blocks=16,
              text="one model at a time, the text encoder and 16 of the transformer's 49 blocks streamed")),
    (14, dict(name="16 GB", offload="pinned", stream_blocks="all",
              text="one model at a time, the text encoder and every transformer block streamed")),
    (0, dict(name="12 GB", offload="pinned", stream_blocks="all",  # the 16 GB placement: 11.6 GB at 480p
             text="one model at a time, the text encoder and every transformer block streamed")),
)
# CPU memory a pinned plan (12 to 32 GB) keeps while it generates, in GB: the page-locked models, 47.3 GB (FP8 VLM
# 25.7, Flash 16.4, VAE 5.2), plus Python, PyTorch and CUDA. The same for every pinned plan (they change only what is
# on the GPU). Measured: 54.5 GB on the 16 GB plan (docs/consumer_gpus.md: capped at 56 GB a run gives the same video,
# at 48 GB it stops while loading); 53.8 GB resident after a 32 GB plan request (RTX PRO 6000, 480p, 56 frames).
PINNED_RAM_GB = 54.5
VLM_FP8_GB = 25.7  # of which the FP8 VLM; left in the page cache (`--pin-vlm no`) it is read again at each request
# `--pin-vlm auto` page-locks an FP8 VLM folder (read as file views, codecopsd.models.fp8_linear) only with this much
# CPU memory available (the plan plus room for a request's frames): page-locked memory cannot be reclaimed, so locking
# it on a machine short of memory stalls it. Below that the VLM stays in the page cache, read again at each request.
PIN_VLM_RAM_GB = 64


def gpu_plan(gpu_gb):
    """The placement plan (`GPU_PLANS`) for a GPU of `gpu_gb` GB: the first whose floor it reaches."""
    return next(plan for floor, plan in GPU_PLANS if gpu_gb >= floor)


def memory_plan(total, gpu_memory=None):
    """(budget, plan): the GPU memory to plan for in whole GB (`gpu_memory`, at most the GPU's `total`) and the
    HyperFlow H3 Flash plan of that budget: dict(name, offload, stream_blocks, text)."""
    budget = round(min(total, gpu_memory or total))
    return budget, gpu_plan(budget)


def flash_weights():
    """The default HyperFlow H3 Flash folder: ref2va_int8_convrot/ of `VideoRebirthAI/hyperflow-h3-flash`, next to
    the MiniMax-H3 folder in use (`codecopsd.download.model_dir`)."""
    return model_dir("hyperflow-h3-flash") / "ref2va_int8_convrot"


def setup(model, gpu_memory=None, offload="auto", prefix="", vlm=None, pin_vlm="auto"):
    """The start of a HyperFlow H3 Flash run, the demo page and the command line alike: the process capped to
    `gpu_memory` GB when that is below the first GPU's own (`hardware.cap_gpu_memory`, CUDA context included), the
    plan of that budget, and with `offload="auto"` its placement, printed as "<prefix>GPU memory plan ...". Then
    whether a pinned plan page-locks the VLM (`vlm`: `load.vlm_source`; an FP8 VLM folder is page-locked with
    `pin_vlm` "yes", left in the page cache with "no", and with "auto" page-locked only with `PIN_VLM_RAM_GB` of CPU
    memory available; a VLM converted at load is always page-locked), printed as one line (`load.vlm_notice`), and
    the hardware report with its warnings (`hardware.announce`): the CPU memory of a pinned plan, the commit headroom
    for the bf16 VLM files of `model` (the MiniMax-H3 folder) when they are converted. Returns dict(total (the GPU's
    GB), budget (whole GB), plan (None unless "auto"), offload, stream_blocks, pin_vlm (bool))."""
    from .load import vlm_notice, vlm_source

    total = cap_gpu_memory(gpu_memory)  # PyTorch's allocator gets the budget minus the context
    budget, plan = memory_plan(total, gpu_memory)
    stream_blocks, chosen = 0, None
    if offload == "auto":
        chosen, offload, stream_blocks = plan, plan["offload"], plan["stream_blocks"]
        print(f"{prefix}GPU memory plan {plan['name']}: {plan['text']}", flush=True)
    vlm = vlm or vlm_source()
    mapped = vlm["kind"] in ("downloaded", "saved")  # file views: page-locked in place, or left in the page cache
    info = detect()
    available = available_ram_gb(info)
    pin = not mapped or pin_vlm == "yes" or (pin_vlm == "auto" and (available is None or available >= PIN_VLM_RAM_GB))
    rule = "--pin-vlm no" if pin_vlm == "no" else f"less than {PIN_VLM_RAM_GB} GB of CPU memory available"
    vlm_notice(vlm, offload, pin, prefix, rule)
    ram = PINNED_RAM_GB - (0 if pin else VLM_FP8_GB)
    announce(ram_gb=ram if offload == "pinned" else None,
             commit_gb=vlm_files_gb(model) if vlm["kind"] in ("converted", "bf16") else None, info=info)
    return dict(total=total, budget=budget, plan=chosen, offload=offload, stream_blocks=stream_blocks, pin_vlm=pin)


# The extreme path encodes the references with TAEH3 (madebyollin/taehv `taeh3`, the encoder half: 1.5 M parameters,
# fp16) instead of the released VAE encoder (fp32, 256 px tiles), which took 10 to 34 s of a request. Measured on an RTX PRO
# 6000 (32 GB plan, 480p, references as released: pictures at a 2048 short edge, videos on the 768 canvas): the VAE
# encode stage 10.2 to 24.7 s -> 0.2 to 0.4 s; outputs follow the references like the released encoder's on the checked
# examples (identity, objects, motion, camera; no broken faces or frames), though not frame for frame. The codec
# gazer reads the reference pixels, so the kept tokens do not change. The decode keeps the released decoder.
TAEH3 = dict(url="https://raw.githubusercontent.com/madebyollin/taehv/9055eee0391548d1787938d040a161cbe396bf98/"
                 "safetensors/taeh3.safetensors",
             sha256="4fd022bfcab08772fe0536b17ea1a3bbb5625be11e397868d1c5d891863d4c13")


def _fetch(url, path):
    """`url` into `path`, checking HTTPS with certifi's certificates when present (huggingface_hub installs it; some
    Python builds find no system certificates)."""
    import shutil
    import ssl
    import urllib.request

    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    with urllib.request.urlopen(url, context=context, timeout=60) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def taeh3_weights(download=True):
    """models/taehv/taeh3.safetensors (23 MB), downloaded from the pinned taehv commit on first use."""
    import hashlib

    path = model_dir("taehv") / "taeh3.safetensors"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"no TAEH3 weights at {path}: run without --no-download, or with --vae-encoder released")
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        _fetch(TAEH3["url"], part)
        if hashlib.sha256(part.read_bytes()).hexdigest() != TAEH3["sha256"]:
            part.unlink()
            raise RuntimeError(f"TAEH3 weights from {TAEH3['url']} do not match their sha256")
        part.replace(path)
    return path


def use_taeh3_encoder(pipe, weights):
    """Encode the references of `pipe` (MiniMaxH3CodecOPSDPipeline, before its offload is set up) with TAEH3: it takes
    the place of `pipe.vae.encoder` (so the pinned offload moves it in the encoder's part) and `pipe.vae.encode` returns
    its latent as the posterior mean (log variance -30: a picture's sampled latent is that mean). Only the encode changes;
    the released decoder decodes."""
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    from .taeh3 import TAEH3Encoder

    vae = pipe.vae
    pixel_mean, pixel_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # the VAE's pixel normalization (ImageNet)

    class Encoder(torch.nn.Module):
        """ImageNet-normalized pixels [B, 3, T, H, W] -> posterior parameters [B, 48, T', H / 16, W / 16] of the raw
        H3 latent (TAEH3 writes the normalized one)."""

        def __init__(self):
            super().__init__()
            self.tae = TAEH3Encoder(weights).to(torch.float16).eval()
            self.register_buffer("mean", torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1), persistent=False)
            self.register_buffer("std", torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1), persistent=False)

        def forward(self, x):
            m = torch.tensor(pixel_mean, device=x.device).view(1, -1, 1, 1, 1)
            s = torch.tensor(pixel_std, device=x.device).view(1, -1, 1, 1, 1)
            rgb = (x.float() * s + m).clamp(0, 1).transpose(1, 2).to(torch.float16)  # [B, T, 3, H, W] in [0, 1]
            z = self.tae.encode_video(rgb)
            z = z[:, :1] if x.shape[2] == 1 else z  # a picture: the latent of its first chunk
            z = z.transpose(1, 2).float() * self.std.to(x.device) + self.mean.to(x.device)
            return torch.cat([z, torch.full_like(z, -30.0)], 1)

    vae.encoder = Encoder()

    def encode(x, return_dict=True):
        posterior = DiagonalGaussianDistribution(vae.encoder(x))  # a module call: the offload hooks move the part in
        return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)

    vae.encode = encode
    return pipe
