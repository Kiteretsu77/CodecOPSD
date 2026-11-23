"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Few-step samplers for MiniMax-H3 (reference to video). The released base runs 49 transformer calls; two published
step-distilled versions of it run 6 or 8, and the CodecOPSD student LoRA runs on both unchanged (it was trained on the base):

- "pika": Pika H3 Flex (`Pika-Labs/Pika-H3-Flex`, file `ref2va/adapter.safetensors`): a rank-128 LoRA on every row of the
  released Ref2VA transformer. 6 calls on the knots 1000 / 750 / 500 / 250 / 100 / 40 / 0 (sigma x 1000 before the
  released shift); after each call the clean prediction is re-noised to the next sigma with fresh noise (diffusers'
  `MiniMaxH3Scheduler(stochastic_sampling=True)`, as diffusers' `PikaH3FlexModularPipeline` samples it).
- "fasth3": FastH3 Omni 8-step (`FastVideo/FastVideo-FastH3-Omni-8-Step-V1`, folder `transformer_ref/`): a full fine-tune
  of the Ref2VA transformer whose output projection has one head per interval of a 33-point grid (parallel decoding
  distillation). 8 Euler calls from noise x 0.999, each over 4 intervals with their heads averaged by interval length (as
  FastVideo samples it, `fastvideo/layers/pdd.py`). Attention is dense (its sparse VSA gate is not used).
- "flash": HyperFlow H3 Flash (`VideoRebirthAI/hyperflow-h3-flash`, folder `ref2va_int8_convrot/`): a compressed
  Ref2VA transformer of its own (block 9 removed, low-rank and W8A8 INT8 linears, 13.9B parameters, 16.3 GB) with a
  two-time (t, r) 6-step distillation. 6 Euler calls on its raw sigmas shifted by 12; every row also receives the
  endpoint r of its step (generated rows r = 1 - next sigma, as Flash's `build_row_time_pairs`; references and the
  clean audio rows r = t). It replaces the transformer, not a LoRA; the CodecOPSD LoRA runs on the 49 blocks it keeps.

Both keep the CodecOPSD layout and conditioning (silent clean audio rows, references at t = 0.999); an outside LoRA acts on
every row, the CodecOPSD LoRA on the reference rows only.
"""

import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import MiniMaxH3Scheduler
from diffusers.utils import logging
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from ...configs import load

SAMPLERS = ("pika", "fasth3", "flash", "base")
_FLASH = load("minimax_h3_ref2va")["samplers"]["flash"]
FLASH_GRIDS = {6: _FLASH["sigmas"], **{int(k): v for k, v in _FLASH["other_steps"].items()}}  # raw grids by step count
PIKA_KNOTS = load("minimax_h3_ref2va")["samplers"]["pika"]["knots"]


class AllRowsLoRALinear(nn.Module):
    """A frozen linear layer plus an outside LoRA on every row, unmerged as PEFT computes it: `base(x) + B(A(x)) * scale`."""

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base, self.scale = base, scale
        self.down = nn.Parameter(down, requires_grad=False)
        self.up = nn.Parameter(up, requires_grad=False)

    @property
    def weight(self):  # the base layer's, for code that reads the dtype or device of a projection
        return self.base.weight

    def forward(self, x):
        y = self.base(x)
        lora = F.linear(F.linear(x, self.down.to(x.dtype)), self.up.to(x.dtype))
        if torch.is_grad_enabled():
            return y + lora * self.scale
        return y.add_(lora.mul_(self.scale))  # in place, same values: two output-sized buffers less (80 GB GPUs)


def load_pika_adapter(transformer, path):
    """Pika H3 Flex's Ref2VA adapter (PEFT names `{module}.lora_{A,B}.default.weight`, rank = alpha = 128) on every row
    of `transformer`, in its dtype. Load it before the CodecOPSD LoRA, which then wraps these layers."""
    state = load_file(path)
    names = sorted({key.split(".lora_")[0] for key in state})
    dtype = transformer.transformer_blocks[0].attn.to_k.weight.dtype
    for name in names:
        down, up = state[f"{name}.lora_A.default.weight"], state[f"{name}.lora_B.default.weight"]
        parent, child = name.rsplit(".", 1)
        module = transformer.get_submodule(parent)
        base = module[int(child)] if child.isdigit() else getattr(module, child)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"{name}: load the Pika adapter before any other LoRA, found {type(base).__name__}")
        layer = AllRowsLoRALinear(base, down.to(dtype), up.to(dtype), 1.0)  # alpha / rank = 128 / 128
        if child.isdigit():
            module[int(child)] = layer
        else:
            setattr(module, child, layer)
    return len(names)


class PDDLinear(nn.Module):
    """FastH3's widened output projection, one head per fine-grid interval (`grid` x the released width). The forward
    returns the heads of `block` (start, end, weights normalized to sum 1) averaged in float32, as FastVideo fuses them."""

    def __init__(self, weight, bias, grid):
        super().__init__()
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.bias = nn.Parameter(bias.float(), requires_grad=False)
        self.grid, self.block = grid, None

    def forward(self, x):
        start, end, weights = self.block
        fuse = lambda p: torch.einsum("n,n...->...", weights.to(p.device), p.unflatten(0, (self.grid, -1))[start:end])
        return F.linear(x, fuse(self.weight), fuse(self.bias))


def load_fasth3_transformer(cls, folder, torch_dtype):
    """The FastH3 Omni 8-step transformer from its downloaded repo `folder` as `cls`, with its 32-head output projections
    (`PDDLinear`) and its sampling contract (`fastvideo_inference.json`) as `transformer.pdd_contract`."""
    sub = os.path.join(folder, "transformer_ref")
    level = logging.get_verbosity()
    logging.set_verbosity_error()  # the widened heads do not fit the released shapes and are read below; the VSA gate is unused
    try:
        transformer = cls.from_pretrained(folder, subfolder="transformer_ref", torch_dtype=torch_dtype,
                                          ignore_mismatched_sizes=True)
    finally:
        logging.set_verbosity(level)
    grid = json.load(open(os.path.join(sub, "config.json")))["pdd_steps"]
    index = json.load(open(os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    for name in ("proj_out", "audio_proj_out"):
        tensors = []
        for key in (f"{name}.weight", f"{name}.bias"):
            with safe_open(os.path.join(sub, index[key]), "pt") as f:
                tensors.append(f.get_tensor(key))
        setattr(transformer, name, PDDLinear(*tensors, grid))
    transformer.pdd_contract = json.load(open(os.path.join(folder, "fastvideo_inference.json")))
    return transformer


def pdd_plan(contract):
    """Node sigmas and per-interval weights of FastH3's sampler (FastVideo `build_pdd_sampling_plan`, float64): the fine
    grid u = linspace(M, 0, 33) (M = 0.999), shifted per modality with M as a fixed point, f(u) = s u M / ((s - 1) u + M);
    head j moves the state by f(u_{j+1}) - f(u_j). Returns the video node sigmas and, per call, the head block of each
    output projection."""
    m, nodes = contract["grid_max_t"], contract["pdd_step_indices"]
    grid = torch.linspace(m, 0.0, nodes[-1] + 1, dtype=torch.float64).clamp(max=m)
    sigma = lambda u, s: s * u * m / (u * (s - 1.0) + m)
    delta = lambda a, b, s: s * m**2 * (b - a) / ((m + (s - 1.0) * a) * (m + (s - 1.0) * b))
    weights = {name: delta(grid[:-1], grid[1:], contract[key])
               for name, key in (("proj_out", "video_scheduler_shift"), ("audio_proj_out", "audio_scheduler_shift"))}
    blocks = []
    for start, end in zip(nodes[:-1], nodes[1:]):
        blocks.append({name: (start, end, (w[start:end] / w[start:end].sum()).float()) for name, w in weights.items()})
    return sigma(grid[torch.tensor(nodes)], contract["video_scheduler_shift"]), blocks


def flash_sigmas(shift, steps=None):
    """HyperFlow H3 Flash's grid of `steps` calls (4, 6 or 8; default 6) shifted as its `shift_sigmas` does it (float32):
    shift s / (1 + (shift - 1) s)."""
    steps = 6 if steps is None else int(steps)
    if steps not in FLASH_GRIDS:
        raise ValueError(f"HyperFlow H3 Flash samples {sorted(FLASH_GRIDS)} steps, got {steps}")
    grid = torch.tensor(FLASH_GRIDS[steps], dtype=torch.float32)
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def audio_schedule(sampler, scheduler_config, device, num_inference_steps=None):
    """The schedule of a generated soundtrack's rows (`audio=True`), next to the video's in the same calls: the audio
    scheduler (`scheduler_config`, MiniMax-H3's audio_scheduler/, shift 3) on the sampler's grid, as the released model
    and HyperFlow H3 Flash sample audio. "flash": its raw sigmas shifted by 3, Euler, every call with the endpoint
    r = 1 - next sigma (Flash's `build_row_time_pairs`); "base": the released grid of `num_inference_steps` points.
    Returns the scheduler and the endpoints (None for "base")."""
    if sampler == "flash":
        sigmas = flash_sigmas(float(scheduler_config["shift"]), num_inference_steps)
        scheduler = MiniMaxH3Scheduler.from_config(scheduler_config, shift_provided_sigmas=False)
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        return SimpleNamespace(scheduler=scheduler, endpoints=(1.0 - sigmas[1:]).tolist())
    if sampler == "base":
        scheduler = MiniMaxH3Scheduler.from_config(scheduler_config)
        scheduler.set_timesteps(num_inference_steps, device=device)
        return SimpleNamespace(scheduler=scheduler, endpoints=None)
    raise NotImplementedError(f"audio=True runs with the samplers 'flash' and 'base', not {sampler!r}")


def few_step(sampler, scheduler_config, transformer, device, steps=None):
    """The scheduler, the scale of the starting noise, the per-call setup (or None) and, for Flash, the endpoint r of
    every call (else None) of a few-step sampler. `steps`: Flash's 4, 6 (default) or 8 calls."""
    if sampler == "pika":
        scheduler = MiniMaxH3Scheduler.from_config(scheduler_config, stochastic_sampling=True, shift_provided_sigmas=True)
        scheduler.set_timesteps(sigmas=[k / 1000 for k in PIKA_KNOTS], device=device)
        return SimpleNamespace(scheduler=scheduler, noise_scale=1.0, setup=None, endpoints=None)
    if sampler == "flash":  # the shifted grid taken verbatim, as Flash's FlashSetTimestepsStep passes it
        sigmas = flash_sigmas(float(scheduler_config["shift"]), steps)
        scheduler = MiniMaxH3Scheduler.from_config(scheduler_config, shift_provided_sigmas=False)
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        return SimpleNamespace(scheduler=scheduler, noise_scale=1.0, setup=None, endpoints=(1.0 - sigmas[1:]).tolist())
    sigmas, blocks = pdd_plan(transformer.pdd_contract)
    scheduler = MiniMaxH3Scheduler.from_config(scheduler_config, shift_provided_sigmas=False)  # node sigmas as they are
    scheduler.set_timesteps(sigmas=sigmas.float(), device=device)

    def setup(i):
        for name, block in blocks[i].items():
            getattr(transformer, name).block = block

    return SimpleNamespace(scheduler=scheduler, noise_scale=float(sigmas[0]), setup=setup, endpoints=None)
