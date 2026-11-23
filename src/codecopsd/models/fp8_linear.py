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

Converting at load needs the 66 GB bf16 release in CPU memory first (read, cropped, then converted), more than many
machines with a consumer GPU have (and, under strict overcommit, more commit headroom). So the entry points read an
FP8 VLM that is already converted instead (`load_fp8_text_encoder`, `codecopsd.download.vlm_fp8_source`):

- the third-party ModelOpt copy `feizhai123/MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8` (folder Ref2VA/text_encoder/,
  29.6 GB), downloaded into models/. Checked tensor by tensor against our conversion: its 316 FP8 linears (F8_E4M3
  weight [out, in], F32 `weight_scale` [out]) are byte-identical to `Fp8Linear`'s `weight_fp8` and `scale`; the 34
  linears it keeps in bf16 are byte-identical to the release, and are converted here at load (`Fp8Linear`, as
  `text_encoder_to_fp8` does); every other tensor is identical. It holds decoder layers 0-49 only (MiniMax-H3 reads
  the input of layer 50, the pipelines' `vlm_prompt_embeds`) and an unused language-model head, skipped;
- a folder saved once by `save_fp8_text_encoder` (`python -m inference.vlm_fp8`, models/MiniMax-H3-VLM-FP8);
- else the bf16 release, converted at load as above (`text_encoder_to_fp8`).

Both FP8 folders are read as views of read-only file mappings (`codecopsd.offload.map_safetensors`): no bf16 read, no
copy in process memory, and the pinned offload can page-lock them in place or leave them in the page cache.
"""

# TODO (RTX PRO 6000 / RTX 5090): FastVideo runs MiniMax-H3 on these GPUs with NVFP4 weights for the DiT, the text
# encoder, and the VAE decoder by default. Measure speed, GPU memory, and quality against bf16 on an RTX PRO 6000 before
# adopting any of them (on a B200 the NVFP4 DiT was 1.14-1.3x faster per step but changed the content).

import json
from pathlib import Path

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


def fp8_tensor_cores(capability):
    """Whether a GPU of compute capability `capability` ((major, minor)) has FP8 tensor cores: 8.9 and up (Ada, Hopper,
    Blackwell). Without them `matmul="fp8"` computes in bf16 from the FP8 weights."""
    return tuple(capability) >= (8, 9)

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

    @classmethod
    def empty(cls, linear: nn.Linear, matmul: str = "fp8"):
        """An `Fp8Linear` of the shape of `linear` with no weights yet (meta tensors), filled by
        `load_fp8_text_encoder`."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.register_buffer("weight_fp8", torch.empty(linear.out_features, linear.in_features, dtype=FP8,
                                                       device="meta"))
        self.register_buffer("scale", torch.empty(1, linear.out_features, dtype=torch.float32, device="meta"))
        self.bias = linear.bias
        self.matmul = matmul
        return self

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
        if self.matmul != "fp8" or not x.is_cuda or not fp8_tensor_cores(torch.cuda.get_device_capability(x.device)):
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


FP8_FORMAT = "codecopsd_fp8"  # config key of a saved FP8 text encoder: the decoder layers kept, the scale layout


def save_fp8_text_encoder(text_encoder, folder, shard_gb=4.0):
    """Write a text encoder converted by `text_encoder_to_fp8` (and cropped to the layers MiniMax-H3 reads, as the
    pipelines do) to `folder`: its config and every tensor exactly as the converted model holds it, in .safetensors
    shards of at most `shard_gb` GB with an index (`load_fp8_text_encoder` reads it back)."""
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    shards, size = [{}], 0
    for name, tensor in text_encoder.state_dict().items():
        nbytes = tensor.numel() * tensor.element_size()
        if shards[-1] and size + nbytes > shard_gb * 2**30:
            shards.append({})
            size = 0
        shards[-1][name] = tensor.detach().to("cpu").contiguous()
        size += nbytes
    weight_map = {}
    for i, shard in enumerate(shards):
        file = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(folder / file), metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(shard, file))
    total = sum(t.numel() * t.element_size() for shard in shards for t in shard.values())
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    text_encoder.config.save_pretrained(folder)
    config = json.loads((folder / "config.json").read_text())
    config[FP8_FORMAT] = dict(layers=len(text_encoder.model.language_model.layers),
                              weights="float8_e4m3fn [out, in], float32 scale [1, out] per output row")
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    return folder


def fp8_folder_format(folder):
    """"modelopt" (the downloaded ModelOpt copy: FP8 per output channel, dynamic per-token activations), "saved"
    (`save_fp8_text_encoder`) or None (no FP8 text encoder in `folder`), from its config.json."""
    try:
        config = json.loads((Path(folder) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    if FP8_FORMAT in config:
        return "saved"
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") == "modelopt" and quant.get("quant_algo") == "FP8_PER_CHANNEL_PER_TOKEN":
        return "modelopt"
    return None


def load_fp8_text_encoder(folder, matmul: str = "fp8", dtype=torch.bfloat16):
    """The FP8 text encoder in `folder`, in CPU memory, the same tensors as `text_encoder_to_fp8` makes from the bf16
    release (`matmul` as there): a folder of `save_fp8_text_encoder` (51 decoder layers) or the downloaded ModelOpt
    copy (50 decoder layers; see the module docstring). Every weight read from the files is a view of a read-only
    shared mapping of them (`codecopsd.offload.map_safetensors`): nothing is converted or copied, except the linears
    the ModelOpt copy keeps in bf16, converted to `Fp8Linear` here exactly as `text_encoder_to_fp8` converts them."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForImageTextToText

    from ..offload import map_safetensors

    if matmul not in ("fp8", "bf16"):
        raise ValueError(f"matmul must be 'fp8' or 'bf16', got {matmul!r}")
    folder = Path(folder)
    kind = fp8_folder_format(folder)
    if kind is None:
        raise ValueError(f"{folder} holds no FP8 text encoder (neither a ModelOpt FP8 copy nor a saved one)")
    config = AutoConfig.from_pretrained(folder)
    if kind == "modelopt":
        # The copy's ModelOpt quantization_config is not applied: transformers would build its own FP8 modules. The
        # model is built as the bf16 release and its FP8 tensors become our `Fp8Linear` buffers below.
        del config.quantization_config
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    with init_empty_weights(include_buffers=False):  # weights on the meta device, buffers (rotary tables) computed
        text_encoder = AutoModelForImageTextToText.from_config(config, dtype=dtype)
    text_encoder.lm_head = nn.Identity()  # never read by MiniMax-H3 (the copy's lm_head.weight is skipped)
    keep_bf16 = []  # (parent, name) of the linears the ModelOpt copy keeps in bf16: converted after loading
    for i, layer in enumerate(text_encoder.model.language_model.layers):
        for path, parent in list(layer.named_modules()):
            for name, child in list(parent.named_children()):
                if not isinstance(child, nn.Linear):
                    continue
                key = f"model.language_model.layers.{i}.{path + '.' if path else ''}{name}"
                if kind == "modelopt" and f"{key}.weight_scale" not in index:
                    keep_bf16.append((parent, name))
                else:
                    setattr(parent, name, Fp8Linear.empty(child, matmul))
    expected = text_encoder.state_dict(keep_vars=True)  # every tensor the files must give (not the rotary tables)
    wanted = {}  # file key -> (model key, layout)
    for key in index:
        if key.startswith("lm_head."):
            continue
        base, _, leaf = key.rpartition(".")
        if kind == "modelopt" and leaf == "weight" and f"{base}.weight_scale" in index:
            wanted[key] = (f"{base}.weight_fp8", "fp8")
        elif kind == "modelopt" and leaf == "weight_scale":
            wanted[key] = (f"{base}.scale", "scale")
        else:
            wanted[key] = (key, "same")
    missing = sorted(set(expected) - {target for target, _ in wanted.values()})
    unexpected = sorted(target for target, _ in wanted.values() if target not in expected)
    if missing or unexpected:
        raise ValueError(f"{folder} does not match its config: missing {missing[:3]}, unexpected {unexpected[:3]}")
    for file in sorted(set(index.values())):
        for key, tensor in map_safetensors(folder / file).items():
            if key not in wanted:
                continue
            target, layout = wanted[key]
            if layout == "scale":  # ModelOpt: float32 [out]; Fp8Linear: float32 [1, out] (the same bytes)
                tensor = tensor.view(1, -1)
            meta = expected[target]
            if tensor.dtype != meta.dtype or tuple(tensor.shape) != tuple(meta.shape):
                raise ValueError(f"{folder}: {key} is {tensor.dtype} {tuple(tensor.shape)}, expected {meta.dtype} "
                                 f"{tuple(meta.shape)}")
            path, _, leaf = target.rpartition(".")
            module = text_encoder.get_submodule(path)
            if leaf in module._parameters:
                module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
            else:
                module._buffers[leaf] = tensor
    left = [n for n, t in [*text_encoder.named_parameters(), *text_encoder.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{folder}: no weights for {left[:3]}")
    for parent, name in keep_bf16:  # as text_encoder_to_fp8: on the GPU when there is one, the result in CPU memory
        setattr(parent, name, Fp8Linear(getattr(parent, name), matmul))
    torch.cuda.empty_cache()
    return text_encoder.eval()
