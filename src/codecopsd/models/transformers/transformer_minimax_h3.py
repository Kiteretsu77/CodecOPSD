"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

MiniMax-H3 and MiniMax-H3-Image transformers with a condition LoRA that acts only on the reference rows of the packed
sequence."""

import torch
import torch.nn.functional as F
from diffusers import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Transformer3DModel
from torch import nn

# Projections of every block that carry the condition LoRA; in the last block the reference rows only feed the keys and
# values of the generated rows, so only those two carry it.
LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
LAST_BLOCK_TARGETS = ("attn.to_k", "attn.to_v")


class ReferenceRowLoRALinear(nn.Module):
    """A linear layer plus a LoRA applied only to the rows `rows.start : rows.end` (the reference rows; end None = to the
    last row)."""

    def __init__(self, base, down, up, scale, rows):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(down, requires_grad=down.requires_grad)
        self.up = nn.Parameter(up, requires_grad=up.requires_grad)
        self.scale = scale
        self.rows = rows  # shared by every layer: rows.start / rows.end are set per forward

    def forward(self, x):
        y = self.base(x)
        start, end = self.rows.start, self.rows.end
        if start is None or start >= x.shape[1]:
            return y
        lora = F.linear(F.linear(x[:, start:end], self.down.to(x.dtype)), self.up.to(x.dtype))
        if torch.is_grad_enabled():
            lora = lora * self.scale
            return torch.cat([y[:, :start], y[:, start:end] + lora, y[:, start + lora.shape[1] :]], dim=1)
        y[:, start:end] += lora.mul_(self.scale)  # in place: no second reference-rows-sized buffer
        return y


class StreamedAdaLNModulation(nn.Module):
    """A block's AdaLN modulation projection (`MiniMaxH3AdaLayerNormModulation`, 520 MB; 26 GB over the 50 blocks of
    MiniMax-H3, applied to one row per timestep) kept in pinned CPU memory and copied to the GPU at every call, the next
    block's copy on a side stream while this block runs. Same product as the released module (exact); with model
    offload the transformer then needs 26 GB less GPU memory."""

    def __init__(self, module):
        super().__init__()
        self.hidden_size = module.hidden_size
        # plain tensors, not parameters or buffers: `.to(device)` (model offload) leaves them in CPU memory; an already
        # streamed module hands over its pinned copy
        self.host = module.host if isinstance(module, StreamedAdaLNModulation) else [
            t.detach().cpu().pin_memory() for t in (module.linear.weight, module.linear.bias)]
        self.next = None  # the following block's module, prefetched while this one runs
        self.ready = None  # [weight, bias] on the GPU and the event of their copy

    def fetch(self, device):
        stream = _copy_stream(device)
        with torch.cuda.stream(stream):
            tensors = [t.to(device, non_blocking=True) for t in self.host]
            event = torch.cuda.Event()
            event.record(stream)
        self.ready = (tensors, event)

    def forward(self, temb):
        if self.ready is None:
            self.fetch(temb.device)
        (weight, bias), event = self.ready
        self.ready = None
        compute = torch.cuda.current_stream(temb.device)
        compute.wait_event(event)
        for t in (weight, bias):
            t.record_stream(compute)
        if self.next is not None:
            self.next.fetch(temb.device)
        temb = F.linear(F.silu(temb).to(weight.dtype), weight, bias)
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class TabledAdaLNModulation(StreamedAdaLNModulation):
    """A block's AdaLN modulation projection as a table: its output (the six modulation tensors) is computed once for
    each `timestep` input of the transformer (a sampler has a few) and reused; the 520 MB weight stays in pinned CPU
    memory and is streamed in only to fill a new entry (`StreamedAdaLNModulation`). Each entry is the released module's
    product on the same input, so the output is unchanged; the transformer needs 26 GB less GPU memory and, once the
    sampler's timesteps are in the table, copies nothing per call. The blocks only read the six tensors."""

    def __init__(self, module, key):
        super().__init__(module)
        self.key, self.table = key, {}  # key[0]: the transformer's current `timestep` (set by `table_adaln`)

    def forward(self, temb):
        entry = self.table.get(self.key[0])
        if entry is None:
            entry = super().forward(temb)
            if self.key[0] is not None:
                if len(self.table) >= 256:  # about 0.6 MB per entry and block; a sampler needs a few entries
                    self.table.clear()
                self.table[self.key[0]] = entry
        return entry


def table_adaln(transformer):
    """Replace every block's AdaLN projection of a MiniMax-H3 / MiniMax-H3-Image transformer by a
    `TabledAdaLNModulation` (exact: its input `temb` depends on the `timestep` argument only, and no LoRA touches it).
    Call it before moving the transformer to the GPU."""
    if getattr(transformer, "_adaln_key", None) is not None:
        return
    key = transformer._adaln_key = [None]

    def remember(module, args, kwargs):
        t = kwargs.get("timestep")
        key[0] = None if t is None else (str(t.dtype), tuple(t.shape), t.detach().cpu().numpy().tobytes())

    transformer.register_forward_pre_hook(remember, with_kwargs=True)
    tabled = [TabledAdaLNModulation(block.adaln_proj, key) for block in transformer.transformer_blocks]  # streamed ones too
    for block, module in zip(transformer.transformer_blocks, tabled, strict=True):
        block.adaln_proj = module
    for module, following in zip(tabled, tabled[1:]):
        module.next = following


_COPY_STREAMS = {}


def _copy_stream(device):
    device = torch.device(device)
    if device not in _COPY_STREAMS:
        _COPY_STREAMS[device] = torch.cuda.Stream(device)
    return _COPY_STREAMS[device]


class _Rows:
    start = None
    end = None


def _has_module(model, name):
    try:
        model.get_submodule(name)
        return True
    except AttributeError:
        return False


class ConditionLoRAMixin:
    """The CodecOPSD condition LoRA of a MiniMax-H3 transformer: `forward(..., reference_start=i, reference_end=j)` applies
    it to rows `i:j` (the reference rows; j None = to the last row); without `reference_start` the transformer is the
    released one."""

    def _wrap(self, name, down, up, alpha):
        parent, child = name.rsplit(".", 1)
        module = self.get_submodule(parent)
        base = module[int(child)] if child.isdigit() else getattr(module, child)
        base = base.base if isinstance(base, ReferenceRowLoRALinear) else base
        layer = ReferenceRowLoRALinear(base, down, up, alpha / down.shape[0], self._lora_rows)
        if child.isdigit():
            module[int(child)] = layer
        else:
            setattr(module, child, layer)

    def lora_dtype(self):
        """The dtype the condition LoRA runs in: that of the projections it wraps."""
        return self.transformer_blocks[0].attn.to_k.weight.dtype

    def load_condition_lora(self, state_dict, alpha):
        """Wrap every projection named in `state_dict` (`transformer_blocks.{i}.{module}.lora_{A,B}.weight`), frozen,
        in `lora_dtype()`. Projections this transformer does not have (a block a compressed transformer removed) are
        skipped; their number is left in `self.skipped_lora`. Returns the number of wrapped projections."""
        self._lora_rows = getattr(self, "_lora_rows", _Rows())
        names = sorted({key.rsplit(".lora_", 1)[0] for key in state_dict})
        present = [name for name in names if _has_module(self, name)]
        dtype = self.lora_dtype() if present else None
        for name in present:
            down, up = state_dict[f"{name}.lora_A.weight"], state_dict[f"{name}.lora_B.weight"]
            self._wrap(name, down.to(dtype), up.to(dtype), alpha)
        self.skipped_lora = len(names) - len(present)
        return len(present)

    def init_condition_lora(self, rank, alpha, targets=LORA_TARGETS, last_block_targets=LAST_BLOCK_TARGETS):
        """A new trainable float32 LoRA that starts as the released model (down ~ N(0, 1 / rank), up = 0); returns its
        parameters."""
        self._lora_rows = getattr(self, "_lora_rows", _Rows())
        last = len(self.transformer_blocks) - 1
        for i, block in enumerate(self.transformer_blocks):
            for target in last_block_targets if i == last else targets:
                base = block.get_submodule(target)
                base = getattr(base, "base", base)
                device = base.weight.device
                down = torch.randn(rank, base.in_features, device=device) / rank
                up = torch.zeros(base.out_features, rank, device=device)
                self._wrap(f"transformer_blocks.{i}.{target}", down.requires_grad_(), up.requires_grad_(), alpha)
        return [p for n, p in self.named_parameters() if n.endswith((".down", ".up"))]

    def condition_lora_state_dict(self):
        """The LoRA in the file format of `load_condition_lora` (float32). Sharded (FSDP) factors are gathered: call it
        on every process."""
        out = {}
        for name, module in self.named_modules():
            if isinstance(module, ReferenceRowLoRALinear):
                for key, factor in (("lora_A", module.down), ("lora_B", module.up)):
                    factor = factor.detach()
                    factor = factor.full_tensor() if hasattr(factor, "full_tensor") else factor
                    out[f"{name}.{key}.weight"] = factor.float().cpu().contiguous()
        return out

    def forward(self, *args, reference_start=None, reference_end=None, **kwargs):
        if hasattr(self, "_lora_rows"):
            self._lora_rows.start, self._lora_rows.end = reference_start, reference_end
        return super().forward(*args, **kwargs)


class MiniMaxH3CodecOPSDTransformer3DModel(ConditionLoRAMixin, MiniMaxH3Transformer3DModel):
    r"""MiniMax-H3 (Ref2VA) transformer that runs a CodecOPSD condition LoRA on the reference rows.

    The packed sequence is laid out as `[text | target audio | target video | references]` (rotary positions carry the
    real layout, so the order changes nothing else); the LoRA acts on rows `reference_start:`.
    """

    def table_adaln(self):
        """Every block's AdaLN projection as a per-timestep table, its weight in pinned CPU memory (exact, 26 GB less
        GPU memory; `TabledAdaLNModulation`)."""
        table_adaln(self)

    def stream_adaln(self):
        """Keep every block's AdaLN projection in pinned CPU memory, streamed to the GPU block by block
        (`StreamedAdaLNModulation`, exact). `MiniMaxH3CodecOPSDPipeline.enable_model_cpu_offload` calls it."""
        blocks = [block.adaln_proj for block in self.transformer_blocks]
        streamed = [b if isinstance(b, StreamedAdaLNModulation) else StreamedAdaLNModulation(b) for b in blocks]
        for block, module in zip(self.transformer_blocks, streamed, strict=True):
            block.adaln_proj = module
        for module, following in zip(streamed, streamed[1:]):
            module.next = following


class MiniMaxH3ImageCodecOPSDTransformer2DModel(ConditionLoRAMixin, MiniMaxH3ImageTransformer2DModel):
    r"""MiniMax-H3-Image transformer that runs a CodecOPSD condition LoRA on the reference rows.

    The packed sequence keeps the released order `[text | references | generated image]`; the LoRA acts on rows
    `reference_start:reference_end`.
    """

    def table_adaln(self):
        """Every block's AdaLN projection as a per-timestep table, its weight in pinned CPU memory (exact, 26 GB less
        GPU memory; `TabledAdaLNModulation`)."""
        table_adaln(self)
