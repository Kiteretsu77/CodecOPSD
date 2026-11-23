"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Exact fast inference of Cosmos 3 video -> action (robot inverse dynamics, `id_droid`): the same kernels on the same
values as the released forward, so the actions are bit-identical.

A video -> action call is small (17 frames, 1,700 condition tokens, 16 action tokens) and runs about 7,700 short GPU
kernels per denoising step, so on fast GPUs the step time is the host's launch and bookkeeping time, not the GPU's. This
module removes that host time without changing any kernel or its inputs:

1. Once per call instead of at every step: the sparse preprocessing (the codec masks -> kept rows, renumbered indexes,
   LoRA rows) and its checks, and the action domain-id range check; the kept vision rows are then gathered by index (the
   same gather as the boolean mask, without a host sync).
2. A float32 condition LoRA casts the condition rows of an input once (q / k / v and gate / up read the same input);
   the default bfloat16 LoRA reads the rows as they are.
3. Text cache: the text (understanding) pathway never reads the generation tokens (causal attention over the text
   prefix), so its keys / values for the generation attention and its output are computed at the first step of a call
   and reused at the later steps.
4. CUDA graphs: the transformer forward is captured as one CUDA graph per mode (first step: fills the text cache; later
   steps: reads it) with static input buffers; inputs are copied in only when they changed. Graphs are kept for the next
   call with the same shapes (a new input video can change the number of kept tokens and then recaptures).
"""

import contextlib
import types

import torch
from diffusers import Cosmos3OmniTransformer
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_cosmos3 import _rotate_half

# The inputs the released denoiser rebuilds at every step; every other input is the same object for the whole call.
STEP_INPUTS = ("vision_tokens", "vision_timesteps", "sound_tokens", "sound_timesteps", "action_tokens",
               "action_timesteps")
KEEP_ROWS = "_keep_rows"  # the kept vision rows (indices), a graph input next to the forward's own inputs


def _snapshot(value):
    """What identifies an input within one call: tensors by object and version, lists element by element."""
    if isinstance(value, torch.Tensor):
        return ("tensor", value, value._version)
    if isinstance(value, (list, tuple)):
        return ("seq", type(value), [_snapshot(v) for v in value])
    return ("value", value)


def _unchanged(snapshot, value):
    kind = snapshot[0]
    if kind == "tensor":
        return value is snapshot[1] and value._version == snapshot[2]
    if kind == "seq":
        items = snapshot[2]
        return type(value) is snapshot[1] and len(value) == len(items) and all(map(_unchanged, items, value))
    return not isinstance(value, torch.Tensor) and type(value) is type(snapshot[1]) and value == snapshot[1]


def _flatten(inputs):
    """The tensors of `inputs` (top level or in lists of tensors) in a fixed order, and the spec that rebuilds it."""
    tensors, spec = [], []
    for name in sorted(inputs):
        value = inputs[name]
        if isinstance(value, torch.Tensor):
            spec.append((name, "tensor", len(tensors)))
            tensors.append(value)
        elif isinstance(value, (list, tuple)) and value and all(isinstance(v, torch.Tensor) for v in value):
            spec.append((name, type(value), tuple(range(len(tensors), len(tensors) + len(value)))))
            tensors.extend(value)
        else:
            spec.append((name, "value", value))
    return tensors, spec


def _unflatten(spec, tensors):
    out = {}
    for name, kind, ref in spec:
        out[name] = tensors[ref] if kind == "tensor" else ref if kind == "value" else kind(tensors[i] for i in ref)
    return out


def _domain_linear_unchecked(self, x, domain_id):
    """`DomainAwareLinear.forward` without its domain-id range check (a host sync), which the fast path runs once per
    call on the action domain ids (the per-token ids are copies of them)."""
    if domain_id.ndim == 0:
        domain_id = domain_id.unsqueeze(0)
    domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
    if x.shape[0] != domain_id.shape[0]:
        raise ValueError("Cosmos3 action domain_id batch size must match action tokens: "
                         f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}.")
    weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
    bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
    if x.ndim == 2:
        return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
    if x.ndim == 3:
        return torch.bmm(x, weight) + bias.unsqueeze(1)
    raise ValueError(f"Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")


def _attention(attn, und_seq, gen_seq, rotary_emb, kv):
    """The released `Cosmos3AttnProcessor` (same ops in the same form). With `kv` = (text keys for the generation
    attention, text values) of this layer, the text half is not computed: `und_seq` is not read and no text output is
    returned. Returns (text output, generation output, kv)."""
    processor = attn.processor
    heads, kv_heads, dim = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
    cos_und, sin_und, cos_gen, sin_gen = rotary_emb
    q_gen = attn.add_q_proj(gen_seq).view(-1, heads, dim)
    k_gen = attn.add_k_proj(gen_seq).view(-1, kv_heads, dim)
    v_gen = attn.add_v_proj(gen_seq).view(-1, kv_heads, dim)
    q_gen = attn.norm_added_q(q_gen)
    k_gen = attn.norm_added_k(k_gen)
    cos_gen = cos_gen.unsqueeze(1)
    sin_gen = sin_gen.unsqueeze(1)
    q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
    k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen
    und_out = None
    if kv is None:
        q_und = attn.to_q(und_seq).view(-1, heads, dim)
        k_und = attn.to_k(und_seq).view(-1, kv_heads, dim)
        v_und = attn.to_v(und_seq).view(-1, kv_heads, dim)
        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        k_und_for_gen = attn.k_norm_und_for_gen(k_und) if attn.k_norm_und_for_gen is not None else k_und
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
        causal_out = dispatch_attention_fn(q_und.unsqueeze(0), k_und.unsqueeze(0), v_und.unsqueeze(0), is_causal=True,
                                           enable_gqa=True, backend=processor._attention_backend,
                                           parallel_config=processor._parallel_config)
        und_out = attn.to_out(causal_out.squeeze(0).flatten(-2, -1))
        kv = (k_und_for_gen, v_und)
    all_k = torch.cat([kv[0], k_gen], dim=0)
    all_v = torch.cat([kv[1], v_gen], dim=0)
    full_out = dispatch_attention_fn(q_gen.unsqueeze(0), all_k.unsqueeze(0), all_v.unsqueeze(0), is_causal=False,
                                     enable_gqa=True, backend=processor._attention_backend,
                                     parallel_config=processor._parallel_config)
    gen_out = attn.to_add_out(full_out.squeeze(0).flatten(-2, -1))
    return und_out, gen_out, kv


def _decoder_layer(layer, und_seq, gen_seq, rotary_emb, kv):
    """The released `Cosmos3VLTextMoTDecoderLayer.forward`. With `kv` (this layer's text keys / values from the first
    step of the call) only the generation half runs and `und_seq` is passed on unchanged: no generation op reads it.
    Returns (text sequence, generation sequence, kv)."""
    gen_norm = layer.input_layernorm_moe_gen(gen_seq)
    if kv is None:
        und_norm = layer.input_layernorm(und_seq)
        und_attn_out, gen_attn_out, kv = _attention(layer.self_attn, und_norm, gen_norm, rotary_emb, None)
        residual_und = und_seq + und_attn_out
        und_seq = residual_und + layer.mlp(layer.post_attention_layernorm(residual_und))
    else:
        _, gen_attn_out, _ = _attention(layer.self_attn, None, gen_norm, rotary_emb, kv)
    residual_gen = gen_seq + gen_attn_out
    gen_seq = residual_gen + layer.mlp_moe_gen(layer.post_attention_layernorm_moe_gen(residual_gen))
    return und_seq, gen_seq, kv


class Cosmos3FastInference:
    """Exact fast forwards of one `Cosmos3CodecOPSDTransformer` (see the module docstring), for calls with one transformer
    pass per denoising step (video -> action: guidance 1). `Cosmos3CodecOPSDTransformer.fast_inference()` wraps one call."""

    def __init__(self):
        self.active = False
        self.graphs = {}  # "fill" / "use" -> captured graph, kept across calls
        self.fills = 0  # fill graphs captured so far: a "use" graph reads the text cache of the newest one
        self.stream = None
        self.stats = dict(forwards=0, preps=0, captures=0, replays=0, input_copies=0)
        self._prep = None  # this call: the fixed inputs, the sparse inputs and the kept rows
        self._text = None  # this call: the text cache {"kv": [(k, v) per layer], "out": the final text output}
        self._lora_state = None

    def reset(self):
        """Drop the graphs (they read the weights and LoRA factors at the addresses they had at capture)."""
        self.graphs = {}

    @contextlib.contextmanager
    def call(self, model):
        """One pipeline call: forwards of `model` run the fast path; the per-call caches are dropped at the end."""
        if self.active:
            raise RuntimeError("fast inference: nested call")
        if model.device.type != "cuda":
            raise RuntimeError(f"fast inference needs the transformer on a CUDA GPU, it is on {model.device}")
        if model._cp_shard_fn is not None or model._cp_gather_fn is not None:
            raise NotImplementedError("fast inference: context parallelism")
        for layer in model.layers:
            if not model._released_attention(layer.self_attn):
                raise NotImplementedError("fast inference: the released attention processor only")
        lora = model._lora_layers()
        self._lora_state = tuple((id(m), m.enabled, m.scale, m.down.data_ptr(), m.up.data_ptr()) for m in lora)
        rows = getattr(model, "_rows", None)
        heads = [getattr(model, name) for name in ("action_proj_in", "action_proj_out") if hasattr(model, name)]
        self.active = True
        try:
            if rows is not None:
                rows.share_cast = True
            for head in heads:
                head.forward = types.MethodType(_domain_linear_unchecked, head)
            yield self
        finally:
            self.active = False
            for head in heads:
                head.__dict__.pop("forward", None)
            if rows is not None:
                rows.share_cast, rows.cast = False, None
            self._prep = self._text = None

    # ------------------------------------------------------------------ once per call

    def _sparse(self, model, inputs, masks):
        """This call's sparse inputs and kept rows, computed at its first forward (the fixed inputs are the same objects
        at every step; anything else is a second pass per step, which this path does not run)."""
        fixed = {name: value for name, value in inputs.items() if name not in STEP_INPUTS}
        prep = self._prep
        if prep is not None:
            if prep["masks"] is masks and all(_unchanged(s, fixed[name]) for name, s in prep["fixed"].items()):
                return prep["sparse"], prep["keep"]
            raise NotImplementedError("fast inference: one transformer pass per denoising step (guidance 1)")
        self.stats["preps"] += 1
        sparse = model.sparse_inputs(inputs, masks)
        if sparse.layout is not None:
            raise NotImplementedError("fast inference: several transfer controls")
        model._set_layout(None)
        x, und = sparse.inputs, sparse.inputs["und_len"]
        # the text cache holds when the text prefix holds every text token and no generation token
        if not bool((x["text_indexes"] < und).all()):
            raise ValueError("fast inference: a text token lies outside the text prefix")
        for name in ("vision_sequence_indexes", "sound_sequence_indexes", "action_sequence_indexes"):
            if x[name] is not None and x[name].numel() and not bool((x[name] >= und).all()):
                raise ValueError(f"fast inference: a {name} entry lies in the text prefix")
        if x["action_domain_ids"] is not None:  # the range check the released DomainAwareLinear runs at every forward
            n = model.num_embodiment_domains
            for d in x["action_domain_ids"]:
                if bool(torch.any((d < 0) | (d >= n))):
                    raise ValueError(f"Cosmos3 action domain_id must be in [0, {n}), got {d.tolist()}.")
        keep = None if sparse.keep is None else sparse.keep.nonzero().squeeze(1)
        self._prep = dict(fixed={name: _snapshot(value) for name, value in fixed.items()}, masks=masks, sparse=sparse,
                          keep=keep)
        return sparse, keep

    # ------------------------------------------------------------------ every step

    def forward(self, model, inputs, masks):
        """One denoising step's forward: (vision, sound, action) predictions, as `forward(..., return_dict=False)`."""
        if torch.is_grad_enabled():
            raise RuntimeError("fast inference: inference only (torch.no_grad)")
        self.stats["forwards"] += 1
        sparse, keep = self._sparse(model, inputs, masks)
        step = dict(sparse.inputs)
        for name in STEP_INPUTS:
            step[name] = inputs[name]
        step[KEEP_ROWS] = keep
        return self._replay(model, "fill" if self._text is None else "use", step, sparse)

    def decoder_stack(self, model, und_seq, gen_seq, rotary_emb):
        """`_run_decoder_stack` with the text cache: filled by the first forward of a call, read by the later ones."""
        fill = self._text is None
        if fill:
            self._text = {"kv": [], "out": None}
        for i, layer in enumerate(model.layers):
            und_seq, gen_seq, kv = _decoder_layer(layer, und_seq, gen_seq, rotary_emb,
                                                  None if fill else self._text["kv"][i])
            if fill:
                self._text["kv"].append(kv)
        if fill:
            self._text["out"] = model.norm(und_seq)
        return self._text["out"], model.norm_moe_gen(gen_seq)

    def _run(self, model, inputs, sparse):
        """The released forward on the sparse inputs (eager; captured by `_capture`)."""
        inputs = dict(inputs)
        keep = inputs.pop(KEEP_ROWS)
        model._set_rows(sparse.rows)
        model._pending_keep = keep
        try:
            return Cosmos3OmniTransformer.forward(model, **inputs, return_dict=False)
        finally:
            model._pending_keep = None
            if getattr(model, "_rows", None) is not None:
                model._rows.cast = None

    # ------------------------------------------------------------------ CUDA graphs

    def _replay(self, model, mode, inputs, sparse):
        tensors, spec = _flatten(inputs)
        for x in tensors:
            if x.device != model.device:
                raise RuntimeError(f"fast inference: a transformer input on {x.device}")
        sig = (tuple((name, kind if isinstance(kind, str) else kind.__name__, ref if kind == "value" else None)
                     for name, kind, ref in spec),
               tuple((tuple(x.shape), x.stride(), x.dtype) for x in tensors), sparse.rows, self._lora_state,
               self.fills if mode == "use" else None)
        entry = self.graphs.get(mode)
        if entry is None or entry["sig"] != sig:
            entry = self._capture(model, mode, tensors, spec, sparse, sig)
        else:
            for i, x in enumerate(tensors):
                last, version = entry["last"][i]
                if x is not last or x._version != version:
                    entry["static"][i].copy_(x)
                    entry["last"][i] = (x, x._version)
                    self.stats["input_copies"] += 1
        entry["graph"].replay()
        self.stats["replays"] += 1
        if mode == "fill":
            self._text = entry["text"]  # the fill graph's outputs, refreshed by the replay
        return tuple(None if out is None else [x.clone() for x in out] for out in entry["out"])

    def _capture(self, model, mode, tensors, spec, sparse, sig):
        self.graphs.pop(mode, None)
        if mode == "fill":
            self.graphs.pop("use", None)  # it reads the old fill graph's text cache
        static = []
        for x in tensors:
            if any(s == 0 for s, n in zip(x.stride(), x.shape) if n > 1):
                raise NotImplementedError("fast inference: a transformer input with a zero stride")
            buffer = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
            buffer.copy_(x)
            static.append(buffer)
        inputs = _unflatten(spec, static)
        if self.stream is None:
            self.stream = torch.cuda.Stream(model.device)
        current = torch.cuda.current_stream(model.device)
        self.stream.wait_stream(current)
        text = self._text
        with torch.cuda.stream(self.stream):
            self._run(model, inputs, sparse)  # warm-up outside the capture (lazy initialization); outputs dropped
            self._text = text
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            try:
                out = self._run(model, inputs, sparse)
            finally:
                graph.capture_end()
        current.wait_stream(self.stream)
        entry = dict(graph=graph, sig=sig, static=static, last=[(x, x._version) for x in tensors], out=out,
                     text=self._text if mode == "fill" else None)
        self._text = text
        if mode == "fill":
            self.fills += 1  # a "use" graph is captured again: it must read this fill graph's text cache
        self.graphs[mode] = entry
        self.stats["captures"] += 1
        return entry
