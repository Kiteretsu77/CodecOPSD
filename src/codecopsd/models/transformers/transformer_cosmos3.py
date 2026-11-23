"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Cosmos 3 (Omni) transformer for CodecOPSD: the student reads only the condition tokens the codec gazer keeps, a condition
LoRA acts on those tokens only, and several transfer controls attend as the official Cosmos 3 framework does. Video ->
action calls can run an exact fast path (`fast_inference`, cosmos3_fast.py)."""

import contextlib
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from diffusers import Cosmos3OmniTransformer
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3AttnProcessor, _rotate_half
from torch import nn

from .cosmos3_fast import Cosmos3FastInference

# Generation-pathway projections of every layer that carry the condition LoRA. In the last layer the condition rows only
# feed the keys and values of the generated rows, so only those two carry it.
LORA_TARGETS = ("self_attn.add_q_proj", "self_attn.add_k_proj", "self_attn.add_v_proj", "self_attn.to_add_out",
                "mlp_moe_gen.gate_proj", "mlp_moe_gen.up_proj", "mlp_moe_gen.down_proj")
LAST_LAYER_TARGETS = ("self_attn.add_k_proj", "self_attn.add_v_proj")


class SparseInputs(NamedTuple):
    """`Cosmos3CodecOPSDTransformer.sparse_inputs`: the forward's inputs on the kept sequence, the kept vision tokens, the
    condition LoRA rows and the multi-control layout."""

    inputs: dict
    keep: torch.Tensor | None
    rows: tuple | None
    layout: dict | None


class _Rows:
    """The generation-pathway rows [start, end) that hold the condition tokens of the current forward, and the number
    of generation rows (shared by every LoRA layer). With `share_cast` (fast inference) the condition rows of an input
    are cast to the factors' dtype once, for every LoRA layer that reads the same input (q / k / v, gate / up); with
    bfloat16 factors there is nothing to cast."""

    start = end = total = None
    share_cast = False
    cast = None  # (input, its version, dtype, the cast rows) of the last cast


class ConditionRowLoRALinear(nn.Module):
    """A generation-pathway linear layer plus a LoRA applied only to the condition rows `rows.start:rows.end`.

    The factors keep their own dtype: bfloat16 for inference (`load_condition_lora` default; the scale is folded into
    `up`, so `scale` is 1), float32 for training (`init_condition_lora`) or with `load_condition_lora(..., float32)`.
    The condition rows are cast to that dtype (bfloat16: the rows as they are, the LoRA runs in the base layer's
    precision) and the LoRA output is added in place to the base output, which keeps the base dtype.
    """

    def __init__(self, base, down, up, scale, rows):
        super().__init__()
        self.base = base
        self.down = nn.Parameter(down, requires_grad=down.requires_grad)
        self.up = nn.Parameter(up, requires_grad=up.requires_grad)
        self.scale = scale
        self.rows = rows

    def forward(self, x):
        y = self.base(x)
        start, end = self.rows.start, self.rows.end
        if start is None or end <= start or not self.enabled:
            return y
        if x.shape[0] != self.rows.total:
            raise RuntimeError(f"condition LoRA: {x.shape[0]} rows, the forward has {self.rows.total} generation rows")
        lora = F.linear(F.linear(self._condition_rows(x, start, end), self.down), self.up)
        if self.scale != 1:
            lora = lora * self.scale
        if torch.is_grad_enabled():
            return torch.cat([y[:start], (y[start:end] + lora).to(y.dtype), y[end:]])
        y[start:end] += lora
        return y

    def _condition_rows(self, x, start, end):
        """`x[start:end]` in the factors' dtype; with `rows.share_cast` one cast per input tensor (same values)."""
        rows = self.rows
        if not rows.share_cast:
            return x[start:end].to(self.down.dtype)
        cast = rows.cast
        if cast is not None and cast[0] is x and cast[1] == x._version and cast[2] == self.down.dtype:
            return cast[3]
        out = x[start:end].to(self.down.dtype)
        rows.cast = (x, x._version, self.down.dtype, out)
        return out

    enabled = True


class MultiControlAttnProcessor(Cosmos3AttnProcessor):
    """The released attention, except with several transfer controls: the official framework's attention, in which
    control i attends to [text, control i, target] and the target attends to [text, control i, target] once per control
    and takes the weighted sum of the N outputs. The text pathway is unchanged. `state.layout` (set per forward) holds
    the control row ranges, the target rows and the normalized weights; without it this is the released processor."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def __call__(self, attn, und_seq, gen_seq, rotary_emb):
        layout = self.state.layout
        if layout is None:
            return super().__call__(attn, und_seq, gen_seq, rotary_emb)
        if attn.k_norm_und_for_gen is not None:
            raise NotImplementedError("multi-control attention: a separate text key norm for the generation pathway")
        heads, kv_heads, dim = attn.num_attention_heads, attn.num_key_value_heads, attn.head_dim
        q_und = attn.norm_q(attn.to_q(und_seq).view(-1, heads, dim))
        k_und = attn.norm_k(attn.to_k(und_seq).view(-1, kv_heads, dim))
        v_und = attn.to_v(und_seq).view(-1, kv_heads, dim)
        q_gen = attn.norm_added_q(attn.add_q_proj(gen_seq).view(-1, heads, dim))
        k_gen = attn.norm_added_k(attn.add_k_proj(gen_seq).view(-1, kv_heads, dim))
        v_gen = attn.add_v_proj(gen_seq).view(-1, kv_heads, dim)
        cos_und, sin_und, cos_gen, sin_gen = (r.unsqueeze(1) for r in rotary_emb)
        q_und, k_und = (x * cos_und + _rotate_half(x) * sin_und for x in (q_und, k_und))
        q_gen, k_gen = (x * cos_gen + _rotate_half(x) * sin_gen for x in (q_gen, k_gen))

        def attend(q, k, v, causal=False):
            out = dispatch_attention_fn(q[None], k[None], v[None], is_causal=causal, enable_gqa=True,
                                        backend=self._attention_backend, parallel_config=self._parallel_config)
            return out[0].flatten(-2, -1)

        text_out = attend(q_und, k_und, v_und, causal=True)
        (t0, t1), outs, target_out = layout["target"], [], None
        for (c0, c1), weight in zip(layout["controls"], layout["weights"]):
            k = torch.cat([k_und, k_gen[c0:c1], k_gen[t0:t1]])
            v = torch.cat([v_und, v_gen[c0:c1], v_gen[t0:t1]])
            if c1 > c0:
                outs.append(attend(q_gen[c0:c1], k, v))
            out = weight * attend(q_gen[t0:t1], k, v)
            target_out = out if target_out is None else target_out + out
        return attn.to_out(text_out), attn.to_add_out(torch.cat([*outs, target_out]))


class _LayoutState:
    layout = None


class Cosmos3CodecOPSDTransformer(Cosmos3OmniTransformer):
    r"""Cosmos 3 (Omni) transformer that runs a CodecOPSD student: sparse condition tokens plus a condition-row LoRA.

    Condition tokens = the clean vision tokens of a forward: every control item of a transfer pass, and the clean
    leading frames of the last vision item (the given image of image -> video, the given frames of a continuation, the
    whole input video of video -> audio and of inverse dynamics). `forward(..., condition_masks=[...])` (or
    `condition_masks` set on the model: a list, or a function returning it, called at the first forward) gives one bool
    [T, h, w] mask per clean
    block on its token grid (h, w = the latent size / 2, rounded up); the dropped tokens leave the joint sequence and the
    kept ones keep their dense rotary positions. A forward without clean tokens (the control-free guidance pass) is dense.
    The condition LoRA (`load_condition_lora`) acts on the kept condition rows only, so text, target, sound and action
    tokens run the released weights. With two or more controls the attention is the official framework's
    (`MultiControlAttnProcessor`), for the teacher and the student alike. Without masks and LoRA this is the released
    transformer, except for that multi-control attention. `fast_inference()` runs video -> action calls on an exact
    fast path (same outputs; `Cosmos3FastInference`).
    """

    condition_masks = None
    control_weights = None  # one weight per control of a multi-control forward (normalized); None = equal weights
    framework_multi_control = True

    # ---------------------------------------------------------------- condition LoRA

    def _lora_layers(self):
        return [m for m in self.modules() if isinstance(m, ConditionRowLoRALinear)]

    def _wrap(self, name, down, up, scale):
        parent, child = name.rsplit(".", 1)
        module = self.get_submodule(parent)
        base = getattr(module, child)
        base = base.base if isinstance(base, ConditionRowLoRALinear) else base
        setattr(module, child, ConditionRowLoRALinear(base, down, up, scale, self._rows))

    def load_condition_lora(self, state_dict, alpha, dtype=torch.bfloat16):
        """Wrap every projection named in `state_dict` (`layers.{i}.{module}.lora_{A,B}.weight`) with the frozen LoRA,
        for inference. `dtype` of the factors and of the LoRA computation: bfloat16 (default, the precision of the base
        model; the scale alpha / rank is folded into the up factor in float32 before the cast) or float32 (as trained;
        the scale is applied to the LoRA output)."""
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"condition LoRA dtype: float32 or bfloat16, got {dtype}")
        self._rows = getattr(self, "_rows", _Rows())
        self._drop_fast_graphs()
        names = sorted({key.rsplit(".lora_", 1)[0] for key in state_dict})
        for name in names:
            module = self.get_submodule(name)  # a LoRA loaded before: its wrapper, the base layer inside
            device = (module.base if isinstance(module, ConditionRowLoRALinear) else module).weight.device
            down = state_dict[f"{name}.lora_A.weight"].to(device, torch.float32)
            up = state_dict[f"{name}.lora_B.weight"].to(device, torch.float32)
            scale = alpha / down.shape[0]
            if dtype == torch.bfloat16:
                up, scale = up * scale, 1.0
            self._wrap(name, down.to(dtype), up.to(dtype), scale)
        return len(names)

    def init_condition_lora(self, rank, alpha, targets=LORA_TARGETS, last_layer_targets=LAST_LAYER_TARGETS, seed=0):
        """A new trainable float32 LoRA that starts as the released model (down ~ U(-1 / sqrt(in), 1 / sqrt(in)) as
        PEFT's default, up = 0); returns its parameters."""
        self._rows = getattr(self, "_rows", _Rows())
        self._drop_fast_graphs()
        generator = torch.Generator().manual_seed(seed)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            for target in last_layer_targets if i == last else targets:
                base = layer.get_submodule(target)
                base = getattr(base, "base", base)
                bound = 1.0 / math.sqrt(base.in_features)
                down = (torch.rand(rank, base.in_features, generator=generator) * 2 - 1) * bound
                up = torch.zeros(base.out_features, rank)
                device = base.weight.device
                self._wrap(f"layers.{i}.{target}", down.to(device).requires_grad_(), up.to(device).requires_grad_(),
                           alpha / rank)
        return [p for layer in self._lora_layers() for p in (layer.down, layer.up)]

    def condition_lora_state_dict(self):
        """The LoRA in the file format of `load_condition_lora` (float32). Sharded (FSDP) factors are gathered: call it
        on every process. The float32 LoRA only (training, or loaded with float32): a bfloat16 inference LoRA holds
        rounded factors with the scale folded into `up`."""
        out = {}
        for name, module in self.named_modules():
            if isinstance(module, ConditionRowLoRALinear):
                if module.down.dtype != torch.float32 or module.up.dtype != torch.float32:
                    raise RuntimeError(f"condition_lora_state_dict: {name} has {module.down.dtype} factors (a bfloat16 "
                                       "inference LoRA); export the float32 LoRA")
                for key, factor in (("lora_A", module.down), ("lora_B", module.up)):
                    factor = factor.detach()
                    factor = factor.full_tensor() if hasattr(factor, "full_tensor") else factor
                    out[f"{name}.{key}.weight"] = factor.float().cpu().contiguous()
        return out

    def set_condition_lora_enabled(self, enabled):
        """Turn the condition LoRA off (the teacher: the released model) or on (the student)."""
        for layer in self._lora_layers():
            layer.enabled = enabled

    # ---------------------------------------------------------------- multi-control attention

    def _set_layout(self, layout):
        state = getattr(self, "_layout_state", None)
        if state is None:
            if layout is None:
                return
            state = self._layout_state = _LayoutState()
            for module in self.modules():
                if hasattr(module, "processor") and type(module.processor) is Cosmos3AttnProcessor:
                    processor = MultiControlAttnProcessor(state)
                    processor._attention_backend = module.processor._attention_backend
                    processor._parallel_config = module.processor._parallel_config
                    module.set_processor(processor)
        state.layout = layout  # kept after the forward: gradient checkpointing recomputes the layers with it

    # ---------------------------------------------------------------- sparse condition tokens

    def _apply_timestep_embeds_to_noisy_tokens(self, packed_tokens, *args, **kwargs):
        tokens = super()._apply_timestep_embeds_to_noisy_tokens(packed_tokens, *args, **kwargs)
        keep, self._pending_keep = getattr(self, "_pending_keep", None), None  # the vision tokens: the first call
        if keep is None:
            return tokens
        # a bool mask, or (fast inference) the kept rows as indices: the same gather, without a host sync
        return tokens[keep] if keep.dtype == torch.bool else tokens.index_select(0, keep)

    def sparse_inputs(self, inputs, masks):
        """The released forward's inputs (a dict) with the condition tokens that `masks` drops removed.

        Returns `SparseInputs`: `inputs` (the dropped tokens' rows and rotary positions removed, every index into the
        sequence renumbered), `keep` (bool over the packed vision tokens, None = all kept), `rows` ((start, end, total)
        of the kept condition rows among the generation rows, None without a condition LoRA) and `layout` (the
        multi-control attention layout, None for one control)."""
        inputs = dict(inputs)
        und_len, sequence_length = inputs["und_len"], inputs["sequence_length"]
        vision_sequence_indexes = inputs["vision_sequence_indexes"]
        shapes = [tuple(int(s) for s in shape) for shape in inputs["vision_token_shapes"]]
        sizes = [math.prod(shape) for shape in shapes]
        # clean blocks in packing order: every earlier item whole, then the last item's clean leading frames
        for shape, noisy in zip(shapes[:-1], inputs["vision_noisy_frame_indexes"][:-1]):
            if noisy.numel():
                raise ValueError(f"a vision item before the last has noisy frames: {shape}")
        frames, h, w = shapes[-1]
        noisy = inputs["vision_noisy_frame_indexes"][-1]
        clean = frames - noisy.numel()
        if not torch.equal(noisy.cpu(), torch.arange(clean, frames)):
            raise ValueError(f"the clean frames of the last vision item must lead, noisy frames {noisy.tolist()}")
        blocks = [*shapes[:-1], (clean, h, w)] if clean else shapes[:-1]
        keep = None
        if masks is not None and blocks:
            if [tuple(m.shape) for m in masks] != blocks:
                raise ValueError(f"condition masks {[tuple(m.shape) for m in masks]} != clean blocks {blocks}")
            device = vision_sequence_indexes.device
            parts = [m.reshape(-1).to(device, torch.bool) for m in masks]
            parts.append(torch.ones((frames - clean) * h * w, dtype=torch.bool, device=device))
            keep = torch.cat(parts)
            if keep.all():
                keep = None
        num_condition = sum(math.prod(b) for b in blocks)
        if keep is not None:  # drop the tokens' rows and rotary positions; renumber every index into the sequence
            joint = torch.ones(sequence_length, dtype=torch.bool, device=keep.device)
            joint[vision_sequence_indexes[~keep]] = False
            renumber = torch.cumsum(joint.long(), 0) - 1
            inputs["position_ids"] = inputs["position_ids"][..., joint]
            inputs["sequence_length"] = sequence_length = int(joint.sum())
            inputs["vision_sequence_indexes"] = vision_sequence_indexes = renumber[vision_sequence_indexes[keep]]
            for key in ("text_indexes", "vision_mse_loss_indexes", "sound_sequence_indexes", "sound_mse_loss_indexes",
                        "action_sequence_indexes", "action_mse_loss_indexes"):
                if inputs[key] is not None:
                    inputs[key] = renumber[inputs[key]]
            num_condition = int(keep[:num_condition].sum())
        # the kept condition tokens are the first vision rows: one block of generation rows
        rows = None
        if getattr(self, "_rows", None) is not None:
            start = end = 0
            if num_condition:
                first = vision_sequence_indexes[:num_condition]
                if not torch.equal(first, torch.arange(int(first[0]), int(first[0]) + num_condition,
                                                       device=first.device)) or int(first[0]) < und_len:
                    raise ValueError("the condition tokens are not one block of generation rows")
                start, end = int(first[0]) - und_len, int(first[0]) - und_len + num_condition
            rows = (start, end, sequence_length - und_len)
        layout = None
        controls = len(shapes) - 1
        if self.framework_multi_control and controls >= 2 and inputs["sound_tokens"] is None \
                and inputs["action_tokens"] is None:
            kept = sizes if keep is None else [int(p.sum()) for p in torch.split(keep, sizes)]
            if sequence_length - und_len != sum(kept) or not torch.equal(
                    vision_sequence_indexes, torch.arange(und_len, sequence_length, device=vision_sequence_indexes.device)):
                raise ValueError("multi-control attention: the generation rows must be exactly the vision tokens")
            weights = self.control_weights or [1.0] * controls
            if len(weights) != controls or min(weights) < 0 or sum(weights) <= 0:
                raise ValueError(f"control weights {weights} for {controls} controls")
            bounds = [0, *torch.tensor(kept).cumsum(0).tolist()]
            layout = dict(controls=[(bounds[i], bounds[i + 1]) for i in range(controls)],
                          target=(bounds[-2], bounds[-1]), weights=[float(x) / sum(weights) for x in weights])
        return SparseInputs(inputs, keep, rows, layout)

    def _set_rows(self, rows):
        if rows is not None:
            self._rows.start, self._rows.end, self._rows.total = rows

    def forward(self, input_ids, text_indexes, position_ids, und_len, sequence_length, vision_tokens,
                vision_token_shapes, vision_sequence_indexes, vision_mse_loss_indexes, vision_timesteps,
                vision_noisy_frame_indexes, sound_tokens=None, sound_token_shapes=None, sound_sequence_indexes=None,
                sound_mse_loss_indexes=None, sound_timesteps=None, sound_noisy_frame_indexes=None, action_tokens=None,
                action_token_shapes=None, action_sequence_indexes=None, action_mse_loss_indexes=None,
                action_timesteps=None, action_noisy_frame_indexes=None, action_domain_ids=None, return_dict=True,
                condition_masks=None):
        # the released pipeline passes only the arguments named in this signature: keep it the released one
        inputs = dict(input_ids=input_ids, text_indexes=text_indexes, position_ids=position_ids, und_len=und_len,
                      sequence_length=sequence_length, vision_tokens=vision_tokens,
                      vision_token_shapes=vision_token_shapes, vision_sequence_indexes=vision_sequence_indexes,
                      vision_mse_loss_indexes=vision_mse_loss_indexes, vision_timesteps=vision_timesteps,
                      vision_noisy_frame_indexes=vision_noisy_frame_indexes, sound_tokens=sound_tokens,
                      sound_token_shapes=sound_token_shapes, sound_sequence_indexes=sound_sequence_indexes,
                      sound_mse_loss_indexes=sound_mse_loss_indexes, sound_timesteps=sound_timesteps,
                      sound_noisy_frame_indexes=sound_noisy_frame_indexes, action_tokens=action_tokens,
                      action_token_shapes=action_token_shapes, action_sequence_indexes=action_sequence_indexes,
                      action_mse_loss_indexes=action_mse_loss_indexes, action_timesteps=action_timesteps,
                      action_noisy_frame_indexes=action_noisy_frame_indexes, action_domain_ids=action_domain_ids)
        masks = self.condition_masks if condition_masks is None else condition_masks
        if callable(masks):  # computed while the pipeline encoded: wait for them at the first forward
            masks = self.condition_masks = masks()
        fast = getattr(self, "_fast", None)
        if fast is not None and fast.active:
            if return_dict:
                raise NotImplementedError("fast inference: return_dict=False (the released denoiser) only")
            return fast.forward(self, inputs, masks)
        sparse = self.sparse_inputs(inputs, masks)
        self._set_rows(sparse.rows)
        self._set_layout(sparse.layout)
        self._pending_keep = sparse.keep
        try:
            return super().forward(**sparse.inputs, return_dict=return_dict)
        finally:
            self._pending_keep = None

    # ---------------------------------------------------------------- exact fast inference (video -> action)

    @contextlib.contextmanager
    def fast_inference(self):
        """Within the block, forwards run `Cosmos3FastInference` (exact: the same kernels on the same values; one
        transformer pass per denoising step, as video -> action). Its CUDA graphs are kept for the next block."""
        if getattr(self, "_fast", None) is None:
            self._fast = Cosmos3FastInference()
        with self._fast.call(self):
            yield self._fast

    def _drop_fast_graphs(self):
        if getattr(self, "_fast", None) is not None:
            self._fast.reset()

    def _released_attention(self, attn):
        """Whether `attn` runs the released attention (the multi-control processor without a layout does)."""
        processor = attn.processor
        if type(processor) is MultiControlAttnProcessor:
            return processor.state.layout is None
        return type(processor) is Cosmos3AttnProcessor

    def _run_decoder_stack(self, und_seq, gen_seq, rotary_emb):
        fast = getattr(self, "_fast", None)
        if fast is not None and fast.active:
            return fast.decoder_stack(self, und_seq, gen_seq, rotary_emb)
        return super()._run_decoder_stack(und_seq, gen_seq, rotary_emb)
