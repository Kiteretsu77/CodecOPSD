"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The pipelines of the page: built once (`inference.load` with the page's hooks: the denoise timer, the kept masks),
placed by --offload, and the Teacher as the same transformer with the CodecOPSD LoRA switched off.
"""

import gc
import time
from pathlib import Path

import torch

from codecopsd.attention import set_attention

from ..load import build_pipeline, place
from ..utils import watch_encode
from . import state
from .state import LABELS, PARKED, PIPES, STATE


def build_pipe(name, text_encoder=None):
    """The pipeline of `name` (the released model + its CodecOPSD LoRA when present), in CPU memory, with the page's
    hooks. `text_encoder`: one already loaded (`preload`)."""
    args = state.ARGS
    lora = args.lora if args.lora and Path(args.lora).exists() else None
    pipe = build_pipeline(name, args.model, lora, text_encoder=args.text_encoder, sampler=getattr(args, "sampler", None),
                          sampler_weights=getattr(args, "flash_weights", None), shared_text_encoder=text_encoder,
                          text_encoder_path=getattr(args, "vlm_fp8", None))  # an FP8 VLM folder (main.py)
    if name == "h3_flash_ref2va" and getattr(args, "vae_encoder", "released") == "taeh3":  # the extreme path only
        from .. import extreme
        extreme.use_taeh3_encoder(pipe, extreme.taeh3_weights(download=not args.no_download))
    time_forward(pipe.transformer)  # before any offload hook, so that moving weights is not counted
    set_attention(pipe, "auto")  # cudnn on Hopper / Blackwell, sdpa on older GPUs
    capture_masks(pipe)
    count_work(pipe)
    return pipe


class Parked:
    """A transformer whose weights keep a pinned CPU copy: `gpu()` copies them to the GPU (about 50 GB/s on a B200,
    1-2 s for 62 GB), `cpu()` points them back at the CPU copy (nothing is copied)."""

    def __init__(self, module):
        self.tensors = [t for _, t in module.named_parameters()] + [t for _, t in module.named_buffers()]
        self.host = [t.detach().to("cpu").pin_memory() for t in self.tensors]
        self.cpu()

    def gpu(self):
        for t, host in zip(self.tensors, self.host):
            t.data = host.to("cuda", non_blocking=True)
        torch.cuda.synchronize()

    def cpu(self):
        for t, host in zip(self.tensors, self.host):
            t.data = host


def same_text_encoder(a, b):
    """Whether two release folders ship the same text encoder (same index and shard sizes; MiniMax-H3 and MiniMax-H3-Image
    ship byte-identical files)."""
    a, b, index = Path(a) / "text_encoder", Path(b) / "text_encoder", "model.safetensors.index.json"
    if not ((a / index).is_file() and (b / index).is_file()) or (a / index).read_bytes() != (b / index).read_bytes():
        return False
    return all((b / f.name).is_file() and (b / f.name).stat().st_size == f.stat().st_size
               for f in a.glob("*.safetensors"))


def preload(models):
    """--offload none: every MiniMax model loaded once before the demo opens (the text encoder once when the releases ship the
    same one). The text encoder and the VAEs go to the GPU; the transformers wait in pinned CPU memory until a request needs one."""
    text_encoder = first = None
    for name in models:
        folder, t = state.ARGS.model, time.perf_counter()
        share = text_encoder is not None and same_text_encoder(first, folder)
        pipe = build_pipe(name, text_encoder if share else None)
        for key, module in pipe.components.items():  # the text encoder and the VAE
            if isinstance(module, torch.nn.Module) and key != "transformer":
                module.to("cuda")
        PARKED[name], PIPES[name] = Parked(pipe.transformer), pipe
        if text_encoder is None:
            text_encoder, first = pipe.text_encoder, folder
        print(f"[app] {name} loaded in {time.perf_counter() - t:.0f} s" + (", text encoder shared" if share else ""),
              flush=True)


def load(name, report=None):
    """The pipeline of `name`, its transformer on the GPU."""
    if name in PIPES:  # --offload none: loaded at start, only the transformer moves
        if STATE["name"] != name:
            if STATE["name"] in PARKED:
                PARKED[STATE["name"]].cpu()
                torch.cuda.empty_cache()
            if report:
                report.stage("Switching the model on the GPU (1-2 s)")
            PARKED[name].gpu()
            STATE.update(name=name, pipe=PIPES[name])
        return PIPES[name]
    if STATE["name"] == name:
        return STATE["pipe"]
    if report:
        report.stage(f"Loading the {LABELS[name]} model onto the GPU (first use of this tab, 1-2 min)")
    if STATE["name"] in PARKED:
        PARKED[STATE["name"]].cpu()
    STATE.update(name=None, pipe=None)
    gc.collect()
    torch.cuda.empty_cache()
    pipe = build_pipe(name)
    place(pipe, state.ARGS.offload, getattr(state.ARGS, "stream_blocks", 0),  # pinned: the text encoder streamed
          pin_text_encoder=getattr(state.ARGS, "pin_vlm", True))
    if state.ARGS.offload not in ("none", "pinned", "stream"):  # diffusers' model offload
        from codecopsd.offload import time_model_offload
        time_model_offload(pipe)  # its model moves get their own row and show on the live line
    STATE.update(name=name, pipe=pipe)
    return pipe


def time_forward(module):
    """Add the GPU time of every forward of `module` (the transformer) to STATE["denoise"]."""
    forward = module.forward

    def timed(*args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = forward(*args, **kwargs)
        torch.cuda.synchronize()
        STATE["denoise"] = STATE.get("denoise", 0.0) + time.perf_counter() - t
        STATE["calls"] = STATE.get("calls", 0) + 1
        if STATE.get("on_step"):
            STATE["on_step"](STATE["calls"])
        return out

    module.forward = timed


def capture_masks(pipe):
    """Keep the gazer masks and references of the last `encode` (the pipeline calls it once per request)."""
    def keep(out):
        pipe._last_encode = out
        if STATE.get("on_encoded"):
            STATE["on_encoded"]()
    watch_encode(pipe, keep)


def count_work(pipe):
    """Count the work of the encode and decode stages for the progress bar, in `STATE["work"]` (dict(enc=[done,
    total], dec=[done, total]), reset by every run): one per VAE encoder or decoder call (a spatial tile of one temporal
    chunk) and one per VLM layer. The totals are known when the stage starts: the encode's from the references as the
    model reads them, the decode's from the latents."""
    vae = pipe.vae
    layers = pipe.text_encoder.model.language_model.layers
    encoding = [False]  # inside `pipe.encode` (the image pipeline prepares its pictures again to draw them)

    def add(key, i, n):
        work = STATE.get("work")
        if work is not None:
            work[key][i] += n

    def tiles(height, width):
        if not getattr(vae, "use_tiling", False):
            return 1
        count = lambda length, tile, overlap: len(vae._split_tiles(length, tile, overlap)[0])
        return (count(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
                * count(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width))

    whole = "encode" in vae.__dict__  # an encoder that takes a reference in one call (the extreme path's TAEH3)

    def encode_calls(frames, height, width):
        if whole:
            return 1
        clip = getattr(vae.config, "clip_length", 1)
        return (1 if frames == 1 else -(-frames // clip)) * tiles(height, width)

    def decode_calls(z):
        ratio = vae.spatial_compression_ratio
        calls = tiles(z.shape[-2] * ratio, z.shape[-1] * ratio)
        if z.ndim == 5 and hasattr(vae, "tokens_chunk_size"):  # the video VAE decodes in temporal chunks (its `_decode`)
            size, drop = vae.tokens_chunk_size, vae.config.token_drop
            tokens = z.shape[2] + drop
            calls *= (tokens + (-tokens) % size) // size - int(drop > 0)
        return calls

    def counted(method, total):
        def wrapped(*args, **kwargs):
            out = method(*args, **kwargs)
            if encoding[0]:
                try:
                    add("enc", 1, total(out))
                except Exception:  # an unknown layout: the bar shows no total for this stage
                    pass
            return out
        return wrapped

    encode = pipe.encode

    def encode_counted(*args, **kwargs):
        encoding[0] = True
        add("enc", 1, len(layers))
        try:
            return encode(*args, **kwargs)
        finally:
            encoding[0] = False
    pipe.encode = encode_counted
    if hasattr(pipe, "normalize_references"):  # Ref2VA: pictures (one frame) and videos as the model reads them
        size = lambda r: (1, r.image.size[1], r.image.size[0]) if r.kind == "image" else r.frames.shape[:3]
        pipe.normalize_references = counted(pipe.normalize_references,
                                            lambda refs: sum(encode_calls(*size(r)) for r in refs))
    elif hasattr(pipe, "prepare_references"):  # the image model: its pictures as the VAE reads them
        pipe.prepare_references = counted(pipe.prepare_references,
                                          lambda out: sum(encode_calls(1, p.size[1], p.size[0]) for p in out[0]))
    decode = vae.decode

    def decode_counted(z, *args, **kwargs):
        try:
            add("dec", 1, decode_calls(z))
        except Exception:
            pass
        return decode(z, *args, **kwargs)
    vae.decode = decode_counted
    vae.encoder.register_forward_pre_hook(lambda module, args: add("enc", 0, 1))
    vae.decoder.register_forward_pre_hook(lambda module, args: add("dec", 0, 1))
    for layer in layers:
        layer.register_forward_pre_hook(lambda module, args: add("enc", 0, 1))


def has_lora(pipe):
    return hasattr(pipe, "condition_lora_metadata")


def set_lora(pipe, on):
    """The teacher = the same transformer with the CodecOPSD LoRA switched off (the pipelines' `velocity`
    with `lora=False`, as in training: the teacher)."""
    pipe.__dict__.pop("velocity", None)
    if not on:
        velocity = type(pipe).velocity
        pipe.velocity = lambda *args, **kwargs: velocity(*args, **{**kwargs, "lora": False})
