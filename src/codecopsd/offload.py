"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Pinned model offload: one model on the GPU at a time, every other one in page-locked CPU memory.

Before a model runs, the model on the GPU leaves it (nothing is copied back: inference never writes the weights, so its
tensors are pointed back at their CPU copy), then the model that runs is copied in from page-locked memory (about
50 GB/s on PCIe 5, 25 GB/s on PCIe 4). Peak GPU memory is the largest model plus its activations. The order follows
the calls of the pipeline; free memory is never looked at, so placements and outputs are the same on every run.
The scheme follows `offload.py` of hyperflow-h3-flash (Apache-2.0); diffusers' model offload moves weights from
pageable memory instead, about 10 times slower.

Layer streaming (`stream`): the layers of a model's block list (the text encoder's decoder layers) do not move with
the model; each is copied in just before it runs, the next one ahead on a side stream while this one computes, and
dropped after it. The model then needs the rest of its weights plus two layers on the GPU (the FP8 text encoder:
about 3 GB instead of 26 GB), for about 0.5 s more per call.

Parts (`parts`): a model can also move in parts, each copied in when one of its modules runs. The video VAE moves
as its encoder (with `quant_conv`) while the references are encoded and as its decoder (with `post_quant_conv`) for the
decode, so the 4.5 GB decoder is never on the GPU during the reference encoding.

Every move is timed (`MoveLog`), so that a request can show how long the moves took (`codecopsd.stats`).

File mappings (`map_safetensors`): the FP8 text encoder (VLM) is loaded as views of a read-only shared mapping of its
.safetensors files, so its weights are the page cache's pages: no copy in process memory, and no commit charge under
strict overcommit (vm.overcommit_memory=2). `pin` page-locks such weights where they are (read-only registration, no
copy) and copies only the other tensors into its page-locked buffer. A model left `pageable` keeps its mapped weights
in the page cache instead, read again at each move: about 26 GB less locked CPU memory, slower moves.
"""

import ctypes
import functools
import itertools
import json
import mmap
import os
import struct
import time
import warnings
import weakref

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from accelerate.utils import send_to_device
from torch import nn

_ALIGN = 256
_PAGE = mmap.PAGESIZE
_MADV_DONTFORK = 10  # Linux
_HOST_REGISTER_READ_ONLY = 8  # cudaHostRegisterReadOnly: page-lock memory mapped read-only
_DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}
_MAPPINGS = {}  # base address -> bytes of each live file mapping made by `map_safetensors`
_LOCKED = {}  # first page -> end of each page range page-locked in place (`_lock_mapped`)
LOCKED_BYTES = {"copied": 0, "mapped": 0}  # page-locked so far: copies in `pin` buffers, file pages locked in place


class _Mapping(mmap.mmap):
    """A file mapping that can be watched (`weakref`), so that `_MAPPINGS` forgets it once unmapped."""


def map_safetensors(path):
    """The tensors of a .safetensors file as views of one read-only shared mapping of it (MAP_SHARED, PROT_READ):
    nothing is copied or converted, and the pages are the page cache's, so the load takes no commit charge under
    strict overcommit (safetensors maps files private and writable, which counts their whole size). The weights must
    never be written: inference only reads them (a write would fault)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mapping = _Mapping(fd, size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    length = struct.unpack("<Q", mapping[:8])[0]
    header = json.loads(mapping[8 : 8 + length])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "the given buffer is not writable": the weights are only read
        flat = torch.frombuffer(mapping, dtype=torch.uint8)  # keeps the mapping alive while any view lives
    _MAPPINGS[flat.data_ptr()] = size
    weakref.finalize(mapping, _MAPPINGS.pop, flat.data_ptr(), None)
    start = 8 + length
    out = {}
    for name, info in header.items():
        if name != "__metadata__":
            a, b = info["data_offsets"]
            out[name] = flat[start + a : start + b].view(_DTYPES[info["dtype"]]).view(info["shape"])
    return out


def _clear_cuda_error():
    """Reset the CUDA runtime's last error after a failed registration, so that PyTorch's next launch check does not
    report it."""
    try:
        ctypes.CDLL("libcudart.so.12").cudaGetLastError()
    except (OSError, AttributeError):
        pass


def is_mapped(tensor):
    """Whether `tensor` views a file mapping of `map_safetensors`."""
    return tensor.untyped_storage().data_ptr() in _MAPPINGS


def _lock_mapped(module, tensors):
    """Page-lock the file pages that `tensors` (views of `map_safetensors` mappings) read, where they are: read-only
    registration, no copy, no commit charge. Pages locked already (a neighbour's) are skipped. Returns the tensors that
    could not be locked (`cudaHostRegisterReadOnly` unsupported): `pin` copies those."""
    ranges = sorted((t.data_ptr() // _PAGE * _PAGE, -(-(t.data_ptr() + t.numel() * t.element_size()) // _PAGE) * _PAGE)
                    for t in tensors if t.numel())
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    for first, end in _LOCKED.items():  # skip the pages locked already
        merged = [piece for start, stop in merged for piece in ([start, min(stop, first)], [max(start, end), stop])
                  if piece[0] < piece[1]]
    cudart = torch.cuda.cudart()
    done = []
    for start, end in merged:
        if int(cudart.cudaHostRegister(start, end - start, _HOST_REGISTER_READ_ONLY)):
            _clear_cuda_error()
            for first, stop in done:  # undo: the caller copies these tensors instead
                cudart.cudaHostUnregister(first)
                _LOCKED.pop(first)
            LOCKED_BYTES["mapped"] -= sum(stop - first for first, stop in done)
            return tensors
        _LOCKED[start] = end
        LOCKED_BYTES["mapped"] += end - start
        done.append((start, end))
    if done:
        storages = [t.untyped_storage() for t in tensors]  # unmapped only after the pages are unlocked

        def unlock(pages, storages):
            for first, stop in pages:
                cudart.cudaHostUnregister(first)
                _LOCKED.pop(first, None)
                LOCKED_BYTES["mapped"] -= stop - first

        weakref.finalize(module, unlock, done, storages).atexit = False
    return []


def _tensors(module):
    return itertools.chain(module.parameters(), module.buffers())


def _no_fork_copy(arena):
    """Keep page-locked memory out of forked children (`MADV_DONTFORK`). Linux copies pinned pages into a forked
    child at once instead of sharing them, so a fork (imageio-ffmpeg starts ffmpeg with `preexec_fn`, which forks)
    would need the whole arena a second time: 47 GB more CPU memory for the Flash pipeline. The child only runs
    another program, which never reads the parent's memory."""
    libc = ctypes.CDLL(None, use_errno=True)
    start = -(-arena.data_ptr() // _PAGE) * _PAGE
    end = (arena.data_ptr() + arena.numel()) // _PAGE * _PAGE
    if end > start and libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), _MADV_DONTFORK):
        warnings.warn(f"madvise(MADV_DONTFORK) failed (errno {ctypes.get_errno()}): a fork copies the pinned memory")


def pin(module, pageable=False):
    """Page-lock the CPU tensors of `module`: those that view a file mapping (`map_safetensors`) where they are
    (`_lock_mapped`; left in the page cache with `pageable`), every other one moved into one page-locked buffer of
    exactly their size (`Tensor.pin_memory` rounds every allocation up to a power of two, about a quarter more host
    memory). Returns the bytes copied into the buffer."""
    tensors, mapped, seen = [], [], set()
    for tensor in _tensors(module):
        if tensor.device.type == "cpu" and not tensor.is_pinned() and id(tensor) not in seen:
            seen.add(id(tensor))
            (mapped if is_mapped(tensor) else tensors).append(tensor)
    if mapped and not pageable:
        tensors += _lock_mapped(module, mapped)
    if not tensors:
        return 0
    sizes = [t.numel() * t.element_size() for t in tensors]
    offsets = list(itertools.accumulate((-(-size // _ALIGN) * _ALIGN for size in sizes[:-1]), initial=0))
    arena = torch.empty(offsets[-1] + sizes[-1], dtype=torch.uint8)
    for tensor, offset, size in zip(tensors, offsets, sizes, strict=True):
        view = arena[offset : offset + size].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor.data)
        tensor.data = view
    cudart = torch.cuda.cudart()
    if error := int(cudart.cudaHostRegister(arena.data_ptr(), arena.numel(), 0)):
        raise RuntimeError(f"pinning {arena.numel() / 2**30:.1f} GiB of CPU memory failed (CUDA error {error})")
    _no_fork_copy(arena)
    weakref.finalize(module, cudart.cudaHostUnregister, arena.data_ptr()).atexit = False
    module._pinned_arena = arena
    LOCKED_BYTES["copied"] += arena.numel()
    return arena.numel()


class MoveLog:
    """Seconds and bytes of every model moved to the GPU (`seconds` grows; readers take differences)."""

    def __init__(self):
        self.seconds, self.bytes, self.moves = 0.0, 0, []
        self.now = None  # (model name, start time) of the move in progress, for a live display

    def add(self, name, seconds, nbytes):
        self.seconds += seconds
        self.bytes += nbytes
        self.moves.append((name, seconds, nbytes))


class PinnedSwap:
    """A model's weights in page-locked CPU memory, with a GPU copy made on `load` and dropped on `drop`. Tensors in
    `exclude` (ids; streamed layers) stay where they are; with `pageable` its file-mapped weights stay in the page
    cache (`pin`)."""

    def __init__(self, name, module, device, log, exclude=(), pageable=False):
        pin(module, pageable)
        self.name, self.device, self.log = name, torch.device(device), log
        self.host = [(t, t.data) for t in _tensors(module) if id(t) not in exclude]
        self.nbytes = sum(h.numel() * h.element_size() for _, h in self.host)
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        self.log.now = (self.name, t)
        for tensor, host in self.host:
            tensor.data = host.to(self.device, non_blocking=True)
        torch.cuda.synchronize(self.device)
        self.log.now = None
        self.log.add(self.name, time.perf_counter() - t, self.nbytes)
        self.loaded = True

    def drop(self):
        if self.loaded:
            for tensor, host in self.host:
                tensor.data = host
            self.loaded = False


class LayerStream:
    """The layers of a block list in page-locked CPU memory, each copied to the GPU just before it runs (the next one
    ahead on a side stream) and dropped after it. Same weights, same products: exact."""

    def __init__(self, layers, device, log=None, name="layers"):
        self.device, self.log, self.name = torch.device(device), log, name
        self.layers = [[(t, t.data) for t in _tensors(layer)] for layer in layers]
        self.stream = torch.cuda.Stream(self.device)
        self.ready = {}
        for i, layer in enumerate(layers):
            layer.register_forward_pre_hook(functools.partial(self._before, i))
            layer.register_forward_hook(functools.partial(self._after, i))

    def tensor_ids(self):
        return {id(t) for layer in self.layers for t, _ in layer}

    def _fetch(self, i):
        if i >= len(self.layers) or i in self.ready:
            return
        with torch.cuda.stream(self.stream):
            copies = [host.to(self.device, non_blocking=True) for _, host in self.layers[i]]
            event = torch.cuda.Event()
            event.record(self.stream)
        self.ready[i] = (copies, event)

    def _before(self, i, module, args):
        self._fetch(i)
        copies, event = self.ready.pop(i)
        compute = torch.cuda.current_stream(self.device)
        compute.wait_event(event)
        for (tensor, _), copy in zip(self.layers[i], copies, strict=True):
            copy.record_stream(compute)  # its memory is reused only after this layer's work on the compute stream
            tensor.data = copy
        self._fetch(i + 1)

    def _after(self, i, module, args, output):
        for tensor, host in self.layers[i]:
            tensor.data = host


class OneOnGPU(ModelHook):
    """Before its model runs: every other swapped model leaves the GPU, then this one is copied in."""

    def __init__(self, device, swap, swaps):
        self.execution_device = torch.device(device)
        self.swap, self.swaps = swap, swaps

    def pre_forward(self, module, *args, **kwargs):
        dropped = False
        for other in self.swaps:
            if other is not self.swap and other.loaded:
                other.drop()
                dropped = True
        if dropped:  # hand the dropped model's blocks back, so the cache does not keep two models' worth of memory
            torch.cuda.empty_cache()
        self.swap.load()
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class InParts(ModelHook):
    """The hook of a model that moves in parts (`pinned_model_offload(parts=...)`). It moves nothing itself: each part
    has its own `OneOnGPU` hook. It tells diffusers where the model runs (`execution_device`) and lets `offload_all`
    drop every part."""

    def __init__(self, device, swaps):
        self.execution_device = torch.device(device)
        self.swaps = swaps


def pinned_model_offload(models, device="cuda", log=None, stream=None, parts=None, pageable=()):
    """Place `models` ({name: nn.Module}, in CPU memory) for pinned model offload on `device`; `stream` ({name: block
    list}) streams those layers of a model one by one (`LayerStream`) instead of moving them with it; `pageable`
    (names) leaves the file-mapped weights of those models in the page cache instead of page-locking them; `parts`
    ({name: [[module, ...], ...]}) moves a model in parts instead of whole: each part (a list of submodules, which
    together must hold every tensor of the model) is copied in when one of its modules runs, so only the part that runs
    is on the GPU (the video VAE: the encoder while the references are encoded, the decoder for the decode). Returns the
    `MoveLog`."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    log = log or MoveLog()
    swaps = []
    for name, module in models.items():
        if not isinstance(module, nn.Module):
            continue
        module.to("cpu")
        if parts and name in parts:
            groups = [nn.ModuleList(part) for part in parts[name]]
            if {id(t) for g in groups for t in _tensors(g)} != {id(t) for t in _tensors(module)}:
                raise ValueError(f"the parts of {name!r} must hold every tensor of it")
            own = []
            for group in groups:
                swap = PinnedSwap(name, group, device, log, pageable=name in pageable)
                swap.group = group  # keeps the page-locked buffer registered: `pin` unregisters it with its module
                own.append(swap)
                for part in group:
                    add_hook_to_module(part, OneOnGPU(device, swap, swaps))
            swaps.extend(own)
            add_hook_to_module(module, InParts(device, own))
            continue
        exclude = ()
        if stream and name in stream:
            pin(module, name in pageable)
            exclude = LayerStream(stream[name], device, log, name).tensor_ids()
        swap = PinnedSwap(name, module, device, log, exclude, name in pageable)
        swaps.append(swap)
        add_hook_to_module(module, OneOnGPU(device, swap, swaps))
    return log


def offload_all(models):
    """Every model of a pinned model offload back to its CPU copy (frees the GPU between requests)."""
    for module in models:
        hook = getattr(module, "_hf_hook", None)
        if isinstance(hook, OneOnGPU):
            hook.swap.drop()
        elif isinstance(hook, InParts):
            for swap in hook.swaps:
                swap.drop()


def time_model_offload(pipe):
    """Time the moves of diffusers' model offload (`pipe.enable_model_cpu_offload`, accelerate `CpuOffload` hooks) into
    `pipe.moves` (a `MoveLog`), as the pinned offload does: before a model runs, the previous one goes back to CPU
    memory and this one is copied in; `maybe_free_model_hooks` (after the denoising and at the end of a call) sends
    every model back and hooks them again, so the new hooks are timed again after it. Call after
    `enable_model_cpu_offload`. Returns the `MoveLog`."""
    log = pipe.moves = getattr(pipe, "moves", None) or MoveLog()
    enable = pipe.enable_model_cpu_offload

    def timed(name, fn, /, *args, **kwargs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        log.now = (name, t)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            log.now = None
            seconds = time.perf_counter() - t
            if seconds > 0.005:  # a model already in place moves nothing (its hook runs before every call)
                log.add(name, seconds, 0)

    def hook_all():
        names = {id(m): n for n, m in pipe.components.items()}
        for user_hook in getattr(pipe, "_all_hooks", []):
            hook = user_hook.hook
            hook.pre_forward = functools.partial(timed, names.get(id(user_hook.model), "model"), hook.pre_forward)

    def enable_again(*args, **kwargs):
        timed("every model back to the CPU", enable, *args, **kwargs)
        hook_all()

    pipe.enable_model_cpu_offload = enable_again
    hook_all()
    return log
