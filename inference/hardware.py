"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The machine a run is on, and what the code picks for it (run from the repository root):

    python -m inference.hardware      # the whole report, with the warnings that apply

`detect()` reads the GPU (name, compute capability, total and free memory), the CPU cores this process may use, the
host memory (total, available, the cgroup limit that Slurm's --mem sets), the kernel's overcommit mode and commit
headroom, and the torch, CUDA and driver versions. `report()` adds what follows from them; each rule lives in src/ and
is only called here, so it exists once:

    attention kernel ("auto")       codecopsd.attention.auto_backend: cuDNN from compute capability 9, SDPA below
    INT8 GEMMs of HyperFlow Flash   ...transformer_h3_flash.int8_gemm_choice: Triton on 12, cuBLAS on 8.6 / 8.9
    FP8 VLM products                codecopsd.models.fp8_linear.fp8_tensor_cores: FP8 from 8.9, bf16 below

`check()` gives one line per problem that applies: no GPU, a GPU under 12 GB, less CPU memory than the pinned plans
keep, too little commit headroom to load the VLM under strict overcommit. The demo and the command line print the
report and these lines at start, on stdout (`announce`). `cap_gpu_memory` caps the process to a smaller GPU.

Memory sizes are in GB of 2**30 bytes, as PyTorch and the GPU memory plans count them.
"""

import ctypes
import os
from pathlib import Path

import torch

GB = 2**30
# A GPU sold as 12 GB shows PyTorch 11.7 to 12.0 GB; the smallest plan (inference/extreme.py) needs a 12 GB card.
MIN_GPU_GB = 11.5


def cap_gpu_memory(gpu_memory=None):
    """The first GPU's memory in GB (0 without a GPU). A smaller `gpu_memory` (GB) caps the process to it, the CUDA
    context included, so that a run shows what a card of that size does."""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0.0
    if gpu_memory and gpu_memory < total:  # PyTorch's allocator gets the card minus the CUDA context
        torch.cuda.set_per_process_memory_fraction((gpu_memory - 1.5) / total)
    return total


def vlm_files_gb(folder):
    """The size in GB of the VLM (text encoder) files of a MiniMax-H3 release folder (`<folder>/text_encoder/`, bf16 as
    released), or None. Loading maps them before the FP8 conversion: under strict overcommit the commit headroom must
    hold them, else the load fails with ENOMEM."""
    files = list((Path(folder) / "text_encoder").glob("*.safetensors")) if folder else []
    return sum(f.stat().st_size for f in files) / GB if files else None


# ------------------------------------------------------------------------------------------------------- detection

def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _meminfo():
    """/proc/meminfo in bytes ({} where there is none)."""
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, value = line.partition(":")
        if value.split():
            out[key] = int(value.split()[0]) * 1024
    return out


def _cgroup_memory():
    """(limit, in use) in bytes of the tightest memory cgroup around this process (Slurm puts a job's --mem there), or
    None. In use = usage minus the inactive file cache, which the kernel drops first."""
    best = None
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            root, files = Path("/sys/fs/cgroup"), ("memory.max", "memory.current", "inactive_file")
        elif "memory" in controllers.split(","):  # cgroup v1
            root, files = Path("/sys/fs/cgroup/memory"), ("memory.limit_in_bytes", "memory.usage_in_bytes",
                                                          "total_inactive_file")
        else:
            continue
        d = root / path.lstrip("/")
        while True:
            limit, usage = _read(d / files[0]), _read(d / files[1])
            if limit and limit.isdigit() and int(limit) < 2**60 and usage and usage.isdigit():
                stat = dict(s.split() for s in (_read(d / "memory.stat") or "").splitlines() if len(s.split()) == 2)
                used = int(usage) - int(stat.get(files[2], 0))
                if best is None or int(limit) - used < best[0] - best[1]:
                    best = (int(limit), used)
            if d == root or root not in d.parents:
                break
            d = d.parent
    return best


def _driver_version():
    """The NVIDIA driver version (NVML, else /proc/driver/nvidia/version), or None."""
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() == 0:
            try:
                buffer = ctypes.create_string_buffer(96)
                if nvml.nvmlSystemGetDriverVersion(buffer, ctypes.c_uint(96)) == 0:
                    return buffer.value.decode()
            finally:
                nvml.nvmlShutdown()
    except (OSError, AttributeError):
        pass
    text = _read("/proc/driver/nvidia/version") or ""
    words = text.split("Kernel Module")[-1].split() if "Kernel Module" in text else []
    return words[0] if words else None


def _env_int(name):
    value = os.environ.get(name, "")
    return int(value) if value.isdigit() else None


def detect():
    """What this process runs on, as a dict: gpu (None without one: else name, capability, total_gb, free_gb, count),
    cpu (cores, machine, slurm), ram (total_gb, available_gb, cgroup_gb, cgroup_used_gb), overcommit
    (mode, limit_gb, committed_gb, headroom_gb) and versions (torch, cuda, cudnn, driver, triton, alloc_conf)."""
    gpu = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free = torch.cuda.mem_get_info(0)[0]
        gpu = dict(name=props.name, capability=(props.major, props.minor), total_gb=props.total_memory / GB,
                   free_gb=free / GB, count=torch.cuda.device_count())
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        cores = os.cpu_count()
    mem = _meminfo()
    cgroup = _cgroup_memory()
    try:
        triton = __import__("triton").__version__
    except ImportError:
        triton = None
    mode = _read("/proc/sys/vm/overcommit_memory")
    return dict(
        gpu=gpu,
        cpu=dict(cores=cores, machine=os.cpu_count(), slurm=_env_int("SLURM_CPUS_PER_TASK")),
        ram=dict(total_gb=mem.get("MemTotal", 0) / GB or None, available_gb=mem.get("MemAvailable", 0) / GB or None,
                 cgroup_gb=cgroup and cgroup[0] / GB, cgroup_used_gb=cgroup and cgroup[1] / GB),
        overcommit=dict(mode=int(mode) if mode and mode.isdigit() else None, limit_gb=mem.get("CommitLimit", 0) / GB,
                        committed_gb=mem.get("Committed_AS", 0) / GB,
                        headroom_gb=(mem.get("CommitLimit", 0) - mem.get("Committed_AS", 0)) / GB if mem else None),
        versions=dict(torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version() if gpu else None, driver=_driver_version() if gpu else None,
                      triton=triton, alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF")),
    )


def available_ram_gb(info):
    """CPU memory this process can still take in GB: the machine's available memory, or less under a cgroup limit."""
    ram = info["ram"]
    values = [v for v in (ram["available_gb"], ram["cgroup_gb"] and ram["cgroup_gb"] - ram["cgroup_used_gb"]) if v]
    return min(values) if values else None


# ------------------------------------------------------------------------------------------------- what follows

def decisions(info):
    """What the code picks on this GPU: {"attention", "int8_gemm", "fp8_vlm"}, each a short text."""
    from codecopsd.attention import auto_backend
    from codecopsd.models.fp8_linear import fp8_tensor_cores
    from codecopsd.models.transformers.transformer_h3_flash import int8_gemm_choice

    gpu = info["gpu"]
    if gpu is None:
        return dict(attention=auto_backend() + " (no GPU)", int8_gemm="no GPU", fp8_vlm="no GPU")
    capability = gpu["capability"]
    choice = int8_gemm_choice(capability)
    if choice is not None:
        int8 = f"{choice} (set by CodecOPSD on compute capability {capability[0]}.{capability[1]})"
    else:
        from codecopsd.models.transformers.hyperflow_flash.int8 import resolve_int8_gemm
        int8 = f"{resolve_int8_gemm(torch.device('cuda', 0))} (HyperFlow H3 Flash's own choice)"
    fp8 = ("FP8 products on the FP8 tensor cores" if fp8_tensor_cores(capability)
           else "bf16 products from the FP8 weights (no FP8 tensor cores below compute capability 8.9)")
    return dict(attention=f"{auto_backend()} (auto: cudnn from compute capability 9, sdpa below)", int8_gemm=int8,
                fp8_vlm=fp8)


def report(info=None):
    """The whole report as text, one fact per line."""
    info = info or detect()
    gpu, cpu, ram, over, v = info["gpu"], info["cpu"], info["ram"], info["overcommit"], info["versions"]
    f = lambda x: "?" if x is None else f"{x:.1f}"  # noqa: E731
    lines = []
    if gpu:
        lines.append(f"GPU: {gpu['name']}, compute capability {gpu['capability'][0]}.{gpu['capability'][1]}, "
                     f"{f(gpu['total_gb'])} GB, {f(gpu['free_gb'])} GB free" + (f" ({gpu['count']} GPUs visible, "
                     "the first one runs)" if gpu["count"] > 1 else ""))
    else:
        lines.append("GPU: none visible to PyTorch")
    lines.append(f"CPU: {cpu['cores']} cores for this process (of {cpu['machine']})"
                 + (f", Slurm CPUs per task {cpu['slurm']}" if cpu["slurm"] else ""))
    limit = (f"; cgroup limit (Slurm --mem) {f(ram['cgroup_gb'])} GB, {f(ram['cgroup_used_gb'])} GB in use"
             if ram["cgroup_gb"] else "; no cgroup memory limit")
    lines.append(f"CPU memory: {f(ram['total_gb'])} GB total, {f(ram['available_gb'])} GB available{limit}")
    modes = {0: "heuristic", 1: "always", 2: "strict"}
    if over["mode"] is not None:
        lines.append(f"overcommit: vm.overcommit_memory={over['mode']} ({modes.get(over['mode'], '?')}), commit "
                     f"headroom {f(over['headroom_gb'])} GB (CommitLimit {f(over['limit_gb'])} GB minus Committed_AS "
                     f"{f(over['committed_gb'])} GB" + (", counted only when strict)" if over["mode"] != 2 else ")"))
    lines.append(f"software: torch {v['torch']}, CUDA {v['cuda']}, cuDNN {v['cudnn']}, driver {v['driver']}, "
                 f"triton {v['triton']}; PYTORCH_CUDA_ALLOC_CONF={v['alloc_conf'] or '(unset)'}")
    d = decisions(info)
    lines += [f"attention kernel: {d['attention']}", f"INT8 GEMMs (HyperFlow H3 Flash): {d['int8_gemm']}",
              f"FP8 text encoder (VLM): {d['fp8_vlm']}"]
    return "\n".join(f"[hardware] {line}" for line in lines)


def check(info=None, ram_gb=None, commit_gb=None):
    """One line per problem that applies: no GPU; a GPU under 12 GB; less CPU memory available than `ram_gb` (what a
    pinned plan keeps in CPU memory, `extreme.PINNED_RAM_GB`); strict overcommit with less commit headroom than
    `commit_gb` (the VLM files, `vlm_files_gb`)."""
    info = info or detect()
    gpu, over, out = info["gpu"], info["overcommit"], []
    if gpu is None:
        out.append("no CUDA GPU: every model here needs an NVIDIA GPU with 12 GB or more")
    elif gpu["total_gb"] < MIN_GPU_GB:
        out.append(f"the GPU has {gpu['total_gb']:.1f} GB: the smallest plan needs a 12 GB card (README: Consumer GPUs)")
    available = available_ram_gb(info)
    if ram_gb and available is not None and available < ram_gb:
        out.append(f"{available:.1f} GB of CPU memory available, the pinned plan keeps about {ram_gb:.1f} GB in CPU "
                   "memory (most of it page-locked): loading may fail; free memory or ask for more (Slurm --mem)")
    if commit_gb and over["mode"] == 2 and over["headroom_gb"] is not None and over["headroom_gb"] < commit_gb:
        out.append(f"strict overcommit (vm.overcommit_memory=2) with {over['headroom_gb']:.1f} GB of commit headroom: "
                   f"loading the bf16 VLM maps its {commit_gb:.1f} GB of files and fails with ENOMEM; download the FP8 "
                   "VLM (python -m codecopsd.download), or use a machine with more commit headroom (CommitLimit minus "
                   "Committed_AS)")
    return out


def announce(ram_gb=None, commit_gb=None, info=None):
    """Print the report and the warnings that apply (`check`) on stdout; the start of a demo or command-line run."""
    info = info or detect()
    print(report(info), flush=True)
    for line in check(info, ram_gb, commit_gb):
        print(f"[hardware] warning: {line}", flush=True)
    return info


def main():
    """`python -m inference.hardware`: the report, with the warnings of the 12-32 GB plan this GPU would get (its CPU
    memory) and of loading the bf16 VLM of models/MiniMax-H3-Ref2VA (commit headroom; none with an FP8 VLM on disk,
    `codecopsd.download.vlm_fp8_source`)."""
    from codecopsd.download import model_dir, vlm_fp8_source

    from .extreme import PINNED_RAM_GB, memory_plan
    total = cap_gpu_memory()
    pinned = total and memory_plan(total)[1]["offload"] == "pinned"
    bf16 = vlm_fp8_source()[0] is None  # converted at load: its bf16 files are mapped first
    announce(ram_gb=PINNED_RAM_GB if pinned else None,
             commit_gb=vlm_files_gb(model_dir("MiniMax-H3-Ref2VA")) if bf16 else None)


if __name__ == "__main__":
    main()
