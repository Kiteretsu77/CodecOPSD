"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

CodecOPSD training of a MiniMax-H3 student (settings: src/codecopsd/configs/minimax_h3_ref2va_train.yaml).

Teacher = the released Ref2VA transformer with every reference token; student = the same frozen weights + a LoRA on the
reference rows, given only the reference tokens the codec gazer keeps. Per update the student rolls out k of the 16
sampling steps from noise (no gradients); at that state its velocity is matched to the teacher's (MSE). No video
targets are needed: a sample is a prompt and its references.

    # 1. encode every sample once: text embedding, VAE reference latents, codec gazer masks (one GPU per process)
    python scripts/train_minimax_h3_ref2va.py encode --model MiniMax-H3-Ref2VA --data samples.jsonl --out encoded --shard 0/1
    # 2. train the LoRA, one sample per GPU (run the same command again to resume)
    accelerate launch --num_processes 4 scripts/train_minimax_h3_ref2va.py train --model MiniMax-H3-Ref2VA --encoded encoded \\
        --out runs/codecopsd

On GPUs below 120 GB (A100 / H100 80 GB) the transformer is sharded over the GPUs (FSDP, `--fsdp auto`): each GPU
keeps 1 / n of every block and gathers a block only while it runs. Above that every GPU holds the whole transformer.

Data path: DataLoader workers (`--workers` per GPU) load the encoded samples, gaze every reference again from its file
(`--live-gaze`, on by default: the masks the transformer gets are computed while training, and must equal the masks the
student's text encoder embedding was made with), build the packed layouts and hand pinned tensors to the GPU, ahead of the
update that uses them.

A line of samples.jsonl: {"prompt": "...", "references": ["person.png", "motion.mp4"], "frames": 56, "height": 480,
"width": 640}, references in prompt order; frames / height / width default to a draw from the config's shapes.
Checkpoints: <out>/lora_<update>.safetensors (load with `MiniMaxH3CodecOPSDPipeline.load_condition_lora`) and
<out>/state.pt (resume: run the same command again, on the same number of GPUs). <out>/log.jsonl: one line per update
(loss of every sample, k, shape, time per stage, peak memory).
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import video_latent_num_frames
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.fsdp import fully_shard
from torch.utils.data import DataLoader, Dataset

from codecopsd import MiniMaxH3CodecOPSDPipeline, MiniMaxH3CodecOPSDTransformer3DModel
from codecopsd.codec_gazer import MINIMAX_H3, submit_gaze
from codecopsd.configs import load
from codecopsd.pipelines.minimax_h3_ref2va import load_reference
from codecopsd.pipelines.minimax_h3_ref2va.pipeline_minimax_h3_ref2va import CONFIG as INFERENCE

CFG = load("minimax_h3_ref2va_train")
P = MiniMaxH3CodecOPSDPipeline
TEXT_ENCODER_FP8 = {"bf16": None, "fp8": "fp8", "fp8-storage": "bf16"}  # `encode --text-encoder` -> text_encoder_fp8


def use_config(cfg, path):
    """Replace the training settings `cfg` (in place) with the YAML file `path` (none: keep the packaged config)."""
    if path:
        new = yaml.safe_load(open(path))
        cfg.clear()
        cfg.update(new)


def draw_shape(rng):
    data = CFG["data"]
    frames = rng.choices(list(data["frames"]), weights=list(data["frames"].values()))[0]
    size = rng.choices(list(data["sizes"]), weights=list(data["sizes"].values()))[0]
    return int(frames), *map(int, size.split("x"))


def encode(args):
    shard, shards = map(int, args.shard.split("/"))
    pipe = P.from_minimax_h3(args.model, load_transformer=False, text_encoder_fp8=TEXT_ENCODER_FP8[args.text_encoder])
    if torch.cuda.get_device_properties(0).total_memory >= 120 * 2**30:
        pipe.to("cuda")  # the text encoder (50 GB, FP8 26 GB) and the VAE (10 GB) stay on the GPU and encode at the same time
    else:
        pipe.enable_model_cpu_offload()  # one after the other (large pictures need the room on 80 GB)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(open(args.data)):
        if i % shards != shard or not line.strip() or list(out.glob(f"{i:07d}_*.pt")):
            continue
        sample = json.loads(line)
        frames, height, width = draw_shape(random.Random(f"{CFG['seed']}/{i}"))
        frames, height, width = sample.get("frames", frames), sample.get("height", height), sample.get("width", width)
        vlm_drop = CFG["codecopsd"]["vlm_drop"]
        tic = time.time()
        enc = pipe.encode(sample["prompt"], [load_reference(p) for p in sample["references"]], frames,
                          vlm_drop=vlm_drop, teacher_text=vlm_drop)
        masks = enc["masks"]
        torch.save({"text": (enc["teacher_text"] if vlm_drop else enc["text"]).cpu(), "text_tags": enc["text_tags"],
                    "student_text": enc["text"].cpu() if vlm_drop else None, "text_keep": enc["text_keep"],
                    "latents": [z.cpu() for z in enc["latents"]], "masks": masks, "vlm_masks": enc["vlm_masks"],
                    "kinds": [r.kind for r in enc["references"]], "references": sample["references"],
                    "frames": frames, "height": height, "width": width, "text_encoder": args.text_encoder},
                   out / f"{i:07d}_{frames}x{height}x{width}.pt")
        vlm = f", student text encoder rows {int(enc['text_keep'].sum())} of {len(enc['text_keep'])}" if vlm_drop else ""
        print(f"{i}: {len(masks)} references, kept {[round(float(m.float().mean()), 3) for m in masks]}{vlm}, "
              f"{time.time() - tic:.1f} s", flush=True)


def reference_pixels(reference):
    """The uint8 tensor `MiniMaxH3CodecOPSDPipeline.encode` gazes: [T, 3, H, W] for a normalized video, [3, H, W] for a
    picture."""
    if reference.kind == "video":
        return torch.from_numpy(np.ascontiguousarray(reference.frames)).permute(0, 3, 1, 2)
    return torch.from_numpy(np.array(reference.image)).permute(2, 0, 1)


def gaze_again(sample, pool):
    """The `Gaze` of every reference of an encoded sample, gazed again from its file on the CPU with the settings of
    `encode` (threshold decision; the student text encoder's mask with VLM drop)."""
    references = P.normalize_references(None, [load_reference(p) for p in sample["references"]], sample["frames"])
    futures = [submit_gaze(pool, reference_pixels(r), INFERENCE["references"]["gaze_kinds"][r.kind], MINIMAX_H3,
                           vlm=CFG["codecopsd"]["vlm_drop"]) for r in references]
    return [f.result() for f in futures]


def check_gazes(sample, gazes, files):
    """The live masks (`gazes`, of the reference `files`) must equal the encoded ones: the student's text encoder embedding
    was made with them. They then replace them."""
    for i, g in enumerate(gazes):
        vlm = sample["vlm_masks"][i] if sample.get("vlm_masks") else None
        if not torch.equal(g.keep, sample["masks"][i]) or (vlm is not None and not torch.equal(g.vlm_keep, vlm)):
            raise RuntimeError(f"live gaze of {files[i]} differs from its encoded mask: the reference changed since "
                               "`encode` (encode again), or the gazer settings did")
    sample["masks"] = [g.keep for g in gazes]


def prepare(sample, live_gaze, pool):
    """CPU part of a training sample (DataLoader worker): the masks (gazed again with `live_gaze`, checked against the
    stored ones) and the dense / sparse packed layouts."""
    if live_gaze:
        check_gazes(sample, gaze_again(sample, pool), sample["references"])
    shape = (1, 24, video_latent_num_frames(sample["frames"], 17, 5), sample["height"] // 16, sample["width"] // 16)
    keep = torch.cat([m.flatten() for m in sample["masks"]])
    args = (sample["text_tags"], sample["kinds"], sample["latents"], shape, sample["frames"])
    sample["shape"], sample["keep"] = shape, keep
    sample["dense"] = P.layout(*args)
    sample["sparse"] = P.layout(*args, keep, sample.get("text_keep"))
    if sample.get("student_text") is None:
        sample["student_text"] = sample["text"]
    return sample


def codecopsd_loss(model, transformer, scheduler, sample, k, generator, device, timer):
    """MSE between the student's and the teacher's velocity at the student's state after k of the rollout steps.
    `sample` comes from `prepare` (on `device`)."""
    latents, keep, shape = sample["latents"], sample["keep"], sample["shape"]
    text = sample["text"]  # the teacher's prompt embedding (every vision token)
    student_text = sample["student_text"]  # with VLM drop only the gazed vision rows (dense rotary positions kept)
    rows, reference_rows = P.pack(latents, shape, generator)
    kept_rows = reference_rows[keep]
    scheduler.set_timesteps(CFG["codecopsd"]["rollout_steps"] + 1, device=device)
    with torch.no_grad():
        for t in scheduler.timesteps[:k]:  # the student's own rollout
            velocity = P.velocity(transformer, rows, kept_rows, student_text, sample["sparse"], t)
            rows = scheduler.step(velocity.float(), t, rows, return_dict=False)[0]
        timer.mark("rollout")
        t = scheduler.timesteps[k]
        target = P.velocity(transformer, rows, reference_rows, text, sample["dense"], t, lora=False)  # teacher, dense
        timer.mark("teacher")
    prediction = P.velocity(model, rows, kept_rows, student_text, sample["sparse"], t)  # with gradients
    return F.mse_loss(prediction.float(), target.float())


def shard(transformer):
    """FSDP: every GPU keeps 1 / n of each block (and of its LoRA, gradients and optimizer states) and gathers a block
    only while it runs. Every weight keeps its dtype (no mixed-precision cast), so the maths are those of one GPU."""
    for block in transformer.transformer_blocks:
        fully_shard(block)
    fully_shard(transformer)


class Timer:
    """CUDA events at the stage boundaries of an update; read after the update has synchronized."""

    def __init__(self):
        self.marks = []

    def mark(self, name):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.marks.append((name, event))

    def seconds(self):
        out = defaultdict(float)
        for (_, a), (name, b) in zip(self.marks, self.marks[1:]):
            out[name] += a.elapsed_time(b) / 1000
        return dict(out)


class Samples(Dataset):
    """This GPU's samples in update order: item i = (update, micro step, file); `prepare_fn` runs in the workers."""

    def __init__(self, items, prepare_fn, live_gaze):
        self.items, self.prepare_fn, self.live_gaze, self.pool = items, prepare_fn, live_gaze, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        update, micro, path, index = self.items[i]
        if self.pool is None:  # one per worker: the references of a sample are gazed in parallel (x264 outside the GIL)
            self.pool = ThreadPoolExecutor(8)
        tic = time.time()
        sample = self.prepare_fn(torch.load(path, map_location="cpu", weights_only=False), self.live_gaze, self.pool)
        return {"update": update, "micro": micro, "index": index, "prepare_s": time.time() - tic, "sample": sample}


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def train(args, cfg=CFG, transformer_cls=MiniMaxH3CodecOPSDTransformer3DModel, subfolder="transformer_ref", loss_fn=None,
          lora_meta=None, prepare_fn=None):
    """The training loop. `cfg` (training config), `transformer_cls` / `subfolder`, `loss_fn` (default `codecopsd_loss`),
    `prepare_fn` (default `prepare`) and `lora_meta` (extra LoRA metadata) make it the loop of another student
    (scripts/train_minimax_h3_image.py)."""
    loss_fn, prepare_fn = loss_fn or codecopsd_loss, prepare_fn or prepare
    batch, opt_cfg = cfg["batch"], cfg["optimizer"]
    if args.deterministic:  # bitwise reproducible runs and resumes (the attention backward about 1.7x slower)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    accelerator = Accelerator(gradient_accumulation_steps=batch["gradient_accumulation"])
    device, rank, world = accelerator.device, accelerator.process_index, accelerator.num_processes
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fsdp = args.fsdp == "on" or (args.fsdp == "auto" and torch.cuda.get_device_properties(device).total_memory < 120 * 2**30)
    state = torch.load(out / "state.pt", map_location="cpu", weights_only=False) if (out / "state.pt").exists() else None
    if state and state.get("world", world) != world:
        raise SystemExit(f"{out} was trained on {state['world']} GPUs: resume on the same number (now {world})")

    transformer = transformer_cls.from_pretrained(args.model, subfolder=subfolder, torch_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    torch.manual_seed(cfg["seed"])
    transformer.init_condition_lora(cfg["lora"]["rank"], cfg["lora"]["alpha"], cfg["lora"]["targets"],
                                    cfg["lora"]["last_block_targets"])
    init = out / f"lora_{state['update']:06d}.safetensors" if state else args.init_lora
    if init:  # resume: the LoRA of the last checkpoint; or a LoRA to start from (e.g. a released student)
        lora = load_file(init)
        loaded = 0
        for name, module in transformer.named_modules():
            if hasattr(module, "down") and f"{name}.lora_A.weight" in lora:
                module.down.data.copy_(lora[f"{name}.lora_A.weight"])
                module.up.data.copy_(lora[f"{name}.lora_B.weight"])
                loaded += 1
        if loaded * 2 != len(lora):
            raise SystemExit(f"{init}: {len(lora)} tensors, {loaded * 2} match this LoRA")
    if args.adaln == "table":  # exact: the AdaLN input depends on the timestep alone, and the rollout grid is fixed
        transformer.table_adaln()  # every block's AdaLN projection (26 GB) in pinned CPU memory, a table per timestep
    transformer.enable_gradient_checkpointing()
    if fsdp:
        shard(transformer)
    else:
        transformer.to(device)
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], betas=opt_cfg["betas"], eps=opt_cfg["eps"],
                                  weight_decay=opt_cfg["weight_decay"])
    full = StateDictOptions(full_state_dict=True, cpu_offload=True)
    start = 0
    if state:  # resume: optimizer, update count
        set_optimizer_state_dict(transformer, optimizer, state["optimizer"], options=full)
        start = state["update"]
    if fsdp:
        model, optimizer = transformer, accelerator.prepare(optimizer)
    else:
        model, optimizer = accelerator.prepare(transformer, optimizer)
    accelerator.print(f"{world} GPUs, {'FSDP' if fsdp else 'data parallel'}, update {start} of {cfg['updates']}"
                      f"{f', LoRA from {init}' if init else ''}", flush=True)
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.model, subfolder="scheduler")

    # Samples grouped by shape; every GPU of an update takes a sample of the same shape (no GPU waits for a longer one).
    files = sorted(Path(args.encoded).glob("*.pt"))
    index = {f: i for i, f in enumerate(files)}  # a sample's noise is a function of the seed, the update and the sample
    buckets = defaultdict(list)
    for f in files:
        buckets[f.stem.split("_", 1)[1]].append(f)
    shapes = sorted(buckets)
    for shape in shapes:
        random.Random(f"{cfg['seed']}/{shape}").shuffle(buckets[shape])
    cursor = defaultdict(int)
    plan = []  # (shape, k, offset) of every update, a pure function of the seed: a resumed run continues the same data
    for update in range(cfg["updates"]):
        rng = random.Random(f"{cfg['seed']}/update/{update}")
        shape = rng.choices(shapes, weights=[len(buckets[s]) for s in shapes])[0]
        plan.append((shape, rng.randrange(cfg["codecopsd"]["rollout_steps"]), cursor[shape]))
        cursor[shape] += batch["gradient_accumulation"] * world
    items = []  # this GPU's samples, in order, from the first update to run
    for update in range(start, cfg["updates"]):
        shape, _, offset = plan[update]
        for micro in range(batch["gradient_accumulation"]):
            f = buckets[shape][(offset + micro * world + rank) % len(buckets[shape])]
            items.append((update, micro, str(f), index[f]))
    loader = DataLoader(Samples(items, prepare_fn, args.live_gaze == "on"), batch_size=None, shuffle=False,
                        num_workers=args.workers, pin_memory=True, prefetch_factor=2 if args.workers else None,
                        persistent_workers=args.workers > 0)
    samples = iter(loader)

    lora_meta = {"lora_rank": str(cfg["lora"]["rank"]), "lora_alpha": str(float(cfg["lora"]["alpha"])), "gazer": "codec",
                 "vlm_drop": str(bool(cfg["codecopsd"]["vlm_drop"])).lower(), **(lora_meta or {})}
    log = open(out / "log.jsonl", "a") if accelerator.is_main_process else None
    model.train()
    for update in range(start, cfg["updates"]):
        shape, k, _ = plan[update]
        tic, losses, seen, timer, waits = time.time(), [], [], Timer(), defaultdict(float)
        torch.cuda.reset_peak_memory_stats(device)
        for micro in range(batch["gradient_accumulation"]):
            wait = time.time()
            item = next(samples)
            assert (item["update"], item["micro"]) == (update, micro), (item["update"], item["micro"], update, micro)
            waits["data_wait"] += time.time() - wait
            waits["prepare_cpu"] += item["prepare_s"]
            timer.mark("start")
            sample = to_device(item["sample"], device)
            seed = (cfg["seed"] * 1_000_003 + update) * 1_000_003 + item["index"]
            with accelerator.accumulate(model):
                timer.mark("to_gpu")
                loss = loss_fn(model, transformer, scheduler, sample, k, torch.Generator(device).manual_seed(seed),
                               device, timer)
                timer.mark("student_forward")
                accelerator.backward(loss)
                timer.mark("student_backward")
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(params, opt_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                timer.mark("optimizer")
            losses.append(loss.detach())
            seen.append(item["index"])
        norm = float(norm.full_tensor() if hasattr(norm, "full_tensor") else norm)  # before clipping
        per_sample = accelerator.gather(torch.stack(losses)).view(world, -1)
        indices = accelerator.gather(torch.tensor(seen, device=device)).view(world, -1)
        peak = accelerator.gather(torch.tensor([[torch.cuda.max_memory_allocated(device),
                                                 torch.cuda.max_memory_reserved(device)]], device=device)).amax(0)
        peak, reserved = peak.tolist()
        stages = {n: round(v, 3) for n, v in {**timer.seconds(), **waits}.items()}
        busy = sum(stages.get(n, 0.0) for n in ("rollout", "teacher", "student_forward"))  # before the gradient sync
        busy = accelerator.gather(torch.tensor([busy], device=device)).tolist()
        seconds = time.time() - tic
        loss = per_sample.mean().item()
        accelerator.print(f"update {update + 1}/{cfg['updates']}  loss {loss:.5f}  grad norm {norm:.3f}  k {k}  "
                          f"shape {shape}  {seconds:.1f} s  ("
                          + ", ".join(f"{n} {v:.2f}" for n, v in stages.items()) + ")  "
                          f"peak {peak / 2**30:.1f} GiB (reserved {reserved / 2**30:.1f})", flush=True)
        if log:
            log.write(json.dumps({"update": update + 1, "loss": loss, "grad_norm": norm, "k": k, "shape": shape,
                                  "end": time.time(), "seconds": seconds, "stages": stages, "peak_gib": peak / 2**30,
                                  "reserved_gib": reserved / 2**30, "rank_forward_seconds": busy, "world": world,
                                  "fsdp": fsdp,
                                  "samples": indices.tolist(), "losses": per_sample.tolist()}) + "\n")
            log.flush()
        if (update + 1) % cfg["save_every"] == 0 or update + 1 == cfg["updates"]:
            lora = transformer.condition_lora_state_dict()  # every GPU: gathers sharded factors
            optimizer_state = get_optimizer_state_dict(transformer, optimizer.optimizer, options=full)
            if accelerator.is_main_process:  # written to a temporary name first: a kill never leaves half a file
                save_file(lora, out / "lora.tmp", lora_meta)
                os.replace(out / "lora.tmp", out / f"lora_{update + 1:06d}.safetensors")
                torch.save({"update": update + 1, "world": world, "optimizer": optimizer_state}, out / "state.tmp")
                os.replace(out / "state.tmp", out / "state.pt")
            accelerator.wait_for_everyone()
    accelerator.end_training()


def parser(doc, model_help):
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("encode", help="encode the samples (text encoder, VAE, codec gazer)")
    e.add_argument("--model", required=True, help=model_help)
    e.add_argument("--data", required=True, help="samples.jsonl")
    e.add_argument("--out", required=True)
    e.add_argument("--shard", default="0/1", help="i/n: encode every n-th sample from i (one process per GPU)")
    e.add_argument("--config", help="training settings (YAML); default: the packaged config")
    e.add_argument("--text-encoder", choices=list(TEXT_ENCODER_FP8), default="fp8",
                   help="text encoder linears, as in inference: fp8 (default): FP8 weights and FP8 products, 26.2 GB; "
                        "fp8-storage: FP8 weights, bf16 products; bf16: as released, 50.3 GB")
    t = sub.add_parser("train", help="train the CodecOPSD LoRA (accelerate launch)")
    t.add_argument("--model", required=True, help=model_help)
    t.add_argument("--encoded", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config", help="training settings (YAML); default: the packaged config")
    t.add_argument("--fsdp", choices=["auto", "on", "off"], default="auto",
                   help="shard the transformer over the GPUs (auto: on GPUs below 120 GB)")
    t.add_argument("--adaln", choices=["table", "weights"], default="table",
                   help="table (default): every block's AdaLN projection as a per-timestep table, its weights in pinned "
                        "CPU memory (exact, 26 GB less GPU memory); weights: the projections on the GPU as released")
    t.add_argument("--init-lora", help="start from this LoRA (.safetensors) instead of a new one (not when resuming)")
    t.add_argument("--workers", type=int, default=2, help="DataLoader workers per GPU (0: load in the training loop)")
    t.add_argument("--deterministic", action="store_true",
                   help="deterministic kernels: a run and its resumes repeat bit for bit (slower; for checks)")
    t.add_argument("--live-gaze", choices=["on", "off"], default="on",
                   help="gaze every reference again while training and check it against the encoded mask (on), or "
                        "use the encoded masks (off; the reference files are then not needed)")
    return p


def main():
    args = parser(__doc__, "the released MiniMax-H3 folder").parse_args()
    use_config(CFG, args.config)
    encode(args) if args.command == "encode" else train(args)


if __name__ == "__main__":
    main()
