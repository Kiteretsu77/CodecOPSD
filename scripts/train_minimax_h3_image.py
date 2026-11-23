"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

CodecOPSD training of a MiniMax-H3-Image student (settings: src/codecopsd/configs/minimax_h3_image_train.yaml).

Teacher = the released transformer with every reference token; student = the same frozen weights + a LoRA on the
reference rows, given only the reference tokens the codec gazer keeps. Per update one state k of the released six-step
grid is drawn: the student takes k steps from noise (no gradients), then at that state its velocity is matched to the
teacher's (MSE). No target images: a sample is a prompt and its pictures.

    # 1. encode every sample once: text embedding, VAE reference latents, codec gazer masks (one GPU per process)
    python scripts/train_minimax_h3_image.py encode --model MiniMax-H3-Image --data samples.jsonl --out encoded
    # 2. train the LoRA, one sample per GPU (run the same command again to resume)
    accelerate launch --num_processes 4 scripts/train_minimax_h3_image.py train --model MiniMax-H3-Image \\
        --encoded encoded --out runs/codecopsd_image

A line of samples.jsonl: {"prompt": "...", "images": ["person.png", "scene.png"], "height": 2048, "width": 2048},
pictures in prompt order (`<Picture 1>`, ...); height / width default to 2048 x 2048. The training loop (FSDP below
120 GB, DataLoader workers with the live gaze, checkpoints <out>/lora_<update>.safetensors, resume, <out>/log.jsonl) is
the one of scripts/train_minimax_h3_ref2va.py; every GPU of an update takes a sample with the same canvas and number of
pictures.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import get_image_canvas_size
from PIL import Image
from train_minimax_h3_ref2va import TEXT_ENCODER_FP8, check_gazes, parser, train, use_config  # the shared training loop

from codecopsd import MiniMaxH3ImageCodecOPSDPipeline, MiniMaxH3ImageCodecOPSDTransformer2DModel
from codecopsd.codec_gazer import MINIMAX_H3, submit_gaze
from codecopsd.configs import load

CFG = load("minimax_h3_image_train")
INFERENCE = load("minimax_h3_image")
P = MiniMaxH3ImageCodecOPSDPipeline
RESIZER = SimpleNamespace(image_processor=VaeImageProcessor(vae_scale_factor=16, do_normalize=False))


def vlm_scale():
    """The student text encoder's picture threshold scale, or None without VLM drop."""
    return CFG["codecopsd"]["vlm_picture_threshold_scale"] if CFG["codecopsd"]["vlm_drop"] else None


def encode(args):
    shard, shards = map(int, args.shard.split("/"))
    pipe = P.from_minimax_h3_image(args.model, load_transformer=False,  # text encoder 50 GB (FP8 26 GB) + VAE 2 GB
                                   text_encoder_fp8=TEXT_ENCODER_FP8[args.text_encoder]).to("cuda")
    vlm = vlm_scale()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(open(args.data)):
        if i % shards != shard or not line.strip() or list(out.glob(f"{i:07d}_*.pt")):
            continue
        sample = json.loads(line)
        height, width = get_image_canvas_size(sample.get("height", CFG["data"]["default_size"][0]),
                                              sample.get("width", CFG["data"]["default_size"][1]),
                                              INFERENCE["canvas"]["min_area"], INFERENCE["canvas"]["multiple"])
        generator = torch.Generator().manual_seed(CFG["seed"] * 1_000_003 + i)  # the VAE's posterior draws
        tic = time.time()
        enc = pipe.encode(sample["prompt"], [Image.open(p) for p in sample["images"]], generator, vlm_drop=vlm,
                          teacher_text=True)
        masks = enc["masks"]
        torch.save({"text": enc["teacher_text"].cpu(), "text_tags": enc["text_tags"],
                    "student_text": enc["text"].cpu() if vlm is not None else None, "text_keep": enc["text_keep"],
                    "latents": [z.cpu() for z in enc["latents"]], "masks": masks, "vlm_masks": enc["vlm_masks"],
                    "images": sample["images"], "height": height, "width": width},
                   out / f"{i:07d}_{height}x{width}x{len(masks)}.pt")
        vlm_rows = f", student text encoder rows {int(enc['text_keep'].sum())} of {len(enc['text_keep'])}" if vlm else ""
        print(f"{i}: {len(masks)} pictures, kept {[round(float(m.float().mean()), 3) for m in masks]}{vlm_rows}, "
              f"{time.time() - tic:.1f} s", flush=True)


def gaze_again(sample, pool):
    """The `Gaze` of every picture of an encoded sample, gazed again from its file on the CPU exactly as `encode` gazed
    it: the VAE copy, cut to whole 32 px tokens."""
    vae_images, _ = P.prepare_references(RESIZER, [Image.open(p) for p in sample["images"]])
    scale = vlm_scale()
    futures = []
    for image in vae_images:
        pixels = torch.from_numpy(np.array(image)[: image.height // 32 * 32, : image.width // 32 * 32])
        futures.append(submit_gaze(pool, pixels.permute(2, 0, 1), INFERENCE["references"]["gaze_kind"], MINIMAX_H3,
                                   vlm=scale if scale is not None else False))
    return [f.result() for f in futures]


def prepare(sample, live_gaze, pool):
    """CPU part of a training sample (DataLoader worker): the masks (gazed again with `live_gaze`, checked against the
    stored ones) and the dense / sparse packed layouts."""
    if live_gaze:
        check_gazes(sample, gaze_again(sample, pool), sample["images"])
    shape = (1, 24, 1, sample["height"] // 16, sample["width"] // 16)
    keep = torch.cat([m.flatten() for m in sample["masks"]])
    sample["shape"], sample["keep"] = shape, keep
    sample["dense"] = P.layout(sample["text_tags"], sample["latents"], *shape[-2:], "cpu")
    sample["sparse"] = P.layout(sample["text_tags"], sample["latents"], *shape[-2:], "cpu", keep,
                                sample.get("text_keep"))
    if sample.get("student_text") is None:
        sample["student_text"] = sample["text"]
    return sample


def codecopsd_loss(model, transformer, scheduler, sample, k, generator, device, timer):
    """MSE between the student's and the teacher's velocity (head k) at the student's state after k rollout steps.
    `sample` comes from `prepare` (on `device`)."""
    text, student_text = sample["text"], sample["student_text"]  # the teacher's (every vision token), the student's
    rows, reference_rows = P.pack(sample["latents"], sample["shape"], scheduler, generator)
    kept_rows = reference_rows[sample["keep"]]
    timesteps = P.set_timesteps(scheduler, device)
    with torch.no_grad():
        for i, t in enumerate(timesteps[:k]):  # the student's own rollout
            velocity = P.velocity(transformer, rows, kept_rows, student_text, sample["sparse"], t, i)
            rows = scheduler.step(velocity.float(), t, rows, return_dict=False)[0]
        timer.mark("rollout")
        target = P.velocity(transformer, rows, reference_rows, text, sample["dense"], timesteps[k], k, lora=False)
        timer.mark("teacher")
    prediction = P.velocity(model, rows, kept_rows, student_text, sample["sparse"], timesteps[k], k)  # with gradients
    return F.mse_loss(prediction.float(), target.float())


def main():
    args = parser(__doc__, "the released MiniMax-H3-Image folder").parse_args()
    use_config(CFG, args.config)
    if args.command == "encode":
        encode(args)
    else:
        train(args, CFG, MiniMaxH3ImageCodecOPSDTransformer2DModel, "transformer_diffusers", codecopsd_loss,
              {"model": "image", "vlm_picture_threshold_scale": str(float(CFG["codecopsd"]["vlm_picture_threshold_scale"]))},
              prepare)


if __name__ == "__main__":
    main()
