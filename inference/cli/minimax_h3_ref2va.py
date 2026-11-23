"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Generate one video with MiniMax-H3 (reference to video) + a CodecOPSD student LoRA.

    python inference/generate.py minimax_h3_ref2va --prompt-file prompt.txt --reference person.png \\
        --reference motion.mp4 --out out.mp4

`generate.py` fills in --model and --lora from models/ (give them to use other files). References are plain RGB
pictures (.png / .jpg / .webp) and videos, given in the order the prompt numbers them (`<Picture 1>`, `<Picture 2>`,
... and `<Video 1>`, ...). Without `--lora` the teacher runs (MiniMax-H3 as released), with every reference token.
Sampling: `--sampler` (default from src/codecopsd/configs/minimax_h3_ref2va.yaml), a few-step version of the base or the
released 49-call grid (`--sampler base`, the paper's numbers).

`--mode` sets the defaults of the flags below (any flag given explicitly wins):
    baseline   the teacher (MiniMax-H3 as released): no LoRA, every reference token, the released 49 calls, bf16 text encoder
    compress   (default) CodecOPSD: drop 70 % of the video reference tokens, pictures by the codec threshold, VLM drop,
               FP8 text encoder, Pika H3 Flex 6 calls
    extreme    compress + drop 90 % of the video reference tokens, the text encoder reads exactly the transformer's tokens
"""

import argparse
import time

import torch
from diffusers.utils import export_to_video

from codecopsd.attention import BACKENDS, set_attention
from codecopsd.configs import load
from codecopsd.pipelines.minimax_h3_ref2va import load_reference, stage_times

from ..load import RESIDENT_GB, build_pipeline, gpu_gb, place
from ..utils import drop_line, memory_line, save_masks, track_drop

MODES = {
    "baseline": dict(sampler="base", text_encoder="bf16", video_drop="0.7", vlm_drop="auto"),
    "compress": dict(sampler="pika", text_encoder="fp8", video_drop="0.7", vlm_drop="auto"),
    "extreme": dict(sampler="pika", text_encoder="fp8", video_drop="0.9", vlm_drop="dit"),
}


def codec_overrides(args):
    """The per-call codec pruning settings of the flags (None: every setting from codec_gazer.yaml)."""
    out = {k: dict(zip(("motion_px", "residual"), v)) for k, v in
           (("video", args.video_thresholds), ("image", args.picture_thresholds)) if v}
    if args.video_drop_select:
        out.setdefault("video", {})["select"] = args.video_drop_select
    return out or None


def main(argv=None):
    """Run one request with the flags `argv` (default: the command line)."""
    p = argparse.ArgumentParser(prog="generate.py minimax_h3_ref2va", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="compress",
                   help="baseline (the teacher, 49 calls) / compress (default) / extreme; sets the defaults below")
    p.add_argument("--model", required=True, help="the released MiniMax-H3 folder (README: Inference)")
    p.add_argument("--lora", default=None, help="CodecOPSD student LoRA (.safetensors), or a Hub repo id + --lora-weight-name")
    p.add_argument("--lora-weight-name", default=None)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    p.add_argument("--reference", action="append", required=True, help="picture or video, in prompt order (repeatable)")
    p.add_argument("--frames", type=int, default=124, help="frames at 24 fps, rounded up to 17n + 5")
    p.add_argument("--height", type=int, default=480, help="multiple of 32 (default 480 x 832; 768 x 1344 = 768p)")
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--sampler", choices=["pika", "fasth3", "base"], default=load("minimax_h3_ref2va")["sampler"],
                   help="pika: Pika H3 Flex, 6 calls; fasth3: FastH3 Omni 8-step, 8 calls; base: the released grid, "
                        "49 calls")
    p.add_argument("--sampler-weights", default=None,
                   help="pika: ref2va/adapter.safetensors of Pika-Labs/Pika-H3-Flex; fasth3: the downloaded "
                        "FastVideo/FastVideo-FastH3-Omni-8-Step-V1 folder (default: next to --model)")
    p.add_argument("--steps", type=int, default=None, help="--sampler base only: sigma grid points (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense", action="store_true", help="give the student every reference token (no codec gazer)")
    p.add_argument("--video-drop", default="0.7",
                   help="video references: a share = at least that share of their tokens dropped (default 0.7): priority "
                        "drop, or the threshold decision when it drops more (codec_gazer.yaml priority_drop.select), "
                        "'elastic' = elastic drop, 'threshold' = the codec gazer's threshold decision alone")
    p.add_argument("--video-drop-select", choices=["max", "exact"], default=None,
                   help="with a --video-drop share: max (default, codec_gazer.yaml) = the threshold decision when it drops "
                        "more; exact = priority drop of exactly that share")
    p.add_argument("--video-thresholds", type=float, nargs=2, metavar=("MOTION_PX", "RESIDUAL"), default=None,
                   help="codec pruning thresholds of the reference videos (default: codec_gazer.yaml kinds.video, 5.0 "
                        "1.0; MOTION_PX at 480 x 640, scaled with the size): the threshold decision, used alone with "
                        "--video-drop threshold, else when it drops more than the share")
    p.add_argument("--picture-thresholds", type=float, nargs=2, metavar=("MOTION_PX", "RESIDUAL"), default=None,
                   help="codec pruning thresholds of the reference pictures (default: codec_gazer.yaml kinds.image, 4.0 1.5)")
    p.add_argument("--vlm-scale", type=float, default=None,
                   help="with --vlm-drop auto / on: the picture threshold scale of the text encoder mask (default 0.25)")
    p.add_argument("--vlm-drop", choices=["auto", "on", "off", "dit"], default="auto",
                   help="the student's text encoder reads only the gazed vision tokens (auto: as the LoRA was trained; dit: "
                        "exactly the transformer's tokens)")
    p.add_argument("--offload", choices=["auto", "model", "none", "stream"], default="auto",
                   help="auto (default): none on a GPU of 150 GB or more (B200), else model; model: one model on the GPU "
                        "at a time; none: every model stays on the GPU and the text encoder encodes at the same time as "
                        "the VAE; stream: the transformer block by block from CPU memory, only when model offload runs "
                        "out of memory (slower)")
    p.add_argument("--text-encoder-device", default=None,
                   help="with --offload none: keep the text encoder on this GPU (e.g. cuda:1) instead of the first one")
    p.add_argument("--text-encoder", choices=["bf16", "fp8", "fp8-storage"], default="fp8",
                   help="text encoder linears: fp8 (default): FP8 weights and FP8 products (bf16 products on GPUs without "
                        "FP8 tensor cores), 26.2 GB; fp8-storage: FP8 weights, bf16 products; bf16: as released, 50.3 GB")
    p.add_argument("--save-masks", default=None,
                   help="save the codec gazer's masks (bool, True = token kept, one array per reference) to this .npz")
    p.add_argument("--attention", choices=list(BACKENDS), default="auto",
                   help="the transformer's attention kernel; all exact")
    p.add_argument("--out", default="out.mp4")
    p.set_defaults(**MODES[p.parse_known_args(argv)[0].mode])
    args = p.parse_args(argv)
    if args.mode == "baseline":
        args.lora = None  # the teacher: every reference token

    references = [load_reference(path) for path in args.reference]
    pipe = build_pipeline("minimax_h3_ref2va", args.model, args.lora, args.lora_weight_name, args.text_encoder,
                          args.sampler, args.sampler_weights)
    if args.offload == "auto":  # every model resident when the GPU holds them all with room (B200: 180 GB)
        args.offload = "none" if gpu_gb() >= RESIDENT_GB else "model"
    place(pipe, args.offload, args.text_encoder_device)
    args.attention = set_attention(pipe, args.attention)
    counts = track_drop(pipe)
    start = time.perf_counter()
    try:
        video = pipe(
            args.prompt if args.prompt is not None else open(args.prompt_file).read(), references,
            height=args.height, width=args.width, num_frames=args.frames, num_inference_steps=args.steps,
            gaze=False if args.dense else None,
            video_drop={"threshold": None, "elastic": "elastic"}.get(args.video_drop, args.video_drop)
            if args.video_drop in ("threshold", "elastic") else float(args.video_drop),
            vlm_drop=(args.vlm_scale if args.vlm_scale and args.vlm_drop in ("auto", "on") else
                      {"auto": None, "on": True, "off": False, "dit": "dit"}[args.vlm_drop]),
            gaze_overrides=codec_overrides(args),
            generator=torch.Generator().manual_seed(args.seed),  # CPU noise: the same video on every GPU type
        ).frames[0]
    except torch.OutOfMemoryError:
        if args.offload == "stream":
            raise
        raise SystemExit(f"out of GPU memory with --offload {args.offload}: rerun with --offload stream (the transformer "
                         "then streams from CPU memory every step, slower at small sizes)") from None
    seconds = time.perf_counter() - start
    export_to_video(video, args.out, fps=24)
    print(f"saved {args.out}: {video.shape[0]} frames, height x width {video.shape[1]} x {video.shape[2]}")
    if args.save_masks and counts.get("masks"):
        save_masks(args.save_masks, counts["masks"])
    print(f"mode {args.mode}: sampler {args.sampler}, {drop_line(counts)}")
    print(stage_times(getattr(pipe, "timings", None)))  # the codec gazer runs on the CPU while the VAE encodes
    print(memory_line(seconds))


if __name__ == "__main__":
    main()
