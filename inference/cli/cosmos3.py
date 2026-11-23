"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Run one Cosmos 3 (Nano) task with a CodecOPSD student LoRA.

    # control -> video (one to four of edge / blur / depth / seg)
    python inference/generate.py cosmos3 --model Cosmos3-Nano --lora weights/codecopsd_cosmos3.safetensors transfer \\
        --prompt-file prompt.txt --control depth=depth.mp4 --control seg=seg.mp4 --out out.mp4
    # image -> video, continuation, video -> audio, robot inverse dynamics (DROID three-view canvas -> 16 actions)
    python inference/generate.py cosmos3 ... i2v --prompt-file prompt.txt --image first_frame.png --out out.mp4
    python inference/generate.py cosmos3 ... continuation --prompt-file prompt.txt --video given.mp4 --given-latent-frames 19 \\
        --frames 193 --out out.mp4
    python inference/generate.py cosmos3 ... v2a --prompt-file prompt.txt --video clip.mp4 --out out.mp4
    python inference/generate.py cosmos3 ... id_droid --prompt "Put the cup in the sink" --video canvas.mp4 --out actions.json

Controls, images and videos are given at any size: they are resized to cover the output size and center-cropped (seg:
nearest neighbour). Without --lora the released model runs, with every condition token. id_droid runs the exact fast
path (bit-identical actions); --no-fast runs the released forward.
"""

import argparse
import json

import imageio.v3 as iio
import numpy as np
import torch

from codecopsd import Cosmos3CodecOPSDPipeline

from ..hardware import announce


def to_uint8(video):
    """[1, C, T, H, W] / [T, C, H, W] / [T, H, W, C] in [0, 1] -> uint8 [T, H, W, 3]."""
    v = video[0] if video.ndim == 5 else video
    v = (v.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
    if v.shape[0] == 3:
        v = v.permute(1, 2, 3, 0)
    elif v.shape[1] == 3:
        v = v.permute(0, 2, 3, 1)
    return v.numpy()


def main(argv=None):
    """Run one task with the flags `argv` (default: the command line)."""
    p = argparse.ArgumentParser(prog="generate.py cosmos3", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the released Cosmos3-Nano folder (README: Downloads)")
    p.add_argument("--lora", default=None, help="CodecOPSD student LoRA (.safetensors), or a Hub repo id + --lora-weight-name")
    p.add_argument("--lora-weight-name", default=None)
    p.add_argument("--lora-dtype", choices=["bfloat16", "float32"], default="bfloat16",
                   help="LoRA factors + computation: bfloat16 (default, the base model's precision) or float32 (as "
                        "trained, slower)")
    p.add_argument("task", choices=["transfer", "i2v", "continuation", "v2a", "id_droid"])
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="caption (transfer: a caption or a JSON prompt; id_droid: the instruction)")
    prompt.add_argument("--prompt-file")
    negative = p.add_mutually_exclusive_group()
    negative.add_argument("--negative-prompt", help="default: the framework's default negative prompt")
    negative.add_argument("--negative-prompt-file", help="e.g. a Physics-IQ case's own negative prompt")
    p.add_argument("--control", action="append", default=[], help="transfer: <edge|blur|depth|seg>=<video> (repeatable)")
    p.add_argument("--image", help="i2v: the first frame")
    p.add_argument("--video", help="continuation: the given frames; v2a: the input video; id_droid: the canvas")
    p.add_argument("--given-latent-frames", type=int, default=None, help="continuation: 4 (g - 1) + 1 frames given")
    p.add_argument("--frames", type=int, default=None, help="output frames, 4n + 1 (default: configs/cosmos3.yaml)")
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--fps", type=float, default=None, help="default: the input's")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dense", action="store_true", help="give the student every condition token (no codec gazer)")
    p.add_argument("--gaze-all-given", action="store_true",
                   help="continuation: gaze the last given latent frame too (default: given whole)")
    p.add_argument("--no-fast", action="store_true",
                   help="id_droid: the released forward instead of the exact fast path (same actions, slower)")
    p.add_argument("--out", default="out.mp4")
    args = p.parse_args(argv)

    announce()  # the hardware report and the warnings that apply
    pipe = Cosmos3CodecOPSDPipeline.from_cosmos3(args.model)
    if args.lora:
        pipe.load_condition_lora(args.lora, args.lora_weight_name, dtype=args.lora_dtype)
    text = args.prompt if args.prompt is not None else open(args.prompt_file).read().strip()
    negative = args.negative_prompt if args.negative_prompt_file is None else open(args.negative_prompt_file).read().strip()
    controls = dict(c.split("=", 1) for c in args.control) or None
    out = pipe(args.task, text, controls=controls, image=args.image, video=args.video, num_frames=args.frames,
               height=args.height, width=args.width, fps=args.fps, given_latent_frames=args.given_latent_frames,
               seed=args.seed, negative_prompt=negative, gaze=False if args.dense else None,
               keep_last_given=False if args.gaze_all_given else None, fast=False if args.no_fast else None)
    tokens = f"condition tokens {out['kept_tokens']} of {out['condition_tokens']}, {out['seconds']:.0f} s"
    if args.task == "id_droid":
        actions = out["action"]
        actions = actions[0] if isinstance(actions, (list, tuple)) else actions
        json.dump(np.asarray(actions.float().cpu()).tolist(), open(args.out, "w"))
        print(f"saved {args.out}: actions {tuple(actions.shape)}, {tokens}")
        return
    video = to_uint8(out["videos"])
    fps = out["request"]["fps"]
    if args.task == "v2a":
        import av  # noqa: F401  (PyAV, for the audio stream)
        from diffusers.utils import export_to_video
        sound = out["sound"]
        sound = sound[0] if isinstance(sound, (list, tuple)) else sound
        export_to_video(list(video), args.out, fps=fps, audio=np.asarray(sound.float().cpu()),
                        audio_sample_rate=int(out["sampling_rate"]))
    else:
        iio.imwrite(args.out, video, fps=fps, plugin="FFMPEG", codec="libx264", output_params=["-crf", "12"])
    print(f"saved {args.out}: {video.shape[0]} frames, height x width {video.shape[1]} x {video.shape[2]}, {tokens}")


if __name__ == "__main__":
    main()
