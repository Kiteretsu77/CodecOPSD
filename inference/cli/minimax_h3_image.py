"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Generate one image with MiniMax-H3-Image (prompt + reference pictures) + a CodecOPSD student LoRA.

    python inference/generate.py minimax_h3_image --prompt-file prompt.txt --image person.png --image scene.png \\
        --out out.png

`generate.py` fills in --model and --lora from models/ (give them to use other files). Reference pictures are given in
the order the prompt numbers them (`<Picture 1>`, `<Picture 2>`, ...); without `--image` it is text to image. Without
`--lora` the teacher runs (MiniMax-H3-Image as released), with every reference token.

`--mode` sets the defaults of the flags below (any flag given explicitly wins; README: Modes):
    baseline   the teacher (MiniMax-H3-Image as released): no LoRA, every reference token, the released 6 steps, bf16 text encoder
    compress   (default) Codec-OPSD: the codec gazer drops reference tokens by its threshold, VLM drop, FP8 text encoder
"""

import argparse
import time

import torch
from PIL import Image

from codecopsd.attention import BACKENDS, set_attention
from codecopsd.pipelines.minimax_h3_image.pipeline_minimax_h3_image import reference_size
from codecopsd.pipelines.minimax_h3_ref2va import stage_times

from ..hardware import announce, cap_gpu_memory, vlm_files_gb
from ..load import RESIDENT_GB, build_pipeline, place, vlm_notice, vlm_source
from ..utils import drop_line, memory_line, save_masks, track_drop


# TODO: Be careful to check whether the teacher (`--mode baseline`: no LoRA, every reference token) fits in the 96 GB of
# an RTX PRO 6000, our main target GPU, for every sample we choose. If it runs out of memory there, record the teacher
# as OOM for that sample.
MODES = {
    "baseline": dict(text_encoder="bf16"),
    "compress": dict(text_encoder="fp8"),
}


def main(argv=None):
    """Run one request with the flags `argv` (default: the command line)."""
    p = argparse.ArgumentParser(prog="generate.py minimax_h3_image", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES), default="compress",
                   help="baseline (the teacher) / compress (default); sets the defaults below")
    p.add_argument("--model", required=True, help="the released MiniMax-H3-Image folder (README: MiniMax-H3-Image)")
    p.add_argument("--lora", default=None, help="CodecOPSD student LoRA (.safetensors), or a Hub repo id + --lora-weight-name")
    p.add_argument("--lora-weight-name", default=None)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    p.add_argument("--image", action="append", default=[], help="reference picture, in prompt order (repeatable)")
    p.add_argument("--height", type=int, default=None,
                   help="output height; default (no --height / --width): the first reference's aspect ratio on the "
                        "2048 x 2048 area, as the demo")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--reference-aspect", action="store_true",
                   help="output at the first reference's aspect ratio on the 2048 x 2048 area even when --height / --width are "
                        "given (the default without them; the paper's GEdit-Bench and OmniContext runs)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense", action="store_true", help="give the student every reference token (no codec gazer)")
    p.add_argument("--picture-thresholds", type=float, nargs=2, metavar=("MOTION_PX", "RESIDUAL"), default=None,
                   help="codec pruning thresholds of the pictures (default: src/codecopsd/configs/codec_gazer.yaml kinds.image, "
                        "4.0 1.5); a token is kept when its motion reaches MOTION_PX or its mean |residual| exceeds RESIDUAL")
    p.add_argument("--vlm-drop", choices=["auto", "on", "off", "dit"], default="auto",
                   help="the student's text encoder reads only the gazed vision tokens (auto / on: the picture thresholds x "
                        "--vlm-scale; dit: exactly the transformer's tokens; off: every vision token)")
    p.add_argument("--vlm-scale", type=float, default=None,
                   help="the picture threshold scale of the text encoder mask (default 0.25, codec_gazer.yaml vlm_drop)")
    p.add_argument("--offload", choices=["auto", "model", "none"], default="auto",
                   help="auto (default): none on a GPU of 150 GB or more (B200), else model; model: one model on the GPU "
                        "at a time; none: every model stays on the GPU")
    p.add_argument("--text-encoder", choices=["bf16", "fp8", "fp8-storage"], default="fp8",
                   help="text encoder linears: fp8 (default): FP8 weights and FP8 products (bf16 products on GPUs without "
                        "FP8 tensor cores), 26.2 GB; fp8-storage: FP8 weights, bf16 products; bf16: as released, 50.3 GB")
    p.add_argument("--vlm-fp8", default=None,
                   help="with an FP8 --text-encoder: the FP8 VLM folder to read (default: the downloaded copy in "
                        "models/, else one saved by python -m inference.vlm_fp8); none: convert the bf16 release at "
                        "load (needs its 66 GB in CPU memory)")
    p.add_argument("--save-masks", default=None,
                   help="save the codec gazer's masks (bool, True = token kept, one array per reference) to this .npz")
    p.add_argument("--attention", choices=list(BACKENDS), default="auto",
                   help="the transformer's attention kernel (README: Acceleration); all exact")
    p.add_argument("--out", default="out.png")
    p.set_defaults(**MODES[p.parse_known_args(argv)[0].mode])
    args = p.parse_args(argv)

    if args.mode == "baseline":
        args.lora = None  # the teacher: every reference token

    vlm = vlm_source(args.text_encoder, args.vlm_fp8)  # the FP8 VLM already converted when there is one (load.py)
    # the hardware report and the warnings that apply (bf16 VLM files: mapped only when read)
    announce(commit_gb=vlm_files_gb(args.model) if vlm["kind"] in ("converted", "bf16") else None)
    pipe = build_pipeline("minimax_h3_image", args.model, args.lora, args.lora_weight_name, args.text_encoder,
                          text_encoder_path=vlm["path"])
    if args.offload == "auto":  # every model resident when the GPU holds them all with room (B200: 180 GB), as the demo
        args.offload = "none" if cap_gpu_memory() >= RESIDENT_GB else "model"
    vlm_notice(vlm, args.offload)
    place(pipe, args.offload)
    images = [Image.open(f) for f in args.image]
    aspect = args.reference_aspect or (args.height is None and args.width is None)  # the demo's default
    height, width = reference_size(images[0]) if aspect and images else (args.height, args.width)
    args.attention = set_attention(pipe, args.attention)
    counts = track_drop(pipe)
    start = time.perf_counter()
    image = pipe(
        args.prompt if args.prompt is not None else open(args.prompt_file).read(), images,
        height=height, width=width, gaze=False if args.dense else None,
        vlm_drop={"auto": None, "on": True, "off": False, "dit": True}[args.vlm_drop],
        vlm_picture_scale="dit" if args.vlm_drop == "dit" else args.vlm_scale,
        gaze_overrides={"image": dict(zip(("motion_px", "residual"), args.picture_thresholds))}
        if args.picture_thresholds else None,
        generator=torch.Generator().manual_seed(args.seed),  # CPU noise: the same image on every GPU type
    ).images[0]
    seconds = time.perf_counter() - start
    image.save(args.out)
    print(f"saved {args.out}: height x width {image.height} x {image.width}")
    if args.save_masks and counts.get("masks"):
        save_masks(args.save_masks, counts["masks"])
    print(f"mode {args.mode}: {drop_line(counts)}")
    print(stage_times(getattr(pipe, "timings", None)))  # the codec gazer runs on the CPU while the VAE encodes
    print(memory_line(pipe, seconds))


if __name__ == "__main__":
    main()
