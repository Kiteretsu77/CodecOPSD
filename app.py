"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The CodecOPSD demos, one page each (run from the repository root):

    pip install -e ".[app]"
    python app.py                     # no page given: ref2va_extreme (also with its flags: --gpu-memory 24)
    python app.py ref2va              # MiniMax H3 Ref2VA (reference to video), 80 GB GPUs and more
    python app.py ref2va_extreme      # the same task on a 12, 16, 24 or 32 GB GPU (HyperFlow H3 Flash)
    python app.py ref2image           # MiniMax H3 reference to image
    python app.py ref2va --help       # the options of a page

Each page shows CodecOPSD (Ours) next to the teacher, the reference tokens that codec pruning drops, and where the
seconds of every request went, stage by stage (`codecopsd.stats`; also appended to outputs/app/stats.jsonl). The
pages themselves are in inference/demo/ (inference/demo/__init__.py lists its parts).
"""

import sys

REF2VA = """Demo: MiniMax H3 Ref2VA, reference pictures and videos to video. CodecOPSD (Ours) next to the teacher (every
reference token), both with Pika H3 Flex (6 steps). For GPUs of 80 GB and more (RTX PRO 6000, H100, H200, B200); on a
12-32 GB GPU run the ref2va_extreme page.

    python app.py ref2va                         # one model on the GPU at a time (--offload model)
    python app.py ref2va --offload none          # 141 GB+ GPU: every model resident, nothing moves on a click

The first launch downloads what is missing (`codecopsd.download`): the FP8 text encoder (VLM, 31.7 GB), MiniMax-H3
without its bf16 text encoder (77 GB) and Pika H3 Flex (2.8 GB) into models/, and the example media into
assets/examples/refs/ (`--no-download` turns this off). Every acceleration of the repository is on: codec pruning,
the CodecOPSD LoRA, text encoder (VLM) pruning, Pika H3 Flex 6 steps, the FP8 text encoder, the bf16 video decoder,
AdaLN tables and the attention kernel of the GPU (README: Acceleration). Under the results, "Where the time went"
lists every stage of the run; the panel at the bottom sums them over all requests.
"""

REF2VA_EXTREME = """Demo: MiniMax H3 Ref2VA on a 12, 16, 24 or 32 GB GPU. Reference pictures and videos to a video with
HyperFlow H3 Flash (Video Rebirth's compressed MiniMax H3 transformer: 13.9B parameters, INT8 and low rank, 16 GB, its
own sampler, 4 steps by default, --steps 6 or 8) and the CodecOPSD LoRA, next to the teacher (Flash with every
reference token).

    python app.py                                    # this page (app.py without a page name), first GPU's memory
    python app.py ref2va_extreme                     # the same
    python app.py ref2va_extreme --gpu-memory 24     # plans for a 24 GB card (12, 16, 24 or 32)

`--gpu-memory` (default: the GPU's own) picks the plan (README: Consumer GPUs), and on a larger GPU also caps the
process to it, CUDA context included, so the page shows what that card does. 32 GB: the models wait in page-locked CPU
memory (48 GB: 64 GB or more in the machine) and one at a time is copied in; the text encoder's 51 layers are streamed
in one by one while it runs. 24 GB: 16 of Flash's 49 blocks are streamed too at every step, 16 and 12 GB: all of them.
Every plan gives the same video.

The first launch downloads what is missing (`codecopsd.download`): the FP8 text encoder (VLM, 31.7 GB), the VAE and
processor of MiniMax-H3 (without its transformer and bf16 text encoder) and HyperFlow H3 Flash
(`VideoRebirthAI/hyperflow-h3-flash`, folder ref2va_int8_convrot/, 16.3 GB) into models/; the example media into
assets/examples/refs/. Every other acceleration of
the ref2va page is on (codec pruning, the CodecOPSD LoRA, VLM pruning, the FP8 text encoder, the bf16 video decoder,
the attention kernel). Under the results, "Where the time went" lists every stage of the run; the panel at the bottom
sums them over all requests.

References are read as on the ref2va page: a picture at its own short edge, kept between 1024 and 2048 px, a video at
the output's tier (480 for a 480p output, 768 for 768p). The bundled examples
use everyday files: pictures with a short edge of 1024 to 1536 px, videos at 480p. The GPU memory plans are measured at
480p; a larger output needs more memory.

The examples are ten cases of the video page with smaller files (five of them anime) and cases published by others
(ComfyUI workflow templates, the Pika H3 Flex showcase, OmniVBench, Comfy Hub;
assets/examples/SOURCES.md), all run at 480p, every prompt in the structured form of the bundled examples.
"""

REF2IMAGE = """Demo: MiniMax H3 Image, reference pictures to a picture. CodecOPSD (Ours) next to the teacher (every
reference token), 6 steps. For GPUs of 80 GB and more.

    python app.py ref2image                         # one model on the GPU at a time (--offload model)
    python app.py ref2image --offload none          # every model resident

The first launch downloads what is missing (`codecopsd.download`): MiniMax-H3-Image (64 GB) into models/ and the example
media into assets/examples/refs/ (`--no-download` turns this off). Under the results, "Where the time went" lists the
denoising and the rest of the run; the panel at the bottom sums them over all requests.
"""

DEMOS = {  # page: dict(model, port, title (browser tab), heading, sizes / size / frames (video pages), examples, doc)
    "ref2va": dict(
        model="minimax_h3_ref2va", port=7861,
        title="CodecOPSD: H3 Ref2VA",
        heading="MiniMax H3 Ref2VA: reference pictures and videos to video, with Pika H3 Flex (6 steps).",
        sizes=["320 x 576", "480 x 832", "576 x 1024", "768 x 1344"],  # about 16:9, multiples of 32 (320p to 768p)
        frames=[56, 90, 124, 158, 243],  # 17n + 5 frames at 24 fps: 2.3 s, 3.8 s, 5.2 s, 6.6 s, 10.1 s
        doc=REF2VA,
    ),
    "ref2va_extreme": dict(
        model="h3_flash_ref2va", port=7862,
        title="CodecOPSD: H3 Ref2VA on a 12-32 GB GPU",
        heading="MiniMax H3 Ref2VA on a 12, 16, 24 or 32 GB GPU: HyperFlow H3 Flash (4 steps) + CodecOPSD.",
        # size: the default (the GPU memory plans are measured at 480p), also of an example whose own size is not offered
        sizes=["480 x 832", "832 x 480", "416 x 736", "736 x 416", "576 x 1024", "768 x 1344"], size=(480, 832),
        frames=[56, 90, 124],
        examples=["flash_lipstick_recast", "flash_record_shop", "flash_anime_karate", "flash_superboy_kaiju",
                  "flash_clash",
                  "flash_cockatiel_dance", "flash_anime_skater", "flash_dolly_zoom_desert", "flash_anime_rooftop_orbit",
                  "flash_horse_pounce", "flash_pilot_station", "flash_sky_island", "flash_anime_rooftop_crane",
                  "flash_yellow_plane", "flash_mystic_robot", "flash_crash_zoom", "flash_orbit_hilltop_v2",
                  "flash_dog_frisbee"],
        doc=REF2VA_EXTREME,
    ),
    "ref2image": dict(
        model="minimax_h3_image", port=7860,
        title="CodecOPSD: H3 reference to image",
        heading="MiniMax H3 Image: reference pictures to a picture, 6 steps.",
        doc=REF2IMAGE,
    ),
}

DEFAULT = "ref2va_extreme"  # the page of `python app.py` without a page name: it runs on the most GPUs (12 GB and up)

if __name__ == "__main__":
    first = sys.argv[1] if len(sys.argv) > 1 else None
    if first in ("-h", "--help") or (first is not None and not first.startswith("-") and first not in DEMOS):
        raise SystemExit(__doc__ + f"\npages: {', '.join(DEMOS)} (default: {DEFAULT})")
    page = DEMOS[sys.argv.pop(1) if first in DEMOS else DEFAULT]
    from inference.demo import main
    main(page)
