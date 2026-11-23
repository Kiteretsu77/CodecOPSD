"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Loading, shared by the command lines: what each model reads, its pipeline with the CodecOPSD LoRA, and where the
weights wait between uses (offload).
"""

import torch

# model: its release folder under models/ (`codecopsd.download.model_dir`; the LoRA: `lora_path(model)`), its command
# line (cli/<cli>.py) and its output file
MODELS = {
    "minimax_h3_ref2va": dict(release="MiniMax-H3-Ref2VA", cli="minimax_h3_ref2va", ext=".mp4"),
    "minimax_h3_image": dict(release="MiniMax-H3-Image", cli="minimax_h3_image", ext=".png"),
}
TEXT_ENCODERS = {"bf16": None, "fp8": "fp8", "fp8-storage": "bf16"}  # --text-encoder: the pipelines' text_encoder_fp8
RESIDENT_GB = 150  # --offload auto: every model stays on a GPU of this size or more (B200), else one at a time


def gpu_gb():
    """The memory of the first GPU in GB (0 without a GPU)."""
    return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0


def build_pipeline(name, folder, lora=None, lora_weight_name=None, text_encoder="fp8", sampler=None,
                   sampler_weights=None):
    """The pipeline of `name` (a key of MODELS) in CPU memory: the release in `folder`, the CodecOPSD LoRA `lora` when
    given (a .safetensors file, or a Hub repo id with `lora_weight_name`), AdaLN as per-timestep tables.
    `text_encoder`: a key of TEXT_ENCODERS (FP8 is converted from the bf16 release at load); `sampler` /
    `sampler_weights`: the Ref2VA sampler and its weights (None: the config's)."""
    fp8 = TEXT_ENCODERS[text_encoder]
    if name == "minimax_h3_image":
        from codecopsd import MiniMaxH3ImageCodecOPSDPipeline
        pipe = MiniMaxH3ImageCodecOPSDPipeline.from_minimax_h3_image(folder, text_encoder_fp8=fp8)
    else:
        from codecopsd import MiniMaxH3CodecOPSDPipeline
        kwargs = {"sampler": sampler} if sampler else {}  # default: the config's few-step sampler
        pipe = MiniMaxH3CodecOPSDPipeline.from_minimax_h3(folder, text_encoder_fp8=fp8, sampler_weights=sampler_weights,
                                                          **kwargs)
    if lora:
        pipe.load_condition_lora(lora, lora_weight_name)
    pipe.transformer.table_adaln()  # AdaLN as per-timestep tables, weights in CPU memory: 24 GB less GPU memory, exact
    return pipe


def place(pipe, offload, text_encoder_device=None):
    """Where the weights of `pipe` wait between uses. none: every model on the GPU (the text encoder on
    `text_encoder_device` when given); stream: the transformer block by block from CPU memory; model: one model on the
    GPU at a time (diffusers' model offload)."""
    if offload == "none":
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to(text_encoder_device or "cuda")
    elif offload == "stream":
        pipe.enable_block_streaming()
    else:
        pipe.enable_model_cpu_offload()
