"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Loading, shared by the command line and the demos: what each model reads, its pipeline with the CodecOPSD LoRA, and
where the weights wait between uses (offload). The GPU memory cap is in hardware.py, the memory plans of HyperFlow H3
Flash in extreme.py.
"""

# model: its release folder under models/ (`codecopsd.download.model_dir`; the LoRA: `lora_path(model)`), its command
# line (cli/<cli>.py) with fixed flags, the bundled example it runs by default, its output file and its label on the demo
MODELS = {
    "minimax_h3_ref2va": dict(release="MiniMax-H3-Ref2VA", cli="minimax_h3_ref2va", example="dog_frisbee", ext=".mp4",
                              label="video (MiniMax H3)"),
    # HyperFlow H3 Flash in place of the released Ref2VA transformer (12 to 32 GB GPUs): the same LoRA, the video examples
    "h3_flash_ref2va": dict(release="MiniMax-H3-Ref2VA", cli="minimax_h3_ref2va",
                            args=["--sampler", "flash", "--steps", "4"],
                            example="anime_karate", ext=".mp4", label="video (HyperFlow H3 Flash)"),
    "minimax_h3_image": dict(release="MiniMax-H3-Image", cli="minimax_h3_image", example="lineart_to_photo", ext=".png",
                             label="image (MiniMax H3 Image)"),
}
TEXT_ENCODERS = {"bf16": None, "fp8": "fp8", "fp8-storage": "bf16"}  # --text-encoder: the pipelines' text_encoder_fp8
RESIDENT_GB = 150  # --offload auto: every model stays on a GPU of this size or more (B200), else one at a time


def vlm_source(text_encoder="fp8", vlm_fp8=None):
    """Where the text encoder (VLM) of a run comes from: dict(kind, path). `text_encoder` "bf16" (a key of
    TEXT_ENCODERS): kind "bf16", the release as it is. Else the FP8 VLM already converted, so that the 66 GB bf16
    release is never read into CPU memory (`codecopsd.download`, VLM_FP8): `vlm_fp8` (a folder, or "none" for the
    release converted at load), else the downloaded ModelOpt copy ("downloaded"), a saved one ("saved"), else the bf16
    release converted to FP8 at load ("converted"). Only what is on disk: the entry points download first
    (`codecopsd.download.ensure`)."""
    from codecopsd.download import vlm_fp8_source
    from codecopsd.models.fp8_linear import fp8_folder_format

    if TEXT_ENCODERS[text_encoder] is None:
        return dict(kind="bf16", path=None)
    if vlm_fp8 and vlm_fp8 != "none":
        kind = fp8_folder_format(vlm_fp8)
        if kind is None:
            raise SystemExit(f"--vlm-fp8 {vlm_fp8}: no FP8 text encoder there (a ModelOpt FP8 copy or a saved one)")
        return dict(kind={"modelopt": "downloaded"}.get(kind, kind), path=str(vlm_fp8))
    if vlm_fp8 == "none":
        return dict(kind="converted", path=None, why="--vlm-fp8 none")
    kind, path = vlm_fp8_source()
    if kind is None:
        return dict(kind="converted", path=None, why="no FP8 VLM downloaded or saved")
    return dict(kind=kind, path=str(path))


def vlm_notice(source, offload, pin_vlm=True, prefix="", pin_rule=""):
    """Print one line: which VLM a run reads (`vlm_source`) and where it waits between requests (`offload`; with
    "pinned", page-locked or read from the page cache, `pin_vlm`, for the reason `pin_rule`)."""
    from codecopsd.download import REPOS, VLM_FP8

    what = {"downloaded": f"FP8, downloaded pre-quantized ({REPOS[VLM_FP8][0]}, {source['path']})",
            "saved": f"FP8, saved once ({source['path']})",
            "converted": f"the bf16 release converted to FP8 at load ({source.get('why')})",
            "bf16": "the bf16 release (--text-encoder bf16)"}[source["kind"]]
    if offload == "pinned" and source["kind"] not in ("downloaded", "saved"):
        where = "page-locked"  # converted at load: a copy in process memory, never in the page cache
    elif offload == "pinned":
        where = "page-locked in place" if pin_vlm else (
            "read from the page cache at each request" + (f" ({pin_rule})" if pin_rule else ""))
    else:
        where = {"none": "kept on the GPU", "stream": "moved to the GPU for each request",
                 "model": "moved to the GPU for each request"}.get(offload, f"offload {offload}")
    print(f"{prefix}text encoder (VLM): {what}; {where}", flush=True)


def build_pipeline(name, folder, lora=None, lora_weight_name=None, text_encoder="fp8", sampler=None,
                   sampler_weights=None, shared_text_encoder=None, text_encoder_path=None):
    """The pipeline of `name` (a key of MODELS) in CPU memory: the release in `folder`, the CodecOPSD LoRA `lora` when
    given (a .safetensors file, or a Hub repo id with `lora_weight_name`), AdaLN as per-timestep tables.
    `text_encoder`: a key of TEXT_ENCODERS; `sampler` / `sampler_weights`: the Ref2VA sampler and its weights (None:
    the config's; h3_flash_ref2va always runs "flash"); `shared_text_encoder`: a text encoder already loaded (MiniMax-H3
    and MiniMax-H3-Image ship the same one); `text_encoder_path`: an FP8 VLM folder (`vlm_source`), read instead of
    converting the bf16 release."""
    fp8 = TEXT_ENCODERS[text_encoder]
    if name == "minimax_h3_image":
        from codecopsd import MiniMaxH3ImageCodecOPSDPipeline
        pipe = MiniMaxH3ImageCodecOPSDPipeline.from_minimax_h3_image(folder, text_encoder_fp8=fp8,
                                                                     text_encoder=shared_text_encoder,
                                                                     text_encoder_path=text_encoder_path)
    else:
        from codecopsd import MiniMaxH3CodecOPSDPipeline
        sampler = "flash" if name == "h3_flash_ref2va" else sampler
        kwargs = {"sampler": sampler} if sampler else {}  # default: the config's few-step sampler
        pipe = MiniMaxH3CodecOPSDPipeline.from_minimax_h3(folder, text_encoder_fp8=fp8, text_encoder=shared_text_encoder,
                                                          sampler_weights=sampler_weights,
                                                          text_encoder_path=text_encoder_path, **kwargs)
    if lora:
        pipe.load_condition_lora(lora, lora_weight_name)
    if hasattr(pipe.transformer, "table_adaln"):  # not HyperFlow H3 Flash, whose AdaLN is 16-dimensional
        pipe.transformer.table_adaln()  # AdaLN as per-timestep tables, weights in CPU memory: 24 GB less GPU memory, exact
    return pipe


def place(pipe, offload, stream_blocks=0, text_encoder_device=None, pin_text_encoder=True):
    """Where the weights of `pipe` wait between uses. none: every model on the GPU (the text encoder on
    `text_encoder_device` when given); pinned: one model on the GPU at a time, from page-locked CPU memory, the text
    encoder streamed layer by layer (from the page cache instead with `pin_text_encoder=False`, when it was read as
    file views) and `stream_blocks` of the transformer's blocks streamed too ("all" or a number);
    stream: the transformer block by block from CPU memory; model: one model on the GPU at a time (diffusers' model
    offload)."""
    if offload == "none":
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.text_encoder.to(text_encoder_device or "cuda")
    elif offload == "pinned":
        pipe.enable_pinned_offload(stream_blocks=stream_blocks, pin_text_encoder=pin_text_encoder)
    elif offload == "stream":
        pipe.enable_block_streaming()
    else:
        pipe.enable_model_cpu_offload()
