"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Prompt + reference pictures / videos -> video, with MiniMax-H3 (Ref2VA) and a CodecOPSD student condition LoRA."""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import torch
from accelerate import cpu_offload_with_hook
from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.before_encoder import MiniMaxH3Ref2VASetupStep
from diffusers.modular_pipelines.minimax_h3.encoders import (
    MiniMaxH3Ref2VATextEncoderStep,
    encode_vae_condition,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_FPS,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin,
)

from ...codec_gazer import MINIMAX_H3, submit_gaze
from ...configs import load
from ...download import count_lora_load
from ...models import H3FlashCodecOPSDTransformer3DModel, MiniMaxH3CodecOPSDTransformer3DModel, text_encoder_to_fp8
from ...models.fp8_linear import load_fp8_text_encoder
from ...offload import offload_all, pinned_model_offload
from ...stats import Clock
from .pipeline_output import MiniMaxH3CodecOPSDPipelineOutput
from .samplers import SAMPLERS, audio_schedule, few_step, load_fasth3_transformer, load_pika_adapter
from .vlm_drop import drop_vision_rows, vision_token_keep

CONFIG = load("minimax_h3_ref2va")  # src/codecopsd/configs/minimax_h3_ref2va.yaml
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # the VAE's pixel normalization (ImageNet)
TEXT_LAYER = CONFIG["text_encoder_layer"]
REFERENCE_T = CONFIG["references"]["noise_t"]
CANVAS = dict(canvas_multiple=CONFIG["canvas"]["multiple"], short_edge=CONFIG["canvas"]["short_edge"],
              max_pixels=CONFIG["canvas"]["max_pixels"])
PICTURE_SHORT_EDGE = CONFIG["references"]["picture_short_edge"]
VIDEO_TIERS = sorted(tuple(t) for t in CONFIG["references"]["video_tiers"])  # (short edge, area cap), Pika H3 Flex's


def video_tier(height, width):
    """(short edge, area cap) of the canvas a reference video is read on for an output of `height` x `width`: the
    smallest of `VIDEO_TIERS` whose short edge reaches the output's, the largest when none does (Pika H3 Flex's rule)."""
    return next((t for t in VIDEO_TIERS if t[0] >= min(height, width)), VIDEO_TIERS[-1])

PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resident(module):
    """On a GPU and not moved by an offload hook (`enable_model_cpu_offload`, `enable_block_streaming`)."""
    return not hasattr(module, "_hf_hook") and next(module.parameters()).device.type == "cuda"


def vlm_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, text_encoder_layer, device, dtype):
    """The text encoder's `hidden_states[text_encoder_layer]`, as diffusers' MiniMax-H3 text encoder step computes it,
    without keeping the hidden state of every layer: the input of decoder layer `text_encoder_layer` is taken by a hook
    (the same tensor). The released model's prompts (every vision token of the references, 30K+ rows) otherwise hold
    50 x 10 KB per row and ran out of memory on 80 GB GPUs.

    A text encoder that ends one layer earlier (the downloaded FP8 VLM has decoder layers 0-49 only,
    `codecopsd.models.fp8_linear`) gives the same tensor as the output of its last layer, taken by a forward hook:
    in the VLM's text model (transformers) the input of layer i + 1 is exactly the output of layer i (pre-norm; the final
    norm runs after the loop and makes a new tensor), except after the first layers, where the deepstack visual
    features are added (one layer per `deepstack_visual_indexes` entry: layers 0-2), which a last layer 49 is not."""
    layers = text_encoder.model.language_model.layers
    captured = {}
    if len(layers) > text_encoder_layer:
        def take(module, args, kwargs):
            captured["hidden"] = args[0] if args else kwargs["hidden_states"]

        handle = layers[text_encoder_layer].register_forward_pre_hook(take, with_kwargs=True)
    else:
        deepstack = len(getattr(text_encoder.config.vision_config, "deepstack_visual_indexes", ()) or ())
        if len(layers) != text_encoder_layer or text_encoder_layer - 1 < deepstack:
            raise ValueError(f"the text encoder has {len(layers)} decoder layers: hidden_states[{text_encoder_layer}] "
                             f"needs {text_encoder_layer} or more, the last one after the {deepstack} deepstack layers")

        def take(module, args, output):
            captured["hidden"] = output[0] if isinstance(output, tuple) else output

        handle = layers[text_encoder_layer - 1].register_forward_hook(take)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    vision = {name: value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)
              for name, value in (vision_inputs or {}).items()}
    hook = getattr(text_encoder, "_hf_hook", None)  # model offload: the hook wraps the top-level forward only
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(text_encoder)
    try:
        text_encoder.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           mm_token_type_ids=mm_token_type_ids, use_cache=False, output_hidden_states=False, **vision)
    finally:
        handle.remove()
    return captured["hidden"].to(device=device, dtype=dtype)


def load_reference(path, num_frames=None):
    """A picture (.png / .jpg / .webp / .bmp, upright by its EXIF orientation) or video file as a diffusers MiniMax-H3
    reference. With `num_frames` (the frames to generate) a video is decoded only as far as the pipeline reads it (its
    first `num_frames` at 24 fps), so a long video is not read whole; the frames the pipeline uses are the same."""
    if str(path).lower().endswith(PICTURES):
        return MiniMaxH3ImageReference(image=ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    fps = float(iio.immeta(path, plugin="FFMPEG")["fps"])
    limit = None if num_frames is None else math.ceil(align_num_frames(num_frames, 17, 5) * fps / MINIMAX_H3_FPS) + 2
    frames = []
    for frame in iio.imiter(path, plugin="FFMPEG"):
        if limit is not None and len(frames) >= limit:
            break
        frames.append(frame)
    return MiniMaxH3VideoReference(frames=np.stack(frames), fps=fps)


def stage_times(timings):
    """One line on where the reference encoding of a request went (`pipe.timings` after a call): preparing the
    references, the VAE, the text encoder, and the codec gazer, which runs on the CPU while the VAE encodes. Every part is a
    duration; when the stages run one after the other the printed total is the sum of the printed parts."""
    t = timings or {}
    if not t:
        return ""
    r = lambda key: round(t.get(key) or 0.0, 1)
    vae = f"VAE {r('vae'):.1f} s" if t.get("vae") else "no reference to encode"
    gazer = (f"; the codec gazer ran {r('gaze'):.1f} s on the CPU during the VAE (waited {r('gaze_wait'):.1f} s for it)"
             if t.get("gaze") else "")
    back = r("offload")
    if t.get("text_parallel"):
        return (f"references {t['encode']:.1f} s: preparing {r('prepare'):.1f} s, then {vae} and the text encoder "
                f"{r('text'):.1f} s at the same time{gazer}")
    parts = [("preparing", r("prepare")), (vae, None), ("waiting for the codec gazer", r("gaze_wait") if t.get("gaze") else None),
             ("text encoder", r("text")), ("text encoder back to the CPU", back or None)]
    total = r("prepare") + r("vae") + (r("gaze_wait") if t.get("gaze") else 0.0) + r("text") + back
    shown = " + ".join(f"{name} {v:.1f} s" if v is not None else name for name, v in parts if v is not None or name == vae)
    return f"references {total:.1f} s = {shown}{gazer}"


class MiniMaxH3CodecOPSDPipeline(DiffusionPipeline):
    r"""MiniMax-H3 reference-to-video (pictures and videos as references) with a CodecOPSD student condition LoRA.

    The CodecOPSD student is a LoRA on the reference rows of the released Ref2VA transformer, distilled on-policy from the
    released model on only the reference tokens the codec gazer keeps. Without a LoRA (or with `gaze=False` and the
    LoRA off) the pipeline is the released model with silent audio rows (the teacher). Sampling: a step-distilled
    version of the released base by default (`samplers.py`: Pika H3 Flex, 6 calls; or FastH3 Omni 8-step), on which the
    LoRA runs unchanged; `sampler="base"` is the released 49-call grid.
    """

    model_cpu_offload_seq = "vae->text_encoder->transformer"
    _optional_components = ["transformer"]

    def __init__(
        self,
        scheduler: MiniMaxH3Scheduler,
        vae: AutoencoderKLMiniMaxH3,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        transformer: MiniMaxH3CodecOPSDTransformer3DModel | None = None,
    ):
        super().__init__()
        layers = text_encoder.model.language_model.layers
        if len(layers) > TEXT_LAYER + 1:  # hidden_states[50] is the input of layer 50: the later layers are never read
            del layers[TEXT_LAYER + 1 :]
            text_encoder.config.text_config.num_hidden_layers = TEXT_LAYER + 1
        # the downloaded FP8 VLM has 50 layers: hidden_states[50] is then the output of layer 49 (`vlm_prompt_embeds`)
        self.register_modules(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, processor=processor,
            transformer=transformer,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
        self.text_step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=CONFIG["vlm_video_fps"])

    @classmethod
    def from_minimax_h3(cls, path, torch_dtype=torch.bfloat16, load_transformer=True, text_encoder_fp8=None,
                        sampler=CONFIG["sampler"], sampler_weights=None, text_encoder=None, text_encoder_path=None):
        """The pipeline from the released MiniMax-H3 folder (`hf download MiniMaxAI/MiniMax-H3`): the Ref2VA
        transformer (`transformer_ref`; not with `load_transformer=False`), the text encoder, the video VAE (diffusers' bf16
        loading: the decoder blocks in bf16, the encoder and the norms in float32; decoding about 7x faster than
        float32 on a B200) and the scheduler.

        `sampler` (`samplers.py`): "pika" (Pika H3 Flex, 6 calls), "fasth3" (FastH3 Omni 8-step, 8 calls), "flash"
        (HyperFlow H3 Flash, its own compressed transformer, 6 calls) or "base" (the released grid, 49 calls). The
        few-step ones read `sampler_weights`: Pika's `ref2va/adapter.safetensors`, the FastH3 repo folder or Flash's
        `ref2va_int8_convrot/` folder (default: `samplers.<sampler>.weights` of the config, next to `path`; README:
        Downloads). With "flash" the released Ref2VA transformer is not loaded.

        `text_encoder`: a text encoder that is already loaded, used as it is (MiniMax-H3 and MiniMax-H3-Image ship the same
        one); then `text_encoder_fp8` is not applied again. `text_encoder_path`: an FP8 text encoder that is already
        converted (the downloaded ModelOpt copy or a saved one, `codecopsd.download.vlm_fp8_source`), read with
        `text_encoder_fp8` instead of converting the bf16 release (`models.fp8_linear.load_fp8_text_encoder`).
        """
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        if load_transformer and sampler != "base":
            sampler_weights = sampler_weights or os.path.join(
                os.path.dirname(os.path.abspath(path)), CONFIG["samplers"][sampler]["weights"])
            if not os.path.exists(sampler_weights):
                raise FileNotFoundError(f"sampler {sampler!r} needs {sampler_weights} (README: Downloads); "
                                        "sampler='base' runs the released 49-call grid without it")
        processor = AutoProcessor.from_pretrained(path, subfolder="processor")
        shared = text_encoder is not None
        if not shared and text_encoder_fp8 and text_encoder_path:  # converted already: nothing to convert below
            text_encoder, shared = load_fp8_text_encoder(text_encoder_path, text_encoder_fp8, torch_dtype), True
        if not shared:
            text_encoder = AutoModelForImageTextToText.from_pretrained(path, subfolder="text_encoder", dtype=torch_dtype)
        transformer = None
        if load_transformer and sampler == "fasth3":
            transformer = load_fasth3_transformer(MiniMaxH3CodecOPSDTransformer3DModel, sampler_weights, torch_dtype)
        elif load_transformer and sampler == "flash":
            transformer = H3FlashCodecOPSDTransformer3DModel.from_flash(sampler_weights, torch_dtype)
        elif load_transformer:
            transformer = MiniMaxH3CodecOPSDTransformer3DModel.from_pretrained(
                path, subfolder="transformer_ref", torch_dtype=torch_dtype
            )
            if sampler == "pika":  # before the CodecOPSD LoRA, which wraps these layers
                load_pika_adapter(transformer, sampler_weights)
        pipe = cls(
            scheduler=MiniMaxH3Scheduler.from_pretrained(path, subfolder="scheduler"),
            vae=AutoencoderKLMiniMaxH3.from_pretrained(path, subfolder="vae", torch_dtype=torch.bfloat16),
            text_encoder=text_encoder, tokenizer=processor.tokenizer, processor=processor, transformer=transformer,
        )
        if text_encoder_fp8 and not shared:  # "fp8": FP8 weights and FP8 products; "bf16": FP8 weights, bf16 products
            text_encoder_to_fp8(pipe.text_encoder, text_encoder_fp8)
        pipe.sampler = sampler
        pipe.minimax_h3_path = path  # the audio VAE and audio scheduler are read from it on the first audio request
        return pipe

    _audio = None  # (audio VAE, audio scheduler config) once loaded (`audio_parts`)

    def audio_parts(self):
        """The audio VAE (float32, as released; 0.6 GB) and the audio scheduler's config (shift 3) of the MiniMax-H3
        folder, loaded on the first request with `audio=True` and kept in CPU memory between requests. They are not
        pipeline components, so the model offloads and `pipe.to()` never move them; a request moves the audio VAE to
        the GPU for its decode only."""
        if self._audio is None:
            from diffusers import AutoencoderKLMiniMaxH3Audio
            path = getattr(self, "minimax_h3_path", None)
            if path is None or not os.path.isdir(os.path.join(path, "audio_vae")):
                raise FileNotFoundError(f"audio=True needs the audio_vae/ and audio_scheduler/ folders of MiniMax-H3 "
                                        f"next to its vae/ (in {path})")
            vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(path, subfolder="audio_vae").eval()
            vae.set_attention_backend("native")  # the float32 audio VAE runs on SDPA, as HyperFlow H3 Flash sets it
            self._audio = (vae, dict(MiniMaxH3Scheduler.load_config(path, subfolder="audio_scheduler")))
        return self._audio

    def load_condition_lora(self, path_or_repo_id, weight_name=None):
        """Load a CodecOPSD student LoRA (local .safetensors or `weight_name` in a Hub repo): tensors
        `transformer_blocks.{i}.{module}.lora_{A,B}.weight`, metadata `lora_alpha` and `gazer` ("codec"). Every load
        of one of our LoRAs counts once in its repository's Hugging Face download statistics: a local file through
        `codecopsd.download.count_lora_load`, a Hub repository through the request for its .safetensors file (a HEAD
        when it is cached), which the Hub counts."""
        if os.path.isfile(path_or_repo_id):
            path = path_or_repo_id
            count_lora_load(path)
        else:
            path = hf_hub_download(path_or_repo_id, weight_name)
        with safe_open(path, framework="pt") as f:
            self.condition_lora_metadata = f.metadata()
        self.transformer.load_condition_lora(load_file(path), float(self.condition_lora_metadata["lora_alpha"]))

    def enable_model_cpu_offload(self, gpu_id=None, device="cuda"):
        """Diffusers' model offload (one model on the GPU at a time). The transformer's AdaLN projections (26 GB,
        applied to one row per timestep) also stay in CPU memory and are streamed in block by block (`stream_adaln`,
        exact, no time added), so the transformer needs 36 GB plus activations instead of 62 GB on every GPU."""
        if self.transformer is not None:
            self.transformer.stream_adaln()
        super().enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def enable_pinned_offload(self, device="cuda", stream_text_encoder=True, stream_blocks=0, pin_text_encoder=True):
        """One model on the GPU at a time, the others in page-locked CPU memory (`codecopsd.offload`): moving a model takes
        about 0.4 s (HyperFlow H3 Flash, 16 GB) on PCIe 5, about 10 times less than `enable_model_cpu_offload`. With
        `stream_text_encoder` the text encoder's decoder layers (26 GB in FP8) are streamed in one by one while it runs
        instead of moving with it, so the largest model on the GPU is the transformer. `stream_blocks` (a number, or
        "all") streams the transformer's last blocks the same way at every step: 24, 16 and 12 GB GPUs (the GPU memory
        plans of inference/extreme.py). The VAE moves in two parts: only its encoder (with `quant_conv`) while the
        references are encoded, only its decoder (with `post_quant_conv`, 4.5 GB) for the decode. Exact. The seconds of
        every whole-model move are in `self.moves`. With `pin_text_encoder=False` a text encoder read as file views (an
        FP8 VLM folder, `from_minimax_h3(text_encoder_path=...)`) stays in the page cache, read at each request: about
        26 GB less page-locked CPU memory, a slower text encoder step."""
        stream = {"text_encoder": self.text_encoder.model.language_model.layers} if stream_text_encoder else {}
        blocks = [b for b in self.transformer.transformer_blocks if any(True for _ in b.parameters())]
        count = len(blocks) if stream_blocks == "all" else int(stream_blocks or 0)
        if count:
            stream["transformer"] = blocks[-count:]
        vae = self.vae
        self.moves = pinned_model_offload(
            {"text_encoder": self.text_encoder, "vae": vae, "transformer": self.transformer}, device,
            stream=stream or None, pageable=() if pin_text_encoder else ("text_encoder",),
            parts={"vae": [[vae.encoder, vae.quant_conv], [vae.post_quant_conv, vae.decoder]]})

    def offload_all(self):
        """With `enable_pinned_offload`: every model back to its CPU copy (the GPU is free between requests)."""
        offload_all([self.text_encoder, self.vae, self.transformer])

    def enable_block_streaming(self, device="cuda"):
        """Run any size on one 80 GB GPU: the transformer (62 GB) stays in CPU memory and reaches `device` block by block
        on a side stream, overlapped with compute (diffusers group offload); the VAE (10 GB) and the text encoder (50 GB) are on
        `device` only while they run. Exact. Use instead of `enable_model_cpu_offload`, after `load_condition_lora`."""
        device = torch.device(device)
        self.transformer.enable_group_offload(device, offload_type="block_level", num_blocks_per_group=1, use_stream=True)
        vae_offload = cpu_offload_with_hook(self.vae, device)[1]
        self._text_encoder_offload = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=vae_offload)[1]

    def _stage(self, name, note=""):
        """`self.clock.stage(name)` while a request runs (`codecopsd.stats`), else nothing."""
        clock = getattr(self, "clock", None)
        return clock.stage(name, note) if clock is not None else nullcontext()

    @property
    def _execution_device(self):
        # Without offloading the pipeline runs where the VAE is (the text encoder may sit on another GPU).
        return self.vae.device if _resident(self.vae) else super()._execution_device

    reference_sizes = None  # `set_reference_sizes`; None = the released sizes

    def set_reference_sizes(self, picture_short_edge=None, picture_long_edge=None, video_short_edge=None):
        """Read the references at other sizes than the defaults (pictures at their own short edge within 1024-2048,
        videos at their output's tier):
        pictures resized (LANCZOS) to `picture_short_edge`, either one size or a (low, high) range that keeps a
        picture's own short edge when it lies inside and brings it to the nearer bound otherwise, their long edge at
        most `picture_long_edge`; videos on the canvas of short edge `video_short_edge` (area capped at that edge's 16:9
        frame, multiples of 32) instead of their output's tier. None keeps the default."""
        self.reference_sizes = dict(picture_short_edge=picture_short_edge, picture_long_edge=picture_long_edge,
                                    video_short_edge=video_short_edge)

    def normalize_references(self, references, num_frames, height=None, width=None):
        """References onto the model's conventions: a picture at its own short edge brought into 1024-2048
        (references.picture_short_edge; released MiniMax-H3 reads every picture at 2048); a video first resized
        onto the canvas of its own aspect ratio at the tier of the `height` x `width` output (`video_tier`, Pika H3
        Flex's rule; without an output size the released 768 canvas), then at 24 fps, cut to `num_frames` and then to
        the longest 17n + 5 frames the VAE encodes. Other sizes: `set_reference_sizes`."""
        sizes = self.reference_sizes or {}
        short_edge, long_edge = sizes.get("picture_short_edge") or PICTURE_SHORT_EDGE, sizes.get("picture_long_edge")
        video_edge = sizes.get("video_short_edge")
        canvas = ((video_edge, video_edge * ((video_edge * 16 // 9 + 31) // 32 * 32)) if video_edge
                  else video_tier(height, width) if height and width else (CANVAS["short_edge"], CANVAS["max_pixels"]))
        out = []
        for ref in references:
            if ref.kind == "image":
                image = ref.image if isinstance(ref.image, Image.Image) else Image.fromarray(np.asarray(ref.image))
                image = image.convert("RGB")
                width, height = image.size
                if isinstance(short_edge, (tuple, list)):  # a range: the picture's own short edge, kept inside it
                    low, high = short_edge
                    scale = min(max(min(width, height), low), high) / min(width, height)
                else:
                    scale = short_edge / min(width, height)
                if long_edge:
                    scale = min(scale, long_edge / max(width, height))
                size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
                image = image if image.size == size else image.resize(size, Image.LANCZOS)
                out.append(MiniMaxH3ImageReference(image=image))
            elif ref.kind == "video":
                frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                    ref.frames, float(ref.fps), num_frames, CANVAS["canvas_multiple"], *canvas, float(MINIMAX_H3_FPS),
                )
                if len(frames) < 22:
                    raise ValueError(f"a reference video needs at least 22 frames at 24 fps, got {len(frames)}")
                frames = frames[: (len(frames) - 5) // 17 * 17 + 5]
                out.append(MiniMaxH3VideoReference(frames=frames, fps=MINIMAX_H3_FPS))
            else:
                raise ValueError(f"references are pictures or videos, got {ref.kind!r}")
        return out

    @torch.no_grad()
    def encode(self, prompt, references, num_frames, gaze=True, video_drop=None, vlm_drop=False, teacher_text=False,
               gaze_overrides=None, height=None, width=None):
        """Everything the transformer reads about one request.

        Args:
            gaze: compute the codec gazer masks of the references.
            video_drop: with `gaze`, the drop of the video references (`codecopsd.codec_gazer.keep_mask`: None = the
                threshold decision, "elastic", or a share for priority drop); pictures keep the threshold decision.
            vlm_drop: with `gaze`, the prompt embedding is the student's: the text encoder reads only the reference vision
                tokens of the masks' `vlm_keep` (`vlm_drop.py`).
            teacher_text: also return the released prompt embedding (every vision token), for training.
            height, width: the output size, which picks the canvas of the reference videos (`video_tier`); None = the
                released 768 canvas (training).

        Returns:
            dict: "references" (normalized), "latents" ([1, 24, T', h, w] per reference), "masks" (the transformer's
            keep-masks, bool [T', h, w] per reference; empty without `gaze`), "vlm_masks" (the student text encoder's masks
            with `vlm_drop`, else None per reference), "text" ([1, L', 5120] text encoder embedding:
            the student's with `vlm_drop`), "text_tags" (H3 modality tag of every row of the full presentation [L]),
            "text_keep" (bool [L]: the rows of the full presentation in "text"; None = all), and with `teacher_text`
            "teacher_text" ([1, L, 5120]).

        Each reference moves to the GPU once. Its gaze starts from the CPU frames just before that copy (the same uint8
        bytes, so the same masks; x264 + the reader on the CPU, one thread per reference), so it runs while the
        references are copied and the GPU encodes them with the VAE; the masks are collected after.
        When the VAE and the text encoder both stay on the GPU (no offload; the text encoder may sit on another GPU), the text encoder runs at
        the same time as the VAE, in its own thread on its own CUDA stream (with `vlm_drop` it waits for the masks, which
        are ready long before the VAE); with offloading they run one after the other (one model on the GPU at a time).
        The seconds of each stage are left in `self.timings` (`stage_times`).
        """
        t0 = time.perf_counter()
        device = self._execution_device
        with self._stage("references: resize, copy to the GPU"):
            references = self.normalize_references(references, num_frames, height, width)
            host = [torch.from_numpy(np.asarray(r.image)[None] if r.kind == "image" else np.ascontiguousarray(r.frames))
                    for r in references]
            vlm_drop = vlm_drop if gaze else False  # True / "dit" (the text encoder reads the transformer's tokens)
            pool = ThreadPoolExecutor(len(host) + 2)
            # the gazes start from the CPU frames (the same uint8 bytes the GPU gets) while they are copied to the GPU
            tg = time.perf_counter()
            gazes = [
                submit_gaze(pool, h.permute(0, 3, 1, 2)[0] if r.kind == "image" else h.permute(0, 3, 1, 2),
                            CONFIG["references"]["gaze_kinds"][r.kind], MINIMAX_H3,
                            video_drop if r.kind == "video" else None, vlm=vlm_drop, overrides=gaze_overrides)
                for r, h in zip(references, host, strict=True)
            ] if gaze else []
            gaze_end = [0.0]
            for g in gazes:  # when the last gaze ends, from its start
                g.add_done_callback(lambda _: gaze_end.__setitem__(0, max(gaze_end[0], time.perf_counter() - tg)))
            pixels = [h.to(device) for h in host]
        t1 = time.perf_counter()  # the references are prepared and on the GPU: the stages below are timed from here
        parallel = _resident(self.vae) and _resident(self.text_encoder)
        with pool:
            vlm_keeps = (lambda: [g.result().vlm_keep for g in gazes]) if vlm_drop else None
            # one text encoder call at a time (the VLM drop hooks the model while it runs)
            prompts = lambda dev: (self.encode_prompt(prompt, references, dev, vlm_keeps),
                                   self.encode_prompt(prompt, references, dev)[0] if teacher_text else None)
            if parallel:
                encoded = pool.submit(self._on_side_stream, prompts, device)
            vae_stage = "references: VAE encode" + (" (text encoder at the same time)" if parallel else "")
            with (nullcontext() if parallel else self._stage(vae_stage)):
                latents = self.encode_references(pixels, device)
                torch.cuda.current_stream(device).synchronize()  # the VAE's stream only (the text encoder has its own)
            vae_end = time.perf_counter() - t1
            with self._stage("codec pruning: waiting for it after the VAE"):
                masks, vlm_masks = [g.result().keep for g in gazes], [g.result().vlm_keep for g in gazes]
            masks_end = time.perf_counter() - t1
            if parallel:
                ((text, text_tags, text_keep), teacher), text_s = encoded.result()
            else:
                t = time.perf_counter()
                with self._stage("text encoder (VLM)"):
                    (text, text_tags, text_keep), teacher = prompts(device)
                    torch.cuda.synchronize(device)
                text_s = time.perf_counter() - t
        del pixels
        t = time.perf_counter()
        if getattr(self, "_text_encoder_offload", None) is not None:  # block streaming: the text encoder back to the CPU
            self._text_encoder_offload.offload()
        self.timings = dict(encode=time.perf_counter() - t0, prepare=t1 - t0, vae=vae_end, gaze=gaze_end[0],
                            gaze_wait=max(0.0, masks_end - vae_end), text=text_s, text_parallel=parallel,
                            offload=time.perf_counter() - t)
        clock = getattr(self, "clock", None)
        if clock is not None:
            if parallel:
                clock.add("references: VAE encode and text encoder (at the same time)", max(vae_end, text_s))
            if gazes:
                vae_row = clock.add(vae_stage if not parallel else
                                    "references: VAE encode and text encoder (at the same time)", 0.0)
                vae_row["note"] = f"codec pruning ran {gaze_end[0]:.1f} s on the CPU meanwhile"
        out = dict(references=references, latents=latents, masks=masks, vlm_masks=vlm_masks, text=text,
                   text_tags=text_tags, text_keep=text_keep)
        if teacher_text:
            out["teacher_text"] = teacher
        return out

    def _on_side_stream(self, prompts, device):
        """`prompts(device)` (the prompt encodings of `encode`) on the text encoder's own device and a CUDA stream of its own
        (run from a worker thread); the embeddings come back on `device`."""
        text_device = self.text_encoder.device
        stream = torch.cuda.Stream(text_device)
        t = time.perf_counter()
        with torch.cuda.device(text_device), torch.cuda.stream(stream):
            (text, text_tags, text_keep), teacher = prompts(text_device)
        stream.synchronize()
        seconds = time.perf_counter() - t
        return ((text.to(device), text_tags, text_keep), (teacher.to(device) if teacher is not None else None)), seconds

    @torch.no_grad()
    def encode_references(self, pixels, device):
        """uint8 [T, H, W, 3] per reference -> normalized latents [1, 24, T', H / 16, W / 16] on `device`.

        As the CodecOPSD students were trained: a picture as the released model encodes it (posterior sampled under seed 42,
        rounded to float16), a video as the posterior mode.
        """
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = []
        for p in pixels:
            x = p.permute(3, 0, 1, 2)[None]  # [1, 3, T, H, W] uint8
            if x.shape[2] == 1:
                latents.append(encode_vae_condition(self.vae, x, PIXEL_MEAN, PIXEL_STD, 42).to(device))
                continue
            pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
            x = (x.float() / 255 - pixel_mean) / torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
            latents.append((self.vae.encode(x).latent_dist.mode().float() - mean) / std)
        return latents

    @torch.no_grad()
    def encode_prompt(self, prompt, references, device, vlm_keeps=None):
        """The text encoder's hidden_states[50] of the released presentation (`<Picture i>: ` / `<Video k>: ` blocks, then the
        prompt) and the H3 modality tag of each row [L].

        `vlm_keeps` (the student, VLM drop): the references' text encoder masks (bool [T', h, w] each, or a callable that
        returns them: called once the vision inputs are ready); only their vision tokens are read. Returns the
        embedding [1, L', 5120] of the kept rows, the tags of every row [L] and the bool keep [L] of the rows
        (None without `vlm_keeps`: every row, L' = L).
        """
        vision, image_counts, video_counts, video_times = self.text_step._gather_vision_features(
            self.processor, references, MINIMAX_H3_FPS
        )
        token_ids, tags = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
            self.tokenizer, prompt, references, image_counts, video_counts, video_times,
            text_tag=MINIMAX_H3_TEXT_TAG, video_tag=MINIMAX_H3_VIDEO_TAG,
        )
        args = (self.text_encoder, self.processor, token_ids, vision, TEXT_LAYER)
        if vlm_keeps is None:
            return vlm_prompt_embeds(*args, device=device, dtype=torch.bfloat16), torch.tensor(tags), None
        keeps = vlm_keeps() if callable(vlm_keeps) else vlm_keeps
        vision_keep = vision_token_keep(references, keeps, vision, self.processor, MINIMAX_H3_FPS, CONFIG["vlm_video_fps"])
        with drop_vision_rows(self.text_encoder, vision_keep) as kept:
            embeds = vlm_prompt_embeds(*args, device=device, dtype=torch.bfloat16)
        return embeds, torch.tensor(tags), kept["rows"]

    @staticmethod
    def pack(latents, shape, generator):
        """Noise of one sample: each reference's latents noised to t = 0.999 (one draw per reference, all its rows),
        then the video noise. Returns the video rows [N, 96] and the dense reference rows [M, 96]."""
        device = latents[0].device
        reference_rows = []
        for z in latents:
            noise = randn_tensor(z.shape, generator=generator, device=device, dtype=torch.float32)
            reference_rows.append(patchify_video_latents(REFERENCE_T * z + (1 - REFERENCE_T) * noise, (1, 2, 2)))
        rows = patchify_video_latents(randn_tensor(shape, generator=generator, device=device, dtype=torch.float32), (1, 2, 2))
        return rows, torch.cat(reference_rows)

    @staticmethod
    def layout(text_tags, kinds, latents, shape, num_frames, keep=None, text_keep=None):
        """The packed layout `[text | target audio | target video | kept reference rows]` of one sample.

        Rotary positions, modality tags and the media clock are those of the released packing (diffusers
        `build_ref2va_packed_sequence`, order `[text | references | audio | video]`); only the physical order differs,
        and dropped reference rows (`keep`, bool per dense reference row) and dropped text rows (`text_keep`, bool per
        row of the full presentation: the student's VLM drop) are left out without moving the others. `text_tags`: the
        tags of the full presentation. `kinds`: "image" / "video" per reference. Returns the transformer's layout
        arguments plus `start` (the first reference row) and the number of text and audio rows.
        """
        device = latents[0].device
        build = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence
        positions, tags, video_idx, audio_idx, text_idx, num_ref, _ = build(
            text_tags, [SimpleNamespace(kind=k, has_audio=False) for k in kinds], latents, [], shape[2], shape[3],
            shape[4], audio_latent_num_frames(num_frames), (1, 2, 2), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )
        ref_idx = video_idx[:num_ref] if keep is None else video_idx[:num_ref][keep.cpu()]
        text_idx = text_idx if text_keep is None else text_idx[text_keep.cpu()]
        order = torch.cat([text_idx, audio_idx, video_idx[num_ref:], ref_idx])
        num_text, num_audio = len(text_idx), len(audio_idx)
        return dict(
            position_ids=positions[order].to(device), token_tags=tags[order].to(device),
            text_indices=torch.arange(num_text, device=device),
            audio_indices=torch.arange(num_text, num_text + num_audio, device=device),
            video_indices=torch.arange(num_text + num_audio, len(order), device=device),
            start=len(order) - len(ref_idx), num_text=num_text, num_audio=num_audio,
        )

    @staticmethod
    def velocity(transformer, rows, reference_rows, text, layout, t, lora=True, endpoint=None, audio_rows=None,
                 audio_t=None, audio_endpoint=None):
        """Data-ward velocity of the video rows at timestep `t` (t = 1 - sigma): the video and text rows at t, the
        references at 0.999, the silent audio rows clean (t = 1). `lora=False` runs the released weights (the teacher).
        `endpoint` (HyperFlow H3 Flash): the endpoint r of the step for the video and text rows; the references and
        the audio rows go nowhere (r = t). `transformer` may be wrapped (e.g. for data-parallel training).

        `audio_rows` (a generated soundtrack, `audio=True` of `__call__`): the audio rows [num_audio, 32] at their own
        timestep `audio_t` (and endpoint `audio_endpoint` with Flash) instead of silent and clean; the call then
        returns (video velocity, audio velocity)."""
        n, start, num_text, num_audio = len(rows), layout["start"], layout["num_text"], layout["num_audio"]
        sound = audio_rows is not None
        row_t = torch.full((len(layout["token_tags"]),), float(t), device=rows.device)
        row_t[num_text : num_text + num_audio] = float(audio_t) if sound else 1.0
        row_t[start:] = REFERENCE_T
        kwargs = {k: v for k, v in layout.items() if k not in ("start", "num_text", "num_audio")}
        if endpoint is None:
            timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
        else:  # distinct (t, r) pairs, as Flash's build_row_time_pairs
            row_r = torch.full_like(row_t, float(endpoint))
            row_r[num_text : num_text + num_audio] = float(audio_endpoint) if sound else 1.0
            row_r[start:] = REFERENCE_T
            pairs, timestep_indices = torch.unique(torch.stack([row_t, row_r], 1), dim=0, sorted=True,
                                                   return_inverse=True)
            timestep, kwargs["endpoint_timestep"] = pairs[:, 0].contiguous(), pairs[:, 1].contiguous()
        channels = getattr(transformer, "module", transformer).config.audio_in_channels
        audio = audio_rows[None] if sound else torch.zeros(1, num_audio, channels, device=rows.device)
        out = transformer(
            hidden_states=torch.cat([rows, reference_rows])[None], audio_hidden_states=audio,
            encoder_hidden_states=text, timestep=timestep, timestep_indices=timestep_indices,
            reference_start=start if lora else None, return_dict=False, **kwargs,
        )
        return (out[0][0, :n], out[1][0]) if sound else out[0][0, :n]

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        references: list,
        height: int | None = None,
        width: int | None = None,
        num_frames: int = CONFIG["default_num_frames"],
        num_inference_steps: int | None = None,
        gaze: bool | None = None,
        video_drop: float | str | None = CONFIG["references"]["video_drop"],
        vlm_drop: bool | float | str | None = None,
        gaze_overrides: dict | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        audio: bool = False,
    ):
        r"""Generate one video (with `audio`, a stereo 32 kHz soundtrack too).

        Args:
            prompt: the text prompt; refer to the references as `<Picture i>` / `<Video k>` (numbered per modality).
            references: `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` (diffusers), plain RGB, in prompt order.
            height, width: output size, multiples of 32 (default 480 x 832 = 480p, the demo's default; 768 x 1344 = 768p).
            num_frames: frames at 24 fps, rounded up to 17n + 5.
            num_inference_steps: sampler "base": sigma grid points (default the released 50, i.e. 49 transformer
                calls); sampler "flash": 4, 6 (default) or 8 calls on HyperFlow's grids. The other few-step samplers
                fixed at loading (`from_minimax_h3(sampler=...)`) run their own 6 (Pika H3 Flex) or 8 (FastH3) calls.
                No guidance.
            gaze: give only the reference tokens the codec gazer keeps (default: what the loaded LoRA was trained with,
                off without a LoRA). The gaze runs on the CPU while the GPU encodes the references.
            video_drop: with `gaze`, how the video references are cut (`codecopsd.codec_gazer.keep_mask`): a share in
                (0, 1) = at least that share of every video's tokens dropped (default 0.70, at any resolution and
                length): priority drop, or the threshold decision when it drops more (`priority_drop.select: max`), "elastic" = elastic drop, None = the threshold decision alone. Pictures always keep the
                threshold decision.
            vlm_drop: with `gaze`, the student's text encoder reads only the gazed vision tokens of the references
                (default: what the loaded LoRA was trained with); "dit" = exactly the transformer's tokens (extreme); a
                number = the picture threshold scale of the text encoder mask (default 0.25, `codec_gazer.yaml`).
            gaze_overrides: {"video" / "image": {"motion_px": ..., "residual": ...}} replaces the codec pruning
                thresholds of that form (`codec_gazer.yaml` kinds) for this call; None = the config's.
            generator: noise of the references (one draw each, in order), then of the video, then (Pika H3 Flex) the
                fresh noise of every call.
            output_type: "np", "pil", "pt" or "latent".
            audio: also generate the soundtrack, as the released model and HyperFlow H3 Flash do: the audio rows start
                from noise (drawn after the video noise, so the video noise of a seed stays the same) and step down
                the audio scheduler's schedule (shift 3) next to the video, in the same transformer calls (the rows
                are in every call anyway; off, they are silent and clean). The output then also holds `audio` (float32
                [2, samples], or the audio rows for "latent") and `sampling_rate`. Samplers "flash" and "base". The
                CodecOPSD LoRA was trained with silent audio rows (off, the default, is what it was trained for).
        """
        lora = getattr(self, "condition_lora_metadata", {})
        gaze = lora.get("gazer", "none") != "none" if gaze is None else gaze
        vlm_drop = lora.get("vlm_drop", "false") == "true" if vlm_drop is None else vlm_drop
        if height is None or width is None:
            height, width = CONFIG["default_size"]
        if height % 32 or width % 32:
            raise ValueError(f"height and width must be multiples of 32, got {height} x {width}")
        num_frames = align_num_frames(num_frames, 17, 5)
        device = self._execution_device
        # Where the seconds go (`codecopsd.stats`): `self.clock` after the call, one row per stage.
        self.clock = Clock(getattr(self, "moves", None))

        # 1. References (VAE latents, codec gazer masks) and the prompt (text encoder).
        enc = self.encode(prompt, references, num_frames, gaze, video_drop, vlm_drop, gaze_overrides=gaze_overrides,
                          height=height, width=width)
        references, latents, text = enc["references"], enc["latents"], enc["text"]

        # 2. Noise (one draw per reference, all its rows, then the video) and the packed layout of the kept rows.
        with self._stage("noise and packed layout"):
            shape = (1, 24, video_latent_num_frames(num_frames, 17, 5), height // 16, width // 16)
            rows, reference_rows = self.pack(latents, shape, generator)
            keep = torch.cat([m.flatten() for m in enc["masks"]]).to(device) if gaze else None
            if keep is not None:
                reference_rows = reference_rows[keep]
            layout = self.layout(enc["text_tags"], [r.kind for r in references], latents, shape, num_frames, keep,
                                 enc["text_keep"])
            sampler = getattr(self, "sampler", "base")
            audio_rows = sound = None
            if audio:  # the soundtrack's noise, after the video's; its schedule (`samplers.audio_schedule`)
                audio_vae, audio_config = self.audio_parts()
                audio_rows = randn_tensor((layout["num_audio"], audio_vae.config.latent_channels), generator=generator,
                                          device=device, dtype=torch.float32)
                steps = num_inference_steps or (CONFIG["num_inference_steps"] if sampler == "base" else None)
                sound = audio_schedule(sampler, audio_config, device, steps)
        denoise = f"denoise ({len(layout['token_tags']):,} tokens per step)"

        def step_audio(i, out):  # the video velocity of a call; the audio rows take their own step
            nonlocal audio_rows
            if audio_rows is None:
                return out
            velocity, audio_velocity = out
            audio_rows = sound.scheduler.step(audio_velocity.float(), sound.scheduler.timesteps[i], audio_rows,
                                              return_dict=False)[0]
            return velocity

        def audio_kwargs(i):
            if audio_rows is None:
                return {}
            return dict(audio_rows=audio_rows, audio_t=sound.scheduler.timesteps[i],
                        audio_endpoint=sound.endpoints[i] if sound.endpoints is not None else None)

        # 3. Denoise, no guidance: the released sigma grid, or a few-step sampler (`samplers.py`).
        if sampler == "base":
            self.scheduler.set_timesteps(num_inference_steps or CONFIG["num_inference_steps"], device=device)
            for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
                with self._stage(denoise):
                    velocity = step_audio(i, self.velocity(self.transformer, rows, reference_rows, text, layout, t,
                                                           **audio_kwargs(i)))
                    rows = self.scheduler.step(velocity.float(), t, rows, return_dict=False)[0]
        else:
            if num_inference_steps is not None and sampler != "flash":
                raise ValueError(f"the {sampler} sampler runs its own calls; num_inference_steps is for the samplers "
                                 "'base' and 'flash' (4, 6 or 8)")
            plan = few_step(sampler, self.scheduler.config, self.transformer, device, num_inference_steps)
            rows = rows * plan.noise_scale
            for i, t in enumerate(self.progress_bar(plan.scheduler.timesteps)):
                with self._stage(denoise):
                    if plan.setup is not None:
                        plan.setup(i)
                    endpoint = plan.endpoints[i] if plan.endpoints is not None else None
                    velocity = step_audio(i, self.velocity(self.transformer, rows, reference_rows, text, layout, t,
                                                           endpoint=endpoint, **audio_kwargs(i)))
                    rows = plan.scheduler.step(velocity.float(), t, rows, generator=generator, return_dict=False)[0]

        _, c, f, h, w = shape
        video = rows.reshape(1, f, h // 2, w // 2, c, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(shape)
        if output_type != "latent":
            self.maybe_free_model_hooks()  # with model offload: the transformer leaves the GPU before the decode
            with self._stage("VAE decode"):
                mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
                std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
                video = self.vae.decode(video * std + mean, return_dict=False)[0].float()
            with self._stage("frames to the CPU"):
                video = video * torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1) + torch.tensor(
                    PIXEL_MEAN, device=device
                ).view(1, -1, 1, 1, 1)
                video = self.video_processor.postprocess_video(video.clamp(0, 1), output_type=output_type)
        soundtrack = sampling_rate = None
        if audio_rows is not None:
            sampling_rate = self._audio[0].config.sampling_rate
            if output_type == "latent":
                soundtrack = audio_rows
            else:
                with self._stage("audio VAE decode"):
                    soundtrack = self.decode_audio(audio_rows, device)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,) if soundtrack is None else (video, soundtrack)
        return MiniMaxH3CodecOPSDPipelineOutput(frames=video, audio=soundtrack, sampling_rate=sampling_rate)

    @torch.no_grad()
    def decode_audio(self, audio_rows, device):
        """The generated audio rows (channel-major [2 x N, 32], normalized) -> float32 stereo waveform [2, samples] on
        the CPU, as diffusers' MiniMax-H3 audio decode step: the mono audio VAE decodes the two channels as two batch
        items. The audio VAE is on the GPU for the decode only, unless the video VAE stays there (no offload)."""
        audio_vae = self.audio_parts()[0]
        audio_vae.to(device)
        try:
            latents = audio_rows.reshape(2, -1, audio_rows.shape[-1]).permute(0, 2, 1)
            mean = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
            std = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
            wave = audio_vae.decode(latents * std + mean, return_dict=False)[0].float()  # [2, 1, samples]
            return wave[:, 0].cpu().numpy()
        finally:
            if not _resident(self.vae):
                audio_vae.to("cpu")


def add_audio_track(path, audio, sampling_rate, bitrate="192k"):
    """Put a soundtrack (float [channels, samples] in [-1, 1]) into the video file `path` (its video stream copied as
    it is, the audio as AAC, cut to the video's length)."""
    import subprocess
    import tempfile
    import wave

    import imageio_ffmpeg
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).round().astype("<i2")
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(path))) as tmp:  # same file system
        wav, out = os.path.join(tmp, "audio.wav"), os.path.join(tmp, "video.mp4")
        with wave.open(wav, "wb") as w:
            w.setnchannels(pcm.shape[0])
            w.setsampwidth(2)
            w.setframerate(int(sampling_rate))
            w.writeframes(pcm.T.tobytes())
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(path), "-i", wav, "-map", "0:v:0",
                        "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate, "-shortest", "-movflags",
                        "+faststart", out], check=True)
        os.replace(out, path)
