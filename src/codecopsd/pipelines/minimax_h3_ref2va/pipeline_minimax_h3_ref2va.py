"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Prompt + reference pictures / videos -> video, with MiniMax-H3 (Ref2VA) and a CodecOPSD student condition LoRA."""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
from ...models import MiniMaxH3CodecOPSDTransformer3DModel, text_encoder_to_fp8
from .pipeline_output import MiniMaxH3CodecOPSDPipelineOutput
from .samplers import SAMPLERS, few_step, load_fasth3_transformer, load_pika_adapter
from .vlm_drop import drop_vision_rows, vision_token_keep

CONFIG = load("minimax_h3_ref2va")  # src/codecopsd/configs/minimax_h3_ref2va.yaml
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # the VAE's pixel normalization (ImageNet)
TEXT_LAYER = CONFIG["text_encoder_layer"]
REFERENCE_T = CONFIG["references"]["noise_t"]
CANVAS = dict(canvas_multiple=CONFIG["canvas"]["multiple"], short_edge=CONFIG["canvas"]["short_edge"],
              max_pixels=CONFIG["canvas"]["max_pixels"])
PICTURE_SHORT_EDGE = CONFIG["references"]["picture_short_edge"]

PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _resident(module):
    """On a GPU and not moved by an offload hook (`enable_model_cpu_offload`, `enable_block_streaming`)."""
    return not hasattr(module, "_hf_hook") and next(module.parameters()).device.type == "cuda"


def vlm_prompt_embeds(text_encoder, processor, token_ids, vision_inputs, text_encoder_layer, device, dtype):
    """The text encoder's `hidden_states[text_encoder_layer]`, as diffusers' MiniMax-H3 text encoder step computes it,
    without keeping the hidden state of every layer: the input of decoder layer `text_encoder_layer` is taken by a hook
    (the same tensor). The released model's prompts (every vision token of the references, 30K+ rows) otherwise hold
    50 x 10 KB per row and ran out of memory on 80 GB GPUs."""
    layer = text_encoder.model.language_model.layers[text_encoder_layer]
    captured = {}

    def take(module, args, kwargs):
        captured["hidden"] = args[0] if args else kwargs["hidden_states"]

    handle = layer.register_forward_pre_hook(take, with_kwargs=True)
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
        self.register_modules(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, processor=processor,
            transformer=transformer,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=16, do_normalize=False)
        self.text_step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=CONFIG["vlm_video_fps"])

    @classmethod
    def from_minimax_h3(cls, path, torch_dtype=torch.bfloat16, load_transformer=True, text_encoder_fp8=None,
                        sampler=CONFIG["sampler"], sampler_weights=None, text_encoder=None):
        """The pipeline from the released MiniMax-H3 folder (`hf download MiniMaxAI/MiniMax-H3`): the Ref2VA
        transformer (`transformer_ref`; not with `load_transformer=False`), the text encoder, the video VAE (diffusers' bf16
        loading: the decoder blocks in bf16, the encoder and the norms in float32; decoding about 7x faster than
        float32 on a B200) and the scheduler.

        `sampler` (`samplers.py`): "pika" (Pika H3 Flex, 6 calls), "fasth3" (FastH3 Omni 8-step, 8 calls) or "base" (the
        released grid, 49 calls). The few-step ones read `sampler_weights`: Pika's `ref2va/adapter.safetensors` or the
        FastH3 repo folder (default: `samplers.<sampler>.weights` of the config, next to `path`; README: Downloads).

        `text_encoder`: a text encoder that is already loaded, used as it is (MiniMax-H3 and MiniMax-H3-Image ship the same
        one); then `text_encoder_fp8` is not applied again.
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
        if not shared:
            text_encoder = AutoModelForImageTextToText.from_pretrained(path, subfolder="text_encoder", dtype=torch_dtype)
        transformer = None
        if load_transformer and sampler == "fasth3":
            transformer = load_fasth3_transformer(MiniMaxH3CodecOPSDTransformer3DModel, sampler_weights, torch_dtype)
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
        return pipe

    def load_condition_lora(self, path_or_repo_id, weight_name=None):
        """Load a CodecOPSD student LoRA (local .safetensors or `weight_name` in a Hub repo): tensors
        `transformer_blocks.{i}.{module}.lora_{A,B}.weight`, metadata `lora_alpha` and `gazer` ("codec")."""
        path = path_or_repo_id if os.path.isfile(path_or_repo_id) else hf_hub_download(path_or_repo_id, weight_name)
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

    def enable_block_streaming(self, device="cuda"):
        """Run any size on one 80 GB GPU: the transformer (62 GB) stays in CPU memory and reaches `device` block by block
        on a side stream, overlapped with compute (diffusers group offload); the VAE (10 GB) and the text encoder (50 GB) are on
        `device` only while they run. Exact. Use instead of `enable_model_cpu_offload`, after `load_condition_lora`."""
        device = torch.device(device)
        self.transformer.enable_group_offload(device, offload_type="block_level", num_blocks_per_group=1, use_stream=True)
        vae_offload = cpu_offload_with_hook(self.vae, device)[1]
        self._text_encoder_offload = cpu_offload_with_hook(self.text_encoder, device, prev_module_hook=vae_offload)[1]

    @property
    def _execution_device(self):
        # Without offloading the pipeline runs where the VAE is (the text encoder may sit on another GPU).
        return self.vae.device if _resident(self.vae) else super()._execution_device

    def normalize_references(self, references, num_frames):
        """References onto the released conventions: a picture at a 2048 short edge, a video at 24 fps on the canvas of
        its own aspect ratio, cut to `num_frames` and then to the longest 17n + 5 frames the VAE encodes."""
        out = []
        for ref in references:
            if ref.kind == "image":
                image = ref.image if isinstance(ref.image, Image.Image) else Image.fromarray(np.asarray(ref.image))
                image = image.convert("RGB")
                width, height = image.size
                scale = PICTURE_SHORT_EDGE / min(width, height)
                size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
                image = image if image.size == size else image.resize(size, Image.LANCZOS)
                out.append(MiniMaxH3ImageReference(image=image))
            elif ref.kind == "video":
                frames = MiniMaxH3Ref2VASetupStep._normalize_video_condition(
                    ref.frames, float(ref.fps), num_frames, CANVAS["canvas_multiple"], CANVAS["short_edge"],
                    CANVAS["max_pixels"], float(MINIMAX_H3_FPS),
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
               gaze_overrides=None):
        """Everything the transformer reads about one request.

        Args:
            gaze: compute the codec gazer masks of the references.
            video_drop: with `gaze`, the drop of the video references (`codecopsd.codec_gazer.keep_mask`: None = the
                threshold decision, "elastic", or a share for priority drop); pictures keep the threshold decision.
            vlm_drop: with `gaze`, the prompt embedding is the student's: the text encoder reads only the reference vision
                tokens of the masks' `vlm_keep` (`vlm_drop.py`).
            teacher_text: also return the released prompt embedding (every vision token), for training.

        Returns:
            dict: "references" (normalized), "latents" ([1, 24, T', h, w] per reference), "masks" (the transformer's
            keep-masks, bool [T', h, w] per reference; empty without `gaze`), "vlm_masks" (the student text encoder's masks
            with `vlm_drop`, else None per reference), "text" ([1, L', 5120] text encoder embedding:
            the student's with `vlm_drop`), "text_tags" (H3 modality tag of every row of the full presentation [L]),
            "text_keep" (bool [L]: the rows of the full presentation in "text"; None = all), and with `teacher_text`
            "teacher_text" ([1, L, 5120]).

        Each reference moves to the GPU once and its gaze starts there (uint8 on the GPU, then x264 + the reader on the
        CPU, one thread per reference) while the GPU encodes the references with the VAE; the masks are collected after.
        When the VAE and the text encoder both stay on the GPU (no offload; the text encoder may sit on another GPU), the text encoder runs at
        the same time as the VAE, in its own thread on its own CUDA stream (with `vlm_drop` it waits for the masks, which
        are ready long before the VAE); with offloading they run one after the other (one model on the GPU at a time).
        The seconds of each stage are left in `self.timings` (`stage_times`).
        """
        t0 = time.perf_counter()
        device = self._execution_device
        references = self.normalize_references(references, num_frames)
        pixels = [
            torch.from_numpy(np.asarray(r.image)[None] if r.kind == "image" else np.ascontiguousarray(r.frames)).to(device)
            for r in references
        ]
        t1 = time.perf_counter()  # the references are prepared and on the GPU: the stages below are timed from here
        vlm_drop = vlm_drop if gaze else False  # True / "dit" (the text encoder reads the transformer's tokens)
        parallel = _resident(self.vae) and _resident(self.text_encoder)
        with ThreadPoolExecutor(len(pixels) + 2) as pool:
            gazes = [
                submit_gaze(pool, p.permute(0, 3, 1, 2)[0] if r.kind == "image" else p.permute(0, 3, 1, 2),
                            CONFIG["references"]["gaze_kinds"][r.kind], MINIMAX_H3,
                            video_drop if r.kind == "video" else None, vlm=vlm_drop, overrides=gaze_overrides)
                for r, p in zip(references, pixels, strict=True)
            ] if gaze else []
            gaze_end = [0.0]
            for g in gazes:  # when the last gaze ends, from the start of `encode`
                g.add_done_callback(lambda _: gaze_end.__setitem__(0, max(gaze_end[0], time.perf_counter() - t1)))
            vlm_keeps = (lambda: [g.result().vlm_keep for g in gazes]) if vlm_drop else None
            # one text encoder call at a time (the VLM drop hooks the model while it runs)
            prompts = lambda dev: (self.encode_prompt(prompt, references, dev, vlm_keeps),
                                   self.encode_prompt(prompt, references, dev)[0] if teacher_text else None)
            if parallel:
                encoded = pool.submit(self._on_side_stream, prompts, device)
            latents = self.encode_references(pixels, device)
            torch.cuda.current_stream(device).synchronize()  # the VAE's stream only (the text encoder has its own)
            vae_end = time.perf_counter() - t1
            masks, vlm_masks = [g.result().keep for g in gazes], [g.result().vlm_keep for g in gazes]
            masks_end = time.perf_counter() - t1
            if parallel:
                ((text, text_tags, text_keep), teacher), text_s = encoded.result()
            else:
                t = time.perf_counter()
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
    def velocity(transformer, rows, reference_rows, text, layout, t, lora=True):
        """Data-ward velocity of the video rows at timestep `t` (t = 1 - sigma): the video and text rows at t, the
        references at 0.999, the silent audio rows clean (t = 1). `lora=False` runs the released weights (the teacher).
        `transformer` may be wrapped (e.g. for data-parallel training)."""
        n, start, num_text, num_audio = len(rows), layout["start"], layout["num_text"], layout["num_audio"]
        row_t = torch.full((len(layout["token_tags"]),), float(t), device=rows.device)
        row_t[num_text : num_text + num_audio] = 1.0
        row_t[start:] = REFERENCE_T
        timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
        channels = getattr(transformer, "module", transformer).config.audio_in_channels
        audio = torch.zeros(1, num_audio, channels, device=rows.device)
        kwargs = {k: v for k, v in layout.items() if k not in ("start", "num_text", "num_audio")}
        return transformer(
            hidden_states=torch.cat([rows, reference_rows])[None], audio_hidden_states=audio,
            encoder_hidden_states=text, timestep=timestep, timestep_indices=timestep_indices,
            reference_start=start if lora else None, return_dict=False, **kwargs,
        )[0][0, :n]

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
    ):
        r"""Generate one video.

        Args:
            prompt: the text prompt; refer to the references as `<Picture i>` / `<Video k>` (numbered per modality).
            references: `MiniMaxH3ImageReference` / `MiniMaxH3VideoReference` (diffusers), plain RGB, in prompt order.
            height, width: output size, multiples of 32 (default 480 x 832 = 480p; 768 x 1344 = 768p).
            num_frames: frames at 24 fps, rounded up to 17n + 5.
            num_inference_steps: sampler "base" only: sigma grid points (default the released 50, i.e. 49 transformer
                calls). The few-step samplers fixed at loading (`from_minimax_h3(sampler=...)`) run their own 6 (Pika
                H3 Flex) or 8 (FastH3) calls. No guidance.
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

        # 1. References (VAE latents, codec gazer masks) and the prompt (text encoder).
        enc = self.encode(prompt, references, num_frames, gaze, video_drop, vlm_drop, gaze_overrides=gaze_overrides)
        references, latents, text = enc["references"], enc["latents"], enc["text"]

        # 2. Noise (one draw per reference, all its rows, then the video) and the packed layout of the kept rows.
        shape = (1, 24, video_latent_num_frames(num_frames, 17, 5), height // 16, width // 16)
        rows, reference_rows = self.pack(latents, shape, generator)
        keep = torch.cat([m.flatten() for m in enc["masks"]]).to(device) if gaze else None
        if keep is not None:
            reference_rows = reference_rows[keep]
        layout = self.layout(enc["text_tags"], [r.kind for r in references], latents, shape, num_frames, keep,
                             enc["text_keep"])

        # 3. Denoise, no guidance: the released sigma grid, or a few-step sampler (`samplers.py`).
        sampler = getattr(self, "sampler", "base")
        if sampler == "base":
            self.scheduler.set_timesteps(num_inference_steps or CONFIG["num_inference_steps"], device=device)
            for t in self.progress_bar(self.scheduler.timesteps):
                velocity = self.velocity(self.transformer, rows, reference_rows, text, layout, t)
                rows = self.scheduler.step(velocity.float(), t, rows, return_dict=False)[0]
        else:
            if num_inference_steps is not None:
                raise ValueError(f"the {sampler} sampler runs its own calls; num_inference_steps is for sampler 'base'")
            plan = few_step(sampler, self.scheduler.config, self.transformer, device)
            rows = rows * plan.noise_scale
            for i, t in enumerate(self.progress_bar(plan.scheduler.timesteps)):
                if plan.setup is not None:
                    plan.setup(i)
                velocity = self.velocity(self.transformer, rows, reference_rows, text, layout, t)
                rows = plan.scheduler.step(velocity.float(), t, rows, generator=generator, return_dict=False)[0]

        _, c, f, h, w = shape
        video = rows.reshape(1, f, h // 2, w // 2, c, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(shape)
        if output_type != "latent":
            self.maybe_free_model_hooks()  # with model offload: the transformer leaves the GPU before the decode
            mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
            video = self.vae.decode(video * std + mean, return_dict=False)[0].float()
            video = video * torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1, 1) + torch.tensor(
                PIXEL_MEAN, device=device
            ).view(1, -1, 1, 1, 1)
            video = self.video_processor.postprocess_video(video.clamp(0, 1), output_type=output_type)
        self.maybe_free_model_hooks()
        return MiniMaxH3CodecOPSDPipelineOutput(frames=video) if return_dict else (video,)
