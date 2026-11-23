"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Prompt + reference pictures -> image, with MiniMax-H3-Image and a CodecOPSD student condition LoRA."""

import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3_image.before_encoder import bound_short_edge
from diffusers.modular_pipelines.minimax_h3_image.modular_pipeline import MINIMAX_H3_IMAGE_TAG, MINIMAX_H3_TEXT_TAG
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageOps
from transformers import (
    AutoModelForImageTextToText, AutoProcessor, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin,
)

from ...codec_gazer import MINIMAX_H3, submit_gaze
from ...configs import load
from ...models import MiniMaxH3ImageCodecOPSDTransformer2DModel, text_encoder_to_fp8
from ...models.fp8_linear import load_fp8_text_encoder
from ...stats import Clock
from ..minimax_h3_ref2va import MiniMaxH3CodecOPSDPipeline
from ..minimax_h3_ref2va.pipeline_minimax_h3_ref2va import vlm_prompt_embeds
from ..minimax_h3_ref2va.vlm_drop import drop_vision_rows, vision_token_keep

CONFIG = load("minimax_h3_image")  # src/codecopsd/configs/minimax_h3_image.yaml
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # the VAE's pixel normalization (ImageNet)
TEXT_LAYER = CONFIG["text_encoder_layer"]
REFERENCE_T = CONFIG["references"]["noise_t"]
PATCH = (1, 2, 2)


def reference_size(image):
    """(height, width) of an output at the aspect ratio of `image` (the first reference, PIL) on the released generation
    area (2048 x 2048): sides multiples of 32 and a fixed point of the canvas rule, so it is generated at exactly that
    size. The paper's GEdit-Bench and OmniContext runs size every output this way."""
    width, height = ImageOps.exif_transpose(image).size if isinstance(image, Image.Image) else image
    area, multiple = CONFIG["canvas"]["min_area"], CONFIG["canvas"]["multiple"]
    ratio = width / height
    size = (round((area / ratio) ** 0.5 / multiple) * multiple, round((area * ratio) ** 0.5 / multiple) * multiple)
    for _ in range(5):
        canvas = get_image_canvas_size(*size, area, multiple)
        if canvas == size:
            return size
        size = canvas
    raise ValueError(f"no canvas fixed point for a {width} x {height} picture")


class MiniMaxH3ImageCodecOPSDPipeline(DiffusionPipeline):
    r"""MiniMax-H3-Image (prompt + reference pictures -> image) with a CodecOPSD student condition LoRA.

    The CodecOPSD student is a LoRA on the reference rows of the released transformer, distilled on-policy from the released
    model on only the reference tokens the codec gazer keeps. Without a LoRA (or with `gaze=False` and the LoRA off)
    the pipeline is the released model.
    """

    model_cpu_offload_seq = "vae->text_encoder->transformer"
    _optional_components = ["transformer"]
    load_condition_lora = MiniMaxH3CodecOPSDPipeline.load_condition_lora
    _stage = MiniMaxH3CodecOPSDPipeline._stage  # `self.clock.stage(name)` while a request runs (`codecopsd.stats`)

    def __init__(
        self,
        scheduler: MiniMaxH3Scheduler,
        vae: AutoencoderKLMiniMaxH3Image,
        text_encoder: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        processor: ProcessorMixin,
        transformer: MiniMaxH3ImageCodecOPSDTransformer2DModel | None = None,
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
        self.image_processor = VaeImageProcessor(vae_scale_factor=16, do_normalize=False)

    @classmethod
    def from_minimax_h3_image(cls, path, torch_dtype=torch.bfloat16, load_transformer=True, text_encoder_fp8=None,
                              text_encoder=None, text_encoder_path=None):
        """The pipeline from the released MiniMax-H3-Image folder (`hf download MiniMaxAI/MiniMax-H3-Image`): the
        transformer (`transformer_diffusers`; not with `load_transformer=False`), the text encoder, the VAE (`vae_diffusers`,
        float32, as released) and the scheduler.

        `text_encoder`: a text encoder that is already loaded, used as it is (MiniMax-H3 and MiniMax-H3-Image ship the same
        one), e.g. `MiniMaxH3CodecOPSDPipeline.text_encoder`; then `text_encoder_fp8` is not applied again.
        `text_encoder_path`: an FP8 text encoder that is already converted (the downloaded ModelOpt copy of the same
        VLM, or a saved one; `codecopsd.download.vlm_fp8_source`), read with `text_encoder_fp8` instead of converting
        the bf16 release (`models.fp8_linear.load_fp8_text_encoder`; 50 layers: see `vlm_prompt_embeds`)."""
        processor = AutoProcessor.from_pretrained(path, subfolder="processor")
        shared = text_encoder is not None
        if not shared and text_encoder_fp8 and text_encoder_path:  # converted already: nothing to convert below
            text_encoder, shared = load_fp8_text_encoder(text_encoder_path, text_encoder_fp8, torch_dtype), True
        if not shared:
            text_encoder = AutoModelForImageTextToText.from_pretrained(path, subfolder="text_encoder", dtype=torch_dtype)
        transformer = None
        if load_transformer:
            transformer = MiniMaxH3ImageCodecOPSDTransformer2DModel.from_pretrained(
                path, subfolder="transformer_diffusers", torch_dtype=torch_dtype
            )
        pipe = cls(
            scheduler=MiniMaxH3Scheduler.from_pretrained(path, subfolder="scheduler"),
            # float32 as released: on a B200 a bf16 decoder is slower here (2048 x 2048 = 121 small tiles, one per call)
            vae=AutoencoderKLMiniMaxH3Image.from_pretrained(path, subfolder="vae_diffusers"),
            text_encoder=text_encoder, tokenizer=processor.tokenizer, processor=processor, transformer=transformer,
        )
        if text_encoder_fp8 and not shared:  # "fp8": FP8 weights and FP8 products; "bf16": FP8 weights, bf16 products
            text_encoder_to_fp8(pipe.text_encoder, text_encoder_fp8)
        return pipe

    def prepare_references(self, images):
        """Each picture upright and RGB, then its two released copies: the VAE's and the text encoder's (a short edge outside
        `vae_short_edge` / `vision_short_edge` is resized to the nearer bound, multiples of 32)."""
        copies = ([], [])
        for image in images:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 32:
                raise ValueError(f"a reference picture needs at least 32 pixels on each side, got {image.size}")
            for out, (low, high) in zip(copies, (CONFIG["references"]["vae_short_edge"],
                                                 CONFIG["references"]["vision_short_edge"])):
                size = bound_short_edge(*image.size, low, high, CONFIG["canvas"]["multiple"])
                out.append(image if image.size == size else self.image_processor.resize(image, size[1], size[0]))
        return copies

    @torch.no_grad()
    def encode(self, prompt, images, generator=None, gaze=True, vlm_drop=None, teacher_text=False, gaze_overrides=None):
        """Everything the transformer reads about one request.

        Args:
            images: the reference pictures (PIL), in prompt order (`<Picture 1>`, ...).
            generator: the request's generator; the VAE draws one posterior sample per reference from it.
            gaze: compute the codec gazer masks of the references.
            vlm_drop: with `gaze`, None = the released prompt embedding; a number = the student's, whose text encoder reads
                only the gazed vision tokens, at this picture threshold scale (1.0 = the transformer's mask).
            teacher_text: also return the released prompt embedding (every vision token), for training.

        Returns:
            dict: "latents" ([1, 24, h, w] per reference), "masks" (the transformer's keep-masks, bool [1, h / 2, w / 2]
            per reference; empty without `gaze`), "vlm_masks" (the student text encoder's masks with `vlm_drop`, else None
            per reference), "text" ([1, L', 5120]: the student's with `vlm_drop`), "text_tags"
            (H3 modality tag of every row of the full presentation [L]), "text_keep" (bool [L]: the rows in "text";
            None = all), and with `teacher_text` "teacher_text" ([1, L, 5120]).

        The gaze runs on the CPU, one thread per reference, while the GPU encodes the references with the VAE. The
        seconds of each stage are left in `self.timings` (`stage_times`).
        """
        t0 = time.perf_counter()
        device = self._execution_device
        with self._stage("references: resize, copy to the GPU"):
            vae_images, vision_images = self.prepare_references(images)
            # A reference keeps its own size: cut at the top left to whole VAE cells (16 px); its latents then to
            # whole tokens (32 px), which are the pixels the gazer reads.
            pixels = [torch.from_numpy(np.array(im)[: im.height // 16 * 16, : im.width // 16 * 16]).to(device)
                      for im in vae_images]
        vlm_drop = vlm_drop if gaze else None
        t1 = time.perf_counter()  # the references are prepared and on the GPU: the stages below are timed from here
        with ThreadPoolExecutor(max(1, len(pixels))) as pool:
            gazes = [
                submit_gaze(pool, p[: p.shape[0] // 32 * 32, : p.shape[1] // 32 * 32].permute(2, 0, 1),
                            CONFIG["references"]["gaze_kind"], MINIMAX_H3, vlm=vlm_drop if vlm_drop is not None else False,
                            overrides=gaze_overrides)
                for p in pixels
            ] if gaze else []
            gaze_end = [0.0]
            for g in gazes:  # when the last gaze ends, from the start of `encode`
                g.add_done_callback(lambda _: gaze_end.__setitem__(0, max(gaze_end[0], time.perf_counter() - t1)))
            with self._stage("references: VAE encode"):
                latents = self.encode_references(pixels, generator, device)
                torch.cuda.current_stream(device).synchronize()
            vae_end = time.perf_counter() - t1 if pixels else 0.0
            with self._stage("codec pruning: waiting for it after the VAE"):
                gazes = [g.result() for g in gazes]
            masks_end = time.perf_counter() - t1
        del pixels
        vlm_keeps = [g.vlm_keep for g in gazes] if vlm_drop is not None else None
        t = time.perf_counter()
        with self._stage("text encoder (VLM)"):
            text, text_tags, text_keep = self.encode_prompt(prompt, vision_images, device, vlm_keeps)
            torch.cuda.synchronize(device)
        self.timings = dict(encode=time.perf_counter() - t0, prepare=t1 - t0, vae=vae_end, gaze=gaze_end[0],
                            gaze_wait=max(0.0, masks_end - vae_end) if gazes else 0.0, text=time.perf_counter() - t,
                            text_parallel=False)
        out = dict(latents=latents, masks=[g.keep for g in gazes], vlm_masks=[g.vlm_keep for g in gazes], text=text,
                   text_tags=text_tags, text_keep=text_keep)
        if teacher_text:
            out["teacher_text"] = self.encode_prompt(prompt, vision_images, device)[0] if vlm_keeps else text
        return out

    @torch.no_grad()
    def encode_references(self, pixels, generator, device):
        """uint8 [H, W, 3] per reference (whole 16 px cells) -> normalized latents [1, 24, h, w], cropped to whole
        tokens. As released: the posterior sampled from `generator` (one draw per reference, in order), rounded to
        float16."""
        pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1)
        latents = []
        for p in pixels:
            x = (p.permute(2, 0, 1)[None].to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            z = self.vae.encode(x, return_dict=False)[0].sample(generator=generator).to(torch.float16).float()
            latents.append((z[..., : z.shape[-2] // 2 * 2, : z.shape[-1] // 2 * 2] - mean) / std)
        return latents

    @torch.no_grad()
    def encode_prompt(self, prompt, vision_images, device, vlm_keeps=None):
        """The text encoder's hidden_states[50] of the released presentation (per reference a `Picture i: ` label and its vision
        block, then the prompt; no chat template) and the H3 modality tag of each row [L].

        `vlm_keeps` (the student, VLM drop): the references' text encoder masks (bool [1, h, w] each, on the transformer's
        token grid of the VAE copy); only their vision tokens are read. Returns the embedding [1, L', 5120] of the kept
        rows, the tags of every row [L] and the bool keep [L] of the rows (None without `vlm_keeps`).
        """
        tokenizer, n, merge = self.tokenizer, CONFIG["max_sequence_length"], self.processor.image_processor.merge_size
        vision = {}
        if vision_images:
            vision = self.processor.image_processor(images=vision_images, return_tensors="pt")
            vision = {"pixel_values": vision["pixel_values"], "image_grid_thw": vision["image_grid_thw"]}
        ids, tags = [], []
        for i, grid in enumerate(vision.get("image_grid_thw", [])):
            label = tokenizer(f"Picture {i + 1}: ", add_special_tokens=False)["input_ids"][:n]
            block = tokenizer.convert_tokens_to_ids(
                ["<|vision_start|>"] + ["<|image_pad|>"] * (int(grid.prod()) // merge ** 2) + ["<|vision_end|>"]
            )
            ids += label + block
            tags += [MINIMAX_H3_TEXT_TAG] * len(label) + [MINIMAX_H3_IMAGE_TAG] * len(block)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:n]
        ids += prompt_ids
        tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)
        args = (self.text_encoder, self.processor, ids, vision, TEXT_LAYER)
        if vlm_keeps is None:
            return vlm_prompt_embeds(*args, device=device, dtype=torch.bfloat16), torch.tensor(tags), None
        refs = [SimpleNamespace(kind="image")] * len(vlm_keeps)
        vision_keep = vision_token_keep(refs, vlm_keeps, vision, self.processor, None, None)
        with drop_vision_rows(self.text_encoder, vision_keep) as kept:
            embeds = vlm_prompt_embeds(*args, device=device, dtype=torch.bfloat16)
        return embeds, torch.tensor(tags), kept["rows"]

    @staticmethod
    def pack(latents, shape, scheduler, generator):
        """Noise of one request, drawn after the VAE's: the generated image's, then one per reference, each reference
        noised to t = 0.999. Returns the generated rows [N, 96] and the dense reference rows [M, 96]."""
        device = latents[0].device if latents else None
        rows = patchify_video_latents(randn_tensor(shape, generator=generator, device=device, dtype=torch.float32), PATCH)
        reference_rows = [rows[:0]]
        for z in latents:
            noise = randn_tensor(z.shape, generator=generator, device=device, dtype=torch.float32)
            noised = scheduler.scale_noise(z, torch.tensor(REFERENCE_T), noise)
            reference_rows.append(patchify_video_latents(noised[:, :, None], PATCH))
        return rows, torch.cat(reference_rows)

    @staticmethod
    def layout(text_tags, latents, latent_height, latent_width, device, keep=None, text_keep=None):
        """The released packed layout `[text | references | generated image]` (diffusers `build_image_packed_sequence`)
        of the kept rows: dropped reference rows (`keep`, bool per dense reference row) and text rows (`text_keep`: the
        student's VLM drop) are left out, every other row keeps its place and rotary position. Returns the
        transformer's layout arguments plus the span of the reference rows."""
        positions, tags, image_idx, text_idx, num_ref = build_image_packed_sequence(
            text_tags, [tuple(z.shape[-2:]) for z in latents], latent_height, latent_width, PATCH, MINIMAX_H3_IMAGE_TAG
        )
        ref_idx = image_idx[:num_ref] if keep is None else image_idx[:num_ref][keep.cpu()]
        text_idx = text_idx if text_keep is None else text_idx[text_keep.cpu()]
        order = torch.cat([text_idx, ref_idx, image_idx[num_ref:]])
        num_text, num_ref = len(text_idx), len(ref_idx)
        return dict(
            position_ids=positions[order].to(device), token_tags=tags[order].to(device),
            text_indices=torch.arange(num_text, device=device),
            image_indices=torch.arange(num_text, len(order), device=device),
            reference_start=num_text, reference_end=num_text + num_ref,
        )

    @staticmethod
    def velocity(transformer, rows, reference_rows, text, layout, t, step, lora=True):
        """Data-ward velocity of the generated rows at timestep `t` (t = 1 - sigma) of step `step` (its output head):
        the generated and text rows at t, the references at 0.999. `lora=False` runs the released weights (the
        teacher). `transformer` may be wrapped (e.g. for data-parallel training)."""
        start, end = layout["reference_start"], layout["reference_end"]
        row_t = torch.full((len(layout["token_tags"]),), float(t), dtype=torch.float32)
        row_t[start:end] = REFERENCE_T
        timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
        kwargs = {k: v for k, v in layout.items() if k not in ("reference_start", "reference_end")}
        return transformer(
            hidden_states=torch.cat([reference_rows, rows])[None], encoder_hidden_states=text,
            timestep=timestep.to(rows.device), timestep_indices=timestep_indices.to(rows.device), pdd_step_index=step,
            reference_start=start if lora else None, reference_end=end, return_dict=False, **kwargs,
        )[0][0, end - start :]

    @staticmethod
    def set_timesteps(scheduler, device):
        """The released six-step grid on `scheduler`; returns its timesteps."""
        timesteps = torch.tensor([1.0 - s for s in CONFIG["sigmas"][:-1]], dtype=torch.float32)
        scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device=device)
        return scheduler.timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        images: list | Image.Image | None = None,
        height: int | None = None,
        width: int | None = None,
        gaze: bool | None = None,
        vlm_drop: bool | None = None,
        vlm_picture_scale: float | str | None = None,
        gaze_overrides: dict | None = None,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ):
        r"""Generate one image.

        Args:
            prompt: the text prompt; refer to the references as `<Picture i>`.
            images: the reference pictures (PIL), in prompt order; none = text to image.
            height, width: output size (default: the first picture's aspect ratio on the 2048 x 2048 area, as the demo;
                2048 x 2048 without pictures); below 2048 x 2048 pixels it is generated on a canvas of that area and
                resized back.
            gaze: give only the reference tokens the codec gazer keeps (default: what the loaded LoRA was trained with,
                off without a LoRA).
            vlm_drop: with `gaze`, the student's text encoder reads only the gazed vision tokens of the references
                (default: what the loaded LoRA was trained with).
            vlm_picture_scale: with `vlm_drop`, the picture thresholds of the student's text encoder mask are those of the
                transformer's mask x this scale (default 0.25, `codec_gazer.yaml` vlm_drop.picture_threshold_scale, as
                in the video model; the v2 LoRA file records 1.0, and 0.25 stays closer to the teacher on the demo cases);
                "dit" = the transformer's own mask.
            gaze_overrides: {"image": {"motion_px": ..., "residual": ...}} replaces the codec pruning thresholds of the
                pictures (`codec_gazer.yaml` kinds.image) for this call; None = the config's.
            generator: the VAE's posterior draws (one per reference), then the noise of the image and of the references.
            output_type: "pil" (default), "np", "pt" or "latent".
        """
        lora = getattr(self, "condition_lora_metadata", {})
        gaze = lora.get("gazer", "none") != "none" if gaze is None else gaze
        vlm_drop = lora.get("vlm_drop", "false") == "true" if vlm_drop is None else vlm_drop
        images = [] if images is None else images if isinstance(images, list) else [images]
        if height is None or width is None:  # the demo's default: the first picture's aspect ratio on the 2048 x 2048 area
            height, width = reference_size(images[0]) if images else CONFIG["default_size"]
        canvas_h, canvas_w = get_image_canvas_size(height, width, CONFIG["canvas"]["min_area"], CONFIG["canvas"]["multiple"])
        device = self._execution_device
        # Where the seconds go (`codecopsd.stats`): `self.clock` after the call, one row per stage.
        self.clock = Clock(getattr(self, "moves", None))

        # 1. References (VAE latents, codec gazer masks) and the prompt (text encoder).
        scale = True if vlm_picture_scale is None else vlm_picture_scale  # True: codec_gazer.yaml (0.25)
        enc = self.encode(prompt, images, generator, gaze, scale if vlm_drop else None, gaze_overrides=gaze_overrides)

        # 2. Noise (the image, then one draw per reference) and the packed layout of the kept rows.
        with self._stage("noise and packed layout"):
            shape = (1, 24, 1, canvas_h // 16, canvas_w // 16)
            rows, reference_rows = self.pack(enc["latents"], shape, self.scheduler, generator)
            rows, reference_rows = rows.to(device), reference_rows.to(device)
            keep = torch.cat([m.flatten() for m in enc["masks"]]).to(device) if gaze and images else None
            if keep is not None:
                reference_rows = reference_rows[keep]
            layout = self.layout(enc["text_tags"], enc["latents"], *shape[-2:], device, keep, enc["text_keep"])
        denoise = f"denoise ({len(layout['token_tags']):,} tokens per step)"

        # 3. Denoise: the released six-step grid, head i at step i, no guidance.
        for i, t in enumerate(self.progress_bar(self.set_timesteps(self.scheduler, device))):
            with self._stage(denoise):
                velocity = self.velocity(self.transformer, rows, reference_rows, enc["text"], layout, t, i)
                rows = self.scheduler.step(velocity.float(), t, rows, return_dict=False)[0]

        h, w = shape[-2:]
        image = rows.reshape(1, h // 2, w // 2, 24, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(1, 24, h, w)
        if output_type != "latent":
            self.maybe_free_model_hooks()  # with model offload: the transformer leaves the GPU before the decode
            with self._stage("VAE decode"):
                mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1)
                std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1)
                image = self.vae.decode(image * std + mean, return_dict=False)[0]
            with self._stage("image to the CPU"):
                pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
                image = (image.float() * torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
                         + pixel_mean).clamp(0, 1)
                image = (image * 255).clamp(0, 255).to(torch.uint8).float() / 255
                if (height, width) != tuple(image.shape[-2:]):  # generated on a larger canvas: back to the asked size
                    image = [self.image_processor.resize(im, height, width)
                             for im in self.image_processor.postprocess(image, output_type="pil")]
                    image = self.image_processor.numpy_to_pt(self.image_processor.pil_to_numpy(image))
                image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=image) if return_dict else (image,)
