"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Cosmos 3 (Nano) with a CodecOPSD student condition LoRA: control -> video transfer (edge, blur, depth, segmentation; one to
four at once), image -> video, video continuation, video -> audio and robot inverse dynamics (DROID, video -> actions).

The released diffusers `Cosmos3OmniModularPipeline` runs every task unchanged; the student only changes what the
transformer reads (`Cosmos3CodecOPSDTransformer`): the condition tokens the codec gazer keeps, with the condition LoRA on them.
Inverse dynamics runs an exact fast path by default (`fast`).
"""

import contextlib
import inspect
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Cosmos3OmniModularPipeline, CosmosActionCondition, UniPCMultistepScheduler
from diffusers.pipelines.cosmos import pipeline_cosmos3_omni
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from ...codec_gazer import COSMOS3, submit_gaze
from ...configs import load
from ...models import Cosmos3CodecOPSDTransformer

CONFIG = load("cosmos3")  # src/codecopsd/configs/cosmos3.yaml
CONTROLS = ("edge", "blur", "depth", "seg")  # the released pipeline packs the controls in this order
TASKS = ("transfer", "i2v", "continuation", "v2a", "id_droid")
FAMILY = {"transfer": "transfer", "i2v": "generation", "continuation": "generation", "v2a": "generation",
          "id_droid": "inverse_dynamics"}
NEGATIVE_PROMPT = json.dumps(json.loads(files("codecopsd.configs").joinpath(CONFIG["negative_prompt"]).read_text()))
VIDEOS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def read_frames(path_or_frames):
    """uint8 RGB frames [T, H, W, 3] (numpy) and the fps (None when unknown) of a video file, a picture file, a list of
    PIL images or an array."""
    if isinstance(path_or_frames, (str, os.PathLike)):
        path = str(path_or_frames)
        if path.lower().endswith(VIDEOS):
            fps = iio.immeta(path, plugin="FFMPEG").get("fps")
            return np.stack(list(iio.imiter(path, plugin="FFMPEG"))), fps
        return np.asarray(Image.open(path).convert("RGB"))[None], None
    if isinstance(path_or_frames, Image.Image):
        return np.asarray(path_or_frames.convert("RGB"))[None], None
    if isinstance(path_or_frames, list):
        return np.stack([np.asarray(f.convert("RGB")) if isinstance(f, Image.Image) else np.asarray(f)
                         for f in path_or_frames]), None
    frames = np.asarray(path_or_frames)
    return (frames[None] if frames.ndim == 3 else frames), None


def fit_frames(frames, height, width, nearest=False):
    """uint8 [T, H, W, 3] -> [T, height, width, 3]: resized to cover (bilinear, antialiased; `nearest` for label maps such
    as segmentation colours), then center-cropped, as the released pipeline fits its inputs."""
    t, h, w, _ = frames.shape
    if (h, w) == (height, width):
        return frames
    scale = max(width / w, height / h)
    size = (math.ceil(scale * h), math.ceil(scale * w))
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    if nearest:
        x = F.interpolate(x, size=size, mode="nearest")
    else:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    top, left = round((size[0] - height) / 2), round((size[1] - width) / 2)
    x = x[:, :, top:top + height, left:left + width].round().clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy()


def default_size(task, h, w):
    """(height, width) of a task's output: the released size of the input's aspect ratio at the task's tier."""
    if task == "id_droid":
        return tuple(CONFIG["id_droid"]["canvas"])
    sizes = CONFIG["sizes"][CONFIG["default_tier"][task]]
    return min(sizes, key=lambda s: abs(math.log((s[1] / s[0]) / (w / h))))


def aspect_ratio(height, width):
    """The framework's aspect-ratio key closest to width / height."""
    keys = {"16,9": 16 / 9, "4,3": 4 / 3, "1,1": 1.0, "3,4": 3 / 4, "9,16": 9 / 16}
    return min(keys, key=lambda k: (keys[k] - width / height) ** 2)


def _sentences(text, num_frames, fps, height, width):
    """`text` + the framework's duration and resolution sentences."""
    text = text.strip()
    if num_frames > 1:
        text = text.rstrip(".") + f". The video is {num_frames / fps:.1f} seconds long and is of {fps:.0f} FPS."
    return (text.strip().rstrip(".") + f". This video is of {height}x{width} resolution.").lstrip(".").strip()


def prompt_text(task, prompt, num_frames, fps, height, width):
    """The prompt as the students were trained (the official Cosmos 3 framework's format): a JSON prompt gets the
    duration / fps / resolution / aspect-ratio fields; a plain caption gets the duration and resolution sentences, except
    for transfer, where it becomes the JSON prompt {"temporal_caption": caption}. Inverse dynamics: the instruction as is
    (the released pipeline builds its action prompt)."""
    fps = int(round(fps))
    if task == "id_droid":
        return prompt
    try:
        obj = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if not isinstance(obj, dict):
        if task != "transfer":
            return _sentences(prompt, num_frames, fps, height, width)
        obj = {"temporal_caption": prompt}
    obj.update({"duration": f"{int(num_frames / fps)}s", "fps": float(fps)} if num_frames > 1 else {})
    obj.update({"resolution": {"H": int(height), "W": int(width)}, "aspect_ratio": aspect_ratio(height, width)})
    return json.dumps(obj)


def negative_text(num_frames, fps, height, width):
    """The official default negative prompt with the sample's duration and resolution sentences (as trained)."""
    return _sentences(NEGATIVE_PROMPT, num_frames, int(round(fps)), height, width)


def droid_canvas(wrist, exterior_1, exterior_2):
    """The DROID three-view canvas [T, 540, 640, 3] the student was trained on: the wrist view (360 x 640) on top, the two
    exterior views (each 180 x 320) side by side below. Inputs uint8 [T, H, W, 3]."""
    def resize(x, h, w):
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    bottom = np.concatenate([resize(exterior_1, 180, 320), resize(exterior_2, 180, 320)], axis=2)
    return np.concatenate([resize(wrist, 360, 640), bottom], axis=1)


@contextlib.contextmanager
def _frozen_properties(*items):
    """Within the block, each (object, property name) returns the value it had on entry, for that object only (other
    instances read the property as usual). The released denoising loop reads the pipeline's execution device and the
    transformer's dtype at every step; each read walks every submodule of the models."""
    patched = []
    try:
        for obj, name in items:
            cls = type(obj)
            descriptor, value = inspect.getattr_static(obj, name), getattr(obj, name)

            def frozen(self, _obj=obj, _value=value, _descriptor=descriptor):
                return _value if self is _obj else _descriptor.__get__(self, type(self))
            patched.append((cls, name, cls.__dict__.get(name)))
            setattr(cls, name, property(frozen))
        yield
    finally:
        for cls, name, own in reversed(patched):
            if own is None:
                delattr(cls, name)
            else:
                setattr(cls, name, own)


@contextlib.contextmanager
def _droid_concat_view_caption():
    """The released DROID caption: the concat-view framing sentence plus the layout of the three views (the framework's
    DROID dataset description, which diffusers' action prompt builder does not add)."""
    templates = pipeline_cosmos3_omni._ACTION_VIEWPOINT_TEMPLATES
    old = templates["concat_view"]
    templates["concat_view"] = old.rstrip() + " " + CONFIG["id_droid"]["view_description"]
    try:
        yield
    finally:
        templates["concat_view"] = old


class Cosmos3CodecOPSDPipeline:
    r"""Cosmos 3 (Nano) for the five CodecOPSD tasks, with a CodecOPSD student condition LoRA.

    The CodecOPSD student is one LoRA on the condition rows of the released transformer, distilled on-policy from the released
    model on only the condition tokens the codec gazer keeps: the control videos of a transfer, the given image / frames
    of image -> video and continuation, the input video of video -> audio and inverse dynamics. Without a LoRA (or with
    `gaze=False` and `lora=False`) this is the released model (the teacher).
    """

    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_cosmos3(cls, path, torch_dtype=torch.bfloat16, device="cuda"):
        """The pipeline from the released Cosmos3-Nano folder (`hf download nvidia/Cosmos3-Nano`), on `device`."""
        pipe = Cosmos3OmniModularPipeline.from_pretrained(path)
        pipe.load_components(names=["video_processor", "text_tokenizer", "vae", "scheduler", "sound_tokenizer"],
                             torch_dtype=torch_dtype)
        transformer = Cosmos3CodecOPSDTransformer.from_pretrained(path, subfolder="transformer", torch_dtype=torch_dtype)
        transformer.requires_grad_(False).eval()
        pipe.update_components(transformer=transformer)
        for name in ("transformer", "vae", "sound_tokenizer"):
            getattr(pipe, name).to(device)
        pipe.disable_safety_checker()  # the released guardrail is optional: `pipeline.pipe.enable_safety_checker()`
        return cls(pipe)

    @property
    def transformer(self):
        return self.pipe.transformer

    @property
    def device(self):
        return self.transformer.device

    def load_condition_lora(self, path_or_repo_id, weight_name=None, dtype=torch.bfloat16):
        """Load a CodecOPSD student LoRA (local .safetensors or `weight_name` in a Hub repo): tensors
        `layers.{i}.{module}.lora_{A,B}.weight`, metadata `lora_alpha` and `gazer` ("codec"). `dtype` of the factors and
        of the LoRA computation, a torch dtype or its name: bfloat16 (default for every task, the base model's precision)
        or float32 (as trained; slower)."""
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        path = path_or_repo_id if os.path.isfile(path_or_repo_id) else hf_hub_download(path_or_repo_id, weight_name)
        with safe_open(path, framework="pt") as f:
            self.condition_lora_metadata = f.metadata()
        self.transformer.load_condition_lora(load_file(path), float(self.condition_lora_metadata["lora_alpha"]), dtype)

    # ------------------------------------------------------------------ inputs

    def prepare(self, task, prompt, *, controls=None, image=None, video=None, num_frames=None, height=None, width=None,
                fps=None, given_latent_frames=None, seed=0, num_inference_steps=None, guidance_scale=None,
                control_guidance=None, negative_prompt=None):
        """The released pipeline's call arguments for one request, and the condition videos the codec gazer reads.

        Args:
            task: "transfer", "i2v", "continuation", "v2a" or "id_droid".
            prompt: the caption (transfer: a caption or a JSON prompt; id_droid: the instruction).
            controls: transfer: {"edge" / "blur" / "depth" / "seg": video file, frames or uint8 [T, H, W, 3]}.
            image: i2v: the first frame.
            video: continuation: the given frames (or a longer clip: its first frames are given); v2a: the input video;
                id_droid: the three-view canvas (`droid_canvas`).
            num_frames: output frames, 4n + 1 (default: the task's, see configs/cosmos3.yaml).
            height, width: output size (default: the released size of the input's aspect ratio).
            fps: output fps (default: the input's, else 24).
            given_latent_frames: continuation: latent frames given (4 (g - 1) + 1 pixel frames; default 2).
            negative_prompt: the negative prompt (default: the framework's default). For image -> video, continuation
                and video -> audio the duration and resolution sentences are added, as the framework does (e.g. a
                Physics-IQ case's own negative prompt); transfer takes it as given.

        Returns:
            dict: "kwargs" (the released call), "conditions" (uint8 [T, H, W, 3] per clean block, in packing order),
            "kinds" (codec gazer kind of each), "blocks" (number of latent frames of each), and the resolved settings.
        """
        if task not in TASKS:
            raise ValueError(f"task {task!r} not in {TASKS}")
        family = FAMILY[task]
        steps = num_inference_steps or CONFIG["num_inference_steps"][family]
        guidance = CONFIG["guidance_scale"][family] if guidance_scale is None else guidance_scale
        kinds = CONFIG["gaze_kinds"]
        if task == "transfer":
            if not controls or set(controls) - set(CONTROLS):
                raise ValueError(f"transfer needs controls among {CONTROLS}, got {list(controls or {})}")
            hints = [h for h in CONTROLS if h in controls]
            read = {h: read_frames(controls[h]) for h in hints}
            src = read[hints[0]][0]
            fps = fps or read[hints[0]][1] or 24.0
            height, width = (height, width) if height else default_size(task, *src.shape[1:3])
            num_frames = num_frames or min(CONFIG["num_frames"]["transfer"], min(len(r[0]) for r in read.values()))
            num_frames = (num_frames - 1) // 4 * 4 + 1
            conditions = [fit_frames(read[h][0][:num_frames], height, width, nearest=h == "seg") for h in hints]
            if any(len(c) < num_frames for c in conditions):
                raise ValueError(f"every control needs {num_frames} frames")
            key = hints[0] if len(hints) == 1 else "multi"
            cg = CONFIG["control_guidance"][key] if control_guidance is None else control_guidance
            kwargs = dict(control_videos={h: [Image.fromarray(f) for f in c] for h, c in zip(hints, conditions)},
                          control_guidance=1.0 + (cg - 1.0) / guidance if guidance > 1.0 else cg,
                          num_video_frames_per_chunk=num_frames, num_conditional_frames=0)
            gaze_kinds = [kinds[h] for h in hints]
        elif task == "id_droid":
            c = CONFIG["id_droid"]
            frames, _ = read_frames(video)
            height, width = c["canvas"]
            if frames.shape[1:3] != (height, width) or len(frames) < c["chunk"] + 1:
                raise ValueError(f"id_droid needs a {height} x {width} canvas of {c['chunk'] + 1} frames (droid_canvas)")
            conditions, num_frames, fps = [frames[: c["chunk"] + 1]], c["chunk"] + 1, c["fps"]
            kwargs = dict(action=CosmosActionCondition(
                mode="inverse_dynamics", chunk_size=c["chunk"], domain_name="droid_lerobot", resolution_tier=480,
                video=[Image.fromarray(f) for f in conditions[0]], view_point="concat_view"), use_system_prompt=False)
            gaze_kinds = [kinds["video"]]
        else:
            frames, src_fps = read_frames(image if task == "i2v" else video)
            fps = fps or src_fps or 24.0
            height, width = (height, width) if height else default_size(task, *frames.shape[1:3])
            if task == "i2v":
                num_frames = num_frames or CONFIG["num_frames"]["i2v"]
                given = fit_frames(frames[:1], height, width)
                kwargs = dict(image=Image.fromarray(given[0]))
                blocks = 1
            elif task == "continuation":
                g = given_latent_frames or CONFIG["continuation"]["given_latent_frames"]
                num_frames = num_frames or CONFIG["num_frames"]["continuation"]
                given = fit_frames(frames[: 4 * (g - 1) + 1], height, width)
                if len(given) < 4 * (g - 1) + 1 or g >= (num_frames - 1) // 4 + 1:
                    raise ValueError(f"continuation of {g} latent frames needs {4 * (g - 1) + 1} given frames and "
                                     f"more than {g} latent frames out")
                kwargs = dict(video=[Image.fromarray(f) for f in given], condition_frame_indexes_vision=list(range(g)),
                              condition_video_keep="first")
                blocks = g
            else:  # v2a
                num_frames = (min(num_frames or len(frames), len(frames)) - 1) // 4 * 4 + 1
                given = fit_frames(frames[:num_frames], height, width)
                blocks = (num_frames - 1) // 4 + 1
                kwargs = dict(video=[Image.fromarray(f) for f in given], condition_frame_indexes_vision=list(range(blocks)),
                              enable_sound=True, output=["videos", "sound", "sampling_rate"])
            conditions = [given]
            kwargs.update(add_resolution_template=False, add_duration_template=False)
            gaze_kinds = [kinds["video"]]
        if task != "id_droid":  # inverse dynamics: the pipeline takes the size and length from the action video
            if negative_prompt is None:
                negative_prompt = negative_text(num_frames, fps, height, width)
            elif family == "generation":  # the framework adds the duration and resolution sentences to a given one
                negative_prompt = _sentences(negative_prompt, num_frames, int(round(fps)), height, width)
            kwargs.update(negative_prompt=negative_prompt, height=height, width=width, num_frames=num_frames)
        kwargs.update(prompt=prompt_text(task, prompt, num_frames, fps, height, width),
                      fps=float(fps), num_inference_steps=steps, guidance_scale=guidance,
                      generator=torch.Generator(self.device).manual_seed(seed))
        kwargs.setdefault("output", "action" if task == "id_droid" else "videos")
        return dict(task=task, kwargs=kwargs, conditions=conditions, kinds=gaze_kinds, height=height, width=width,
                    num_frames=num_frames, fps=float(fps), num_inference_steps=steps, guidance_scale=guidance, seed=seed)

    def _scheduler(self):
        """The released UniPC with the framework's shift (10 at every resolution tier), Karras sigmas off."""
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=CONFIG["flow_shift"], use_karras_sigmas=False)

    def submit_gazes(self, pool, request, keep_last_given=None):
        """Futures of the codec gazer masks of a `prepare`d request (one per clean block, on the token grid), computed in
        `pool`'s threads. continuation with `keep_last_given`: the last given latent frame is kept whole."""
        futures = [submit_gaze(pool, torch.from_numpy(c).permute(0, 3, 1, 2), kind, COSMOS3)
                   for c, kind in zip(request["conditions"], request["kinds"])]
        keep_last = CONFIG["continuation"]["keep_last_given"] if keep_last_given is None else keep_last_given

        def masks():
            out = [f.result().keep for f in futures]
            if request["task"] == "continuation" and keep_last:
                out[0][-1] = True
            return out
        return masks

    # ------------------------------------------------------------------ inference

    @torch.no_grad()
    def __call__(self, task, prompt, *, gaze=None, lora=True, keep_last_given=None, fast=None, **inputs):
        """Run one request (arguments: `prepare`).

        Args:
            gaze: give the student only the condition tokens the codec gazer keeps (default: with a loaded LoRA).
            lora: use the loaded condition LoRA (False: the released model).
            keep_last_given: continuation: keep the last given latent frame whole (default: configs/cosmos3.yaml).
            fast: inverse dynamics: the exact fast path (default True: bit-identical actions, teacher and student
                alike; `Cosmos3FastInference`, CUDA graphs kept for the next call of the same shapes, and the video
                decode, whose result an action call discards, skipped). False: the released forward.

        Returns:
            dict: the released pipeline's outputs ("videos" [1, 3, T, H, W] in [0, 1], "sound" + "sampling_rate" for
            v2a, "action" for id_droid), "masks" (the gazer masks, when gazed), "condition_tokens" / "kept_tokens",
            "seconds".
        """
        has_lora = hasattr(self, "condition_lora_metadata")
        gaze = has_lora if gaze is None else gaze
        fast = task == "id_droid" if fast is None else fast
        if fast and task != "id_droid":
            raise NotImplementedError("the fast path runs inverse dynamics (id_droid) only")
        request = self.prepare(task, prompt, **inputs)
        if fast and request["guidance_scale"] != 1.0:
            raise NotImplementedError("the fast path runs one transformer pass per step (guidance_scale 1)")
        self._scheduler()
        t0 = time.perf_counter()
        tr = self.transformer
        tr.set_condition_lora_enabled(lora and has_lora)
        with ThreadPoolExecutor(max(1, len(request["conditions"]))) as pool:
            masks = self.submit_gazes(pool, request, keep_last_given) if gaze else None
            tr.condition_masks = masks
            try:
                with contextlib.ExitStack() as stack:
                    if task == "id_droid":
                        stack.enter_context(_droid_concat_view_caption())
                    if fast:
                        stack.enter_context(tr.fast_inference())
                        stack.enter_context(_frozen_properties((self.pipe, "_execution_device"), (tr, "dtype"),
                                                               (tr, "device")))
                    output_type = "pt" if task != "id_droid" else "latent" if fast else None
                    extra = {} if output_type is None else {"output_type": output_type}
                    out = self.pipe(**request["kwargs"], **extra)
                masks = tr.condition_masks if gaze else None
            finally:
                tr.condition_masks = None
                tr.set_condition_lora_enabled(True)
        out = out if isinstance(out, dict) else {request["kwargs"]["output"]: out}
        result = {k: out.get(k) for k in ("videos", "sound", "sampling_rate", "action")}
        dense = sum(len(range(0, len(c), 4)) * math.ceil(c.shape[1] / 32) * math.ceil(c.shape[2] / 32)
                    for c in request["conditions"])
        result.update(masks=masks, condition_tokens=dense,
                      kept_tokens=dense if masks is None else int(sum(int(m.sum()) for m in masks)),
                      seconds=time.perf_counter() - t0, request={k: v for k, v in request.items()
                                                                 if k not in ("kwargs", "conditions")})
        return result
