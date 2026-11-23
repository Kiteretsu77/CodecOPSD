"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The checks of a request's inputs before it waits for the GPU: output size, seed, the <Picture i> / <Video k> of the
prompt, and the reference files (`Refused` says what is wrong under the outputs).
"""

import re
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageOps

from .media import VIDEO_FILES
from .progress import Refused


def parse_size(text, low, high_side, max_pixels, multiple=1):
    """(height, width) of the Output box: '1024 x 768', '1024x768', '1024×768' or '1024, 768'. Each side within
    [low, high_side], at most `max_pixels` and 4:1, rounded to the nearest `multiple`."""
    m = re.fullmatch(r"\s*(\d+)\s*[x×X*,]\s*(\d+)\s*", str(text or ""))
    if not m:
        raise Refused(f"Output size: write it as height x width, for example 1024 x 768 (got {text!r}).")
    height, width = (max(multiple, int(v / multiple + 0.5) * multiple) for v in (int(m[1]), int(m[2])))
    if min(height, width) < low or max(height, width) > high_side or height * width > max_pixels \
            or max(height, width) > 4 * min(height, width):
        raise Refused(f"Output size {m[1]} x {m[2]}: each side from {low} to {high_side}, at most {max_pixels:,} "
                       "pixels, and at most 4:1.")
    return height, width


def parse_seed(seed):
    try:
        value = float(seed)
    except (TypeError, ValueError):
        value = None
    if value is None or not value.is_integer() or not 0 <= value < 2 ** 32:
        raise Refused(f"Seed: a whole number from 0 to {2 ** 32 - 1} (got {seed}).")
    return int(value)


def check_tags(prompt, pictures, videos=0, video_tab=False):
    """The <Picture i> / <Video k> the prompt names must be among the references (numbered per kind, in order)."""
    for kind, i in re.findall(r"<\s*(picture|video)\s*(\d+)\s*>", prompt, flags=re.I):
        kind, i = kind.capitalize(), int(i)
        n = pictures if kind == "Picture" else videos
        if not 1 <= i <= n:
            given = f"{n} {kind.lower()}{'' if n == 1 else 's'} {'is' if n == 1 else 'are'} given"
            hint = "" if video_tab or kind == "Picture" else " (the image model takes pictures; videos go in the Video tab)"
            raise Refused(f"The prompt names <{kind} {i}>, but {given}{hint}. References are numbered per kind, in "
                           "the order of the list.")


def open_picture(path, i):
    try:
        image = Image.open(path)
        image.load()
    except Exception:
        raise Refused(f"Reference {i} ({Path(path).name}) is not a picture this demo can read: use PNG, JPEG, WebP "
                       "or BMP.") from None
    if min(image.size) < 32:
        raise Refused(f"Reference {i} ({Path(path).name}) is {image.width} x {image.height}; a picture needs at least "
                       "32 pixels on each side.")
    return image


def open_reference(path, i, frames):
    """Reference `i` of the video tab: a picture (any format PIL reads, upright by its EXIF orientation) or a video
    (decoded only as far as the pipeline reads it), checked before the request waits for the GPU."""
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference
    from codecopsd.pipelines.minimax_h3_ref2va import load_reference
    name = Path(path).name
    if not name.lower().endswith(VIDEO_FILES):
        try:
            image = Image.open(path)
            image.load()
        except Exception:
            image = None
        if image is not None:
            if min(image.size) < 32:
                raise Refused(f"Reference {i} ({name}) is {image.width} x {image.height}; a picture needs at least 32 "
                               "pixels on each side.")
            return MiniMaxH3ImageReference(image=ImageOps.exif_transpose(image).convert("RGB"))
    try:
        ref = load_reference(path, num_frames=frames)
    except Exception:
        raise Refused(f"Reference {i} ({name}) is neither a picture nor a video this demo can read.") from None
    if ref.kind == "video" and int(len(ref.frames) * 24 / ref.fps + 0.5) < 22:
        raise Refused(f"Reference {i} ({name}) is {len(ref.frames) / ref.fps:.2f} s long; a reference video needs at "
                       "least 22 frames at 24 fps (0.92 s).")
    return ref


def source_of(path, ref):
    """What a reference file is before the model reads it: "H x W" of a picture, "S s at F frames per second, H x W"
    of a video."""
    if ref.kind == "image":
        width, height = ref.image.size
        return f"{height} x {width}"
    try:
        seconds = float(iio.immeta(path, plugin="FFMPEG").get("duration") or 0)
    except Exception:
        seconds = 0
    length = f"{seconds:.1f} s at " if seconds else ""
    return f"{length}{float(ref.fps):g} frames per second, {ref.frames.shape[1]} x {ref.frames.shape[2]}"
