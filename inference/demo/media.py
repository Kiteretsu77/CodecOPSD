"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The files of the page: light copies of the bundled media (quick to load on a remote link; a run reads the originals),
the captions of the reference list and the JPEG copy of an image result.
"""

import os
import threading
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageOps

from .state import CACHE, EXAMPLES


VIDEO_FILES = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".3gp", ".ts")


def display_copy(image, path, size=1536):
    """The JPEG the page shows: at most `size` px on the long side (quick to load on a remote link); the PNG keeps
    the full result."""
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.LANCZOS)
    image.save(path, quality=92)


def add_files(current, new):
    """The references after an upload: the ones already listed, then the new files, each with its size."""
    return captioned(paths_of(current) + [getattr(f, "name", f) for f in new or []])


_CAPTIONS = {}
# The page shows light copies of the bundled media (the references in the input gallery, the reference views and the
# results stored for the examples), so that a click on an example loads a few MB instead of tens; a run reads the
# originals (`input_paths`), so results and their keys do not change.
PREVIEW_SUFFIXES = (".preview.jpg", ".preview.mp4")
_LIGHT_LOCK = threading.Lock()


def _ffmpeg(src, dst, long_side=None, crf=None):
    """H.264 copy of a video with its index at the start (+faststart): re-encoded with the long side at most
    `long_side` and `crf`, or the same stream remuxed when `long_side` is None."""
    import subprocess
    import imageio_ffmpeg
    if long_side is None:
        codec = ["-c", "copy"]
    else:
        scale = f"scale='if(gt(iw,ih),min({long_side},iw),-2)':'if(gt(iw,ih),-2,min({long_side},ih))'"
        codec = ["-vf", scale, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-i", str(src), *codec, "-an", "-movflags",
                    "+faststart", str(dst)], check=True)


def _light(src, out, make):
    """`out` made from `src` by `make(src, tmp)` (kept on disk, made again when `src` is newer, e.g. a result computed
    again); `src` itself when that fails."""
    fresh = lambda: out.exists() and out.stat().st_mtime >= Path(src).stat().st_mtime
    if fresh():
        return str(out)
    with _LIGHT_LOCK:
        if not fresh():
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".part" + out.suffix)
            try:
                make(src, tmp)
                tmp.rename(out)
            except Exception as e:
                tmp.unlink(missing_ok=True)
                print(f"[light] {src}: {e!r}", flush=True)
                return str(src)
    return str(out)


def _picture_preview(src, dst):
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1024, 1024), Image.LANCZOS)
        image.save(dst, "JPEG", quality=85)


def preview(path):
    """The light copy of a bundled reference (assets/examples/refs) for the input gallery: pictures as JPEG with the
    long side at most 1024, videos as H.264 with the long side at most 640; any other path as it is."""
    p = Path(path)
    if not p.is_file() or p.parent.resolve() != (EXAMPLES / "refs").resolve():
        return str(path)
    video = p.suffix.lower() in VIDEO_FILES
    out = CACHE / "previews" / (p.name + PREVIEW_SUFFIXES[video])
    return _light(p, out, (lambda a, b: _ffmpeg(a, b, 640, 28)) if video else _picture_preview)


def original(path):
    """The bundled reference a preview stands for (found by its file name: Gradio copies the files it shows and keeps
    their names), else `path`."""
    name = Path(str(path)).name
    for suffix in PREVIEW_SUFFIXES:
        if name.endswith(suffix) and (EXAMPLES / "refs" / name[: -len(suffix)]).is_file():
            return str(EXAMPLES / "refs" / name[: -len(suffix)])
    return path


def input_paths(files):
    """The files a run reads: the gallery's paths with every preview replaced by its original."""
    return [original(p) for p in paths_of(files)]


def light_view(path):
    """A stored reference view (video read by the model | its kept tokens) re-encoded for the page: long side at most
    832, crf 26. Other files as they are."""
    p = Path(path)
    if p.suffix != ".mp4" or not p.is_file() or CACHE.resolve() not in p.resolve().parents:
        return str(path)
    return _light(p, p.with_name(p.stem + ".light.mp4"), lambda a, b: _ffmpeg(a, b, 832, 26))


def fast_result(path):
    """A stored result video remuxed with its index first, so the page starts playing it at once (same frames)."""
    p = Path(str(path)) if path else None
    if p is None or p.suffix != ".mp4" or not p.is_file() or CACHE.resolve() not in p.resolve().parents:
        return path
    return _light(p, p.with_name(p.stem + ".fast.mp4"), _ffmpeg)


def _caption_size(src):
    """' · 480 x 832 · 5.0 s' of a video, ' · 2048 x 3072' of a picture (upright); '' when the file cannot be read."""
    try:
        if str(src).lower().endswith(VIDEO_FILES):
            meta = iio.immeta(src, plugin="FFMPEG")
            (w, h), seconds = meta["size"], meta.get("duration")
            return f" · {h} x {w}" + (f" · {seconds:.1f} s" if seconds else "")
        with Image.open(src) as image:
            w, h = image.size
            if image.getexif().get(0x0112) in (5, 6, 7, 8):  # shown upright, as the model reads it
                w, h = h, w
        return f" · {h} x {w}"
    except Exception:
        return ""


def captioned(paths):
    """The reference list with captions: "Picture 1 · 2048 x 3072", "Video 1 · 480 x 832 · 5.0 s" (numbered per kind,
    height x width as the model reads it). Files not captioned before are read at the same time, one thread each."""
    from codecopsd.pipelines.minimax_h3_ref2va import map_references
    srcs = [original(path) for path in paths]  # a preview is captioned with its original's size
    keys = [(str(src), os.path.getmtime(src) if os.path.exists(src) else 0) for src in srcs]
    new = {key: src for key, src in zip(keys, srcs) if key not in _CAPTIONS}
    _CAPTIONS.update(zip(new, map_references(_caption_size, list(new.values()))))
    out, count = [], {"Picture": 0, "Video": 0}
    for path, src, key in zip(paths, srcs, keys):
        kind = "Video" if str(src).lower().endswith(VIDEO_FILES) else "Picture"
        count[kind] += 1
        out.append((path, f"{kind} {count[kind]}{_CAPTIONS[key]}"))
    return out


def paths_of(files):
    """File paths of a Gallery value ((path, caption) pairs) or a File list (paths)."""
    return [f[0] if isinstance(f, (list, tuple)) else f for f in files or []]


def thumb_video(folder, make=False):
    """The moving thumbnail of a stored example (its tile in the example gallery): its stored result video (Ours, else
    the Teacher) at most 640 px on the long side (360 px short edge at 16:9), silent, crf 30, about 0.2 MB, kept next to
    it as thumb.preview.mp4 (made again on disk, never packed in the example archives). `make`: make it when it is
    missing or older than the result. None when the example has no stored video or the copy is not made yet."""
    folder = Path(folder)
    src = next((folder / f"{who}.mp4" for who in ("student", "released") if (folder / f"{who}.mp4").is_file()), None)
    if src is None:
        return None
    out = folder / "thumb.preview.mp4"
    if make:
        _light(src, out, lambda a, b: _ffmpeg(a, b, 640, 30))
    return str(out) if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime else None
