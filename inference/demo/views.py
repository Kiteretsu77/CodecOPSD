"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The references as the model read them, drawn under the results: kept tokens in colour, dropped tokens dimmed.
"""

import html
import time
import urllib.parse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

from .media import light_view
from .state import OUT, STATE


# Under the results, every reference as the model reads it: a picture with its dropped tokens dimmed; a video twice,
# side by side at the frame rate and length the model reads (every frame, 24 frames per second): as it is, and with its
# dropped tokens dimmed, plus its frames, size and kept tokens.

VIEW_SIDE = 768  # long side of the reference views on the page, in pixels


def view_size(height, width, side=VIEW_SIDE):
    """(height, width) of a view: at most `side` on the long side, even sides (H.264)."""
    scale = min(1.0, side / max(height, width))
    return max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2)


def dim_dropped(rgb, mask, size, resample=Image.BILINEAR):
    """uint8 [H, W, 3] -> uint8 [h, w, 3] at size (h, w), with the dropped tokens dimmed and greyed. mask: bool [th, tw]
    on the 32 x 32 pixel token grid of the full picture, counted from the top left (the gazer pads the last row /
    column of a picture whose side is not a multiple of 32)."""
    full_h, full_w = rgb.shape[:2]
    h, w = size
    image = np.asarray(Image.fromarray(rgb).resize((w, h), resample), dtype=np.float32)
    rows = np.minimum((np.arange(h) * full_h // h) // 32, mask.shape[0] - 1)
    cols = np.minimum((np.arange(w) * full_w // w) // 32, mask.shape[1] - 1)
    keep = np.asarray(mask, dtype=bool)[rows][:, cols]
    dimmed = 0.3 * (0.5 * image + 0.5 * image.mean(axis=2, keepdims=True)) + 20
    return np.where(keep[..., None], image, dimmed).clip(0, 255).astype(np.uint8)


def write_video(path, frames, fps):
    iio.imwrite(path, frames, fps=fps, plugin="FFMPEG", codec="libx264", macro_block_size=1,
                output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-crf", "18", "-preset", "veryfast"])


def read_references(pipe, name):
    """CPU copies of the last request's references as the model read them and their keep-masks, read while the request
    still holds the GPU: [(kind, picture uint8 [H, W, 3] or frames uint8 [T, H, W, 3] at 24 fps, mask, source)]."""
    enc = getattr(pipe, "_last_encode", None)
    if not enc or not enc.get("masks"):
        return []
    sources = STATE.get("sources") or []
    if name == "minimax_h3_image":
        return [("picture", picture, m[0].cpu().numpy(), sources[i] if i < len(sources) else None)
                for i, (picture, m) in enumerate(zip(gazed_pictures(pipe, STATE["images"]), enc["masks"], strict=True))]
    out = []
    for i, (ref, mask) in enumerate(zip(enc["references"], enc["masks"], strict=True)):
        m = mask.cpu().numpy()
        source = sources[i] if i < len(sources) else None
        if ref.kind == "image":
            out.append(("picture", np.asarray(ref.image.convert("RGB")), m[0], source))
        else:
            out.append(("video", np.asarray(ref.frames), m, source))
    return out


def gazed_pictures(pipe, images):
    """The pixels the codec gazer reads of each reference picture of the image model: the VAE's copy, cut at the top
    left to whole tokens (32 px), uint8 [H, W, 3]."""
    out = []
    for picture in pipe.prepare_references(images)[0]:
        rgb = np.asarray(picture.convert("RGB"))
        out.append(np.ascontiguousarray(rgb[: rgb.shape[0] // 32 * 32, : rgb.shape[1] // 32 * 32]))
    return out


def draw_references(seen):
    """The views of `read_references` written to outputs/app, and the kept share of all reference tokens. Returns
    ([dict(kind, number, tokens, natural, frames, fps, height, width, token_frames, grid, kept, total, source)], share)."""
    if not seen:
        return [], None
    from codecopsd.codec_gazer.codec import frame_groups
    from codecopsd.pipelines.minimax_h3_ref2va.pipeline_minimax_h3_ref2va import MINIMAX_H3_FPS
    stamp, count, refs, kept, total = time.time_ns(), {"picture": 0, "video": 0}, [], 0, 0
    for kind, pixels, mask, source in seen:
        count[kind] += 1
        mask = np.asarray(mask, dtype=bool)
        kept, total = kept + int(mask.sum()), total + mask.size
        entry = dict(kind=kind, number=count[kind], kept=int(mask.sum()), total=int(mask.size), source=source,
                     grid=list(mask.shape[-2:]))
        base = OUT / f"ref_{stamp}_{kind}{count[kind]}"
        if kind == "picture":
            entry.update(height=pixels.shape[0], width=pixels.shape[1])
            Image.fromarray(dim_dropped(pixels, mask, view_size(*pixels.shape[:2]), Image.LANCZOS)).save(
                f"{base}_tokens.jpg", quality=90)
            entry["tokens"] = f"{base}_tokens.jpg"
        else:
            frames, (h, w) = pixels, view_size(*pixels.shape[1:3])
            starts = frame_groups(len(frames))
            group = np.searchsorted(np.asarray(starts), np.arange(len(frames)), side="right") - 1  # token frame of each frame
            natural = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in frames])
            dimmed = np.stack([dim_dropped(f, mask[g], (h, w)) for f, g in zip(frames, group)])
            write_video(f"{base}_natural.mp4", natural, MINIMAX_H3_FPS)
            write_video(f"{base}_tokens.mp4", dimmed, MINIMAX_H3_FPS)
            entry.update(natural=f"{base}_natural.mp4", tokens=f"{base}_tokens.mp4", frames=len(frames),
                         fps=MINIMAX_H3_FPS, height=frames.shape[1], width=frames.shape[2], token_frames=len(mask))
        refs.append(entry)
    return refs, kept / max(total, 1)


def file_url(path):
    """The address of a file of outputs/app or assets/examples on this demo's server (both are allowed paths)."""
    return "gradio_api/file=" + urllib.parse.quote(str(Path(path).resolve()))


def drop_note(total, kept):
    """'(−69%)': the share of the tokens dropped, after a token count."""
    return f'<span class="drop-pct">(−{1 - kept / max(total, 1):.0%})</span>'


def refs_html(refs, setting=None):
    """The references as the model read them (pictures in a grid; each video as read | kept tokens, played together),
    one short line each, then the setting in small type."""
    note = f'<p class="setting-note">{html.escape(setting, quote=False)}</p>' if setting else ""
    if not refs:
        return note
    pictures, videos = [], []
    for r in refs:
        if r["kind"] == "legacy":  # stored before the views had their numbers: the view and its caption
            media = (f'<video src="{file_url(r["tokens"])}" autoplay muted loop playsinline controls></video>'
                     if str(r["tokens"]).endswith(".mp4")
                     else f'<img src="{file_url(r["tokens"])}" alt="reference" title="Click to enlarge">')
            pictures.append(f'<figure class="ref-card">{media}<figcaption>{html.escape(r["caption"])}</figcaption></figure>')
            continue
        tokens = f"{r['total']:,} → {r['kept']:,} tokens {drop_note(r['total'], r['kept'])}"
        if r["kind"] == "picture":
            pictures.append(
                f'<figure class="ref-card"><img src="{file_url(r["tokens"])}" alt="Picture {r["number"]}" '
                f'title="Click to enlarge">'
                f'<figcaption><b>Picture {r["number"]}</b>: {tokens}</figcaption></figure>')
            continue
        fps = r["fps"]
        video = lambda path, label: (f'<figure><video src="{file_url(light_view(path))}" autoplay muted loop playsinline controls '
                                     f'preload="auto"></video><figcaption>{label}</figcaption></figure>')
        name = f"<b>Video {r['number']}</b>: {r['frames']} frames, {fps:g} fps"
        videos.append(
            f'<div class="ref-video ref-sync" data-fps="{fps:g}" data-frames="{r["frames"]}">'
            f'<div class="ref-pair">{video(r["natural"], name)}'
            f'{video(r["tokens"], tokens)}</div>'
            f'<div class="ref-lines"><span class="frame-readout">Frame 1 of {r["frames"]}</span>'
            f'<button class="frame-step" data-step="-1">‹ frame</button>'
            f'<button class="frame-step" data-step="1">frame ›</button>'
            f'<button class="ref-zoom" aria-label="enlarge this reference">⤢ Enlarge</button></div></div>')
    head = '<div class="refs-title">Kept tokens in colour, dropped tokens dimmed (click a view to enlarge it)</div>'
    grid = f'<div class="ref-grid">{"".join(pictures)}</div>' if pictures else ""
    return f'<div class="refs">{head}{grid}{"".join(videos)}</div>{note}'


def stored_refs(meta, d):
    """The reference views of an example's stored Ours result, with their files in its cache folder (results stored
    before the views had their numbers: their view and caption, until `upgrade_stored_refs` redraws them)."""
    if "refs" not in meta:
        return [dict(kind="legacy", tokens=str(d / f), caption=c) for f, c in meta.get("cells", [])]
    return [{**r, **{k: str(d / r[k]) for k in ("tokens", "natural") if r.get(k)}} for r in meta["refs"]]
