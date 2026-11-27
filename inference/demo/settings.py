"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The codec pruning setting of the Settings panel: its choices, the checked setting of a request (`parse_codec`), the
release defaults and the values of a bundled example from its command-line flags (`example_codec`).
"""

import gradio as gr

from codecopsd.codec_gazer.codec import SETTINGS as _GAZER

from ..examples import flag_values, flag_values_n
from .progress import Refused


# What the Settings panel controls for Ours (the Teacher always reads every token). One request's setting is, in the
# order of the panel's inputs (and of `request_key`, so the stored results of an example belong to its exact setting):
#   video page:  strategy, video_drop, video_motion, video_residual, picture_motion, picture_residual, vlm, vlm_scale
#   image page:  picture_motion, picture_residual, vlm, vlm_scale
# A bundled example sets them with its command-line flags (example.json "args": --video-drop, --video-thresholds,
# --picture-thresholds, --vlm-drop, --vlm-scale; `example_codec`), so the demo and inference/generate.py run it alike.
# The thresholds' scaling with the size (motion with sqrt(area), picture residual with the short side) is applied
# inside codec pruning and is not shown here.

# Strategy of the reference videos (the pictures always keep the codec's threshold decision):
#   default   = ours: priority drop of 70 % at 768p and above, 66 % below 768p (`video_drop_default`), or the threshold
#               decision when it drops more (codec_gazer.yaml select: max);
#               every threshold at its config value (the panel's fields are hidden and not read)
#   priority  = priority drop of exactly the slider's share (select: exact); thresholds at their config values
#   threshold = the threshold decision alone, with the panel's thresholds (videos and pictures)
# Every bundled example fixes priority or threshold (whichever its default run applied); a user's own references start
# at default.
STRATEGIES = [("Default (recommended)", "default"), ("Priority drop", "priority"), ("Threshold", "threshold")]
STRATEGY_NOTES = {
    "default": "Priority drop of 66 % below 768p and 70 % at 768p, or the threshold decision when it drops more.",
    "priority": "Each reference video drops exactly this share, its least important tokens first.",
    "threshold": "A token is kept when its motion or its residual is above the threshold.",
}
IMAGE_STRATEGIES = [("Default (recommended)", "default"), ("Threshold", "threshold")]
VLM_MODES = [("Threshold x scale (as trained)", "threshold"), ("The VAE tokens' mask (DiT tokens)", "dit"),
             ("Every token (no VLM drop)", "off")]
VLM_SCALE = 0.25  # codec_gazer.yaml vlm_drop.picture_threshold_scale
VIDEO_DROP = 0.70  # codec_gazer.yaml priority_drop.video_default, the default at 768p and above
VIDEO_DROP_LOW = 0.66  # the default below 768p (the page's 480p): fewer tokens per frame, so a little more is kept
THRESHOLDS = {form: (float(_GAZER[form]["motion_px"]), float(_GAZER[form]["residual"])) for form in ("video", "image")}


def video_drop_default(size=None):
    """The default priority drop of the reference videos for an output (height, width): 70 % at 768p and above,
    66 % below (no size: 70 %, the config's value)."""
    return VIDEO_DROP if size is None or min(map(int, size)) >= 768 else VIDEO_DROP_LOW


def _positive(value, name, high):
    """A number in (0, high]; the page lets only digits and a point into these fields (HEAD), this checks the rest."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = -1.0
    if not 0 < value <= high:  # also refuses nan and inf
        raise Refused(f"{name}: a number above 0 and at most {high:g}.")
    return value


def parse_codec(strategy, video_drop, video_motion, video_residual, picture_motion, picture_residual, vlm, vlm_scale,
                video=True, size=None):
    """The checked codec pruning setting of a request: dict(video_drop=pipeline argument, overrides=gaze_overrides,
    vlm=mode, vlm_scale=float, text=setting line). `video` False: the image page (its strategy is default or
    threshold; the video arguments are not read). `size`: the output (height, width); it sets the default's drop
    (`video_drop_default`)."""
    if strategy not in [v for _, v in (STRATEGIES if video else IMAGE_STRATEGIES)]:
        raise Refused(f"Codec pruning: one of {', '.join(v for _, v in STRATEGIES)} (got {strategy!r}).")
    if vlm not in [v for _, v in VLM_MODES]:
        raise Refused(f"VLM tokens: one of {', '.join(v for _, v in VLM_MODES)} (got {vlm!r}).")
    if strategy == "default":  # every setting from codec_gazer.yaml, the text encoder's too (its fields are hidden)
        vlm = "threshold"
    scale = _positive(vlm_scale, "VLM threshold scale", 1.0) if vlm == "threshold" and strategy != "default" else VLM_SCALE
    out = dict(vlm=vlm, vlm_scale=scale, overrides={}, video_drop=None)
    text = []
    if strategy == "threshold":  # the panel's thresholds, for every reference
        pm = _positive(picture_motion, "Pictures: motion threshold", 64)
        pr = _positive(picture_residual, "Pictures: residual threshold", 64)
        out["overrides"]["image"] = {"motion_px": pm, "residual": pr}
        if video:
            vm = _positive(video_motion, "Videos: motion threshold", 64)
            vr = _positive(video_residual, "Videos: residual threshold", 64)
            out["overrides"]["video"] = {"motion_px": vm, "residual": vr}
            text.append(f"reference videos: threshold (motion {vm:g} px, residual {vr:g})")
        text.append(f"reference pictures: threshold (motion {pm:g} px, residual {pr:g})")
    elif strategy == "priority":  # exactly the share, even when the threshold decision would drop more
        out["video_drop"] = share = _positive(video_drop, "Drop share", 0.95)
        out["overrides"]["video"] = {"select": "exact"}
        text += [f"reference videos: priority drop of {share:.0%}", "reference pictures: threshold"]
    else:  # default: everything from codec_gazer.yaml, the video drop by the output size (`video_drop_default`)
        if video:
            out["video_drop"] = share = video_drop_default(size)
            text.append(f"reference videos: default (priority drop of {share:.0%}, or the threshold decision "
                        "when it drops more)")
        text.append("reference pictures: threshold")
    out["overrides"] = out["overrides"] or None
    text.append(vlm_setting(vlm, scale))
    out["text"] = "; ".join(text)
    return out


def vlm_setting(vlm, scale):
    """The text encoder (VLM) part of the setting line."""
    return {"dit": "VLM tokens: the VAE tokens' mask", "off": "VLM tokens: every token"}.get(
        vlm, f"VLM tokens: the picture thresholds x {float(scale):g}")


def example_codec(spec, video=True):
    """The Settings panel values of an example, from its command-line flags (as inference/cli/minimax_h3_ref2va.py reads
    them): video page (strategy, video_drop, video_motion, video_residual, picture_motion, picture_residual, vlm,
    vlm_scale); image page (strategy, picture_motion, picture_residual, vlm, vlm_scale). `--video-drop threshold` =
    threshold, `--video-drop S --video-drop-select exact` = priority S, else default; an image example is threshold."""
    a = spec.get("args", [])
    drop = str((flag_values(a, "--video-drop") or [VIDEO_DROP])[0])
    exact = (flag_values(a, "--video-drop-select") or ["max"])[0] == "exact"
    strategy = "threshold" if drop == "threshold" else "priority" if exact else "default"
    share = float(drop) if strategy == "priority" else VIDEO_DROP
    vm, vr = map(float, flag_values_n(a, "--video-thresholds", 2) or THRESHOLDS["video"])
    pm, pr = map(float, flag_values_n(a, "--picture-thresholds", 2) or THRESHOLDS["image"])
    vlm = {"dit": "dit", "off": "off"}.get((flag_values(a, "--vlm-drop") or ["auto"])[0], "threshold")
    scale = float((flag_values(a, "--vlm-scale") or [VLM_SCALE])[0])
    return (strategy, share, vm, vr, pm, pr, vlm, scale) if video else ("threshold", pm, pr, vlm, scale)


def codec_defaults(video):
    """The Settings panel's codec pruning values back at the release defaults (clear / upload: the user's own
    references), in the order of the panel's inputs."""
    if video:
        return ["default", VIDEO_DROP, *THRESHOLDS["video"], *THRESHOLDS["image"], "threshold", VLM_SCALE]
    return ["default", *THRESHOLDS["image"], "threshold", VLM_SCALE]


def codec_view(strategy, vlm):
    """Which codec controls the panel shows: the share slider (priority), the thresholds (threshold), the text encoder
    row (not with default, which takes every setting from the config), its scale (VLM threshold x scale), and the
    one-line note of the strategy."""
    return (gr.update(visible=strategy == "priority"), gr.update(visible=strategy == "threshold"),
            gr.update(visible=strategy != "default"), gr.update(visible=vlm == "threshold"),
            STRATEGY_NOTES.get(strategy, ""))
