"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.

VLM drop: the CodecOPSD student's text encoder reads only the vision tokens of its references that the codec gazer keeps.

The text encoder sees the same reference pixels as the VAE, and after its 2 x 2 merge one vision token covers 32 x 32 pixels,
like one MiniMax-H3 token. So the gazer's mask of a reference (`codecopsd.codec_gazer.Gaze.vlm_keep`, on the H3 token grid)
is carried over to its vision tokens instead of gazing again:

- picture: the H3 grid is resampled onto the text encoder's merged grid; a vision token is kept when any H3 token it overlaps
  is kept (one to one when the grids agree);
- video: the text encoder reads the reference at 2 fps and merges the sampled frames in pairs, one vision block per pair. A
  block stands for the pixel frames after the previous block's last sampled frame up to its own last one (block 0 from
  frame 0); it keeps a position when any H3 token frame of those pixel frames keeps it, then the grid is resampled
  as for a picture.

The dropped vision tokens are removed right before the text encoder's language model, after the vision tower: every text token
and every kept vision token keeps its multimodal rotary position, its deepstack features and its place, so with every
vision token kept the result is exactly the released encoding. The teacher always reads every vision token.
"""

from contextlib import contextmanager

import torch

from ...codec_gazer import MINIMAX_H3, frame_groups


def sampled_frames(num_frames, fps, sample_fps):
    """Indices of the frames the text encoder reads from a `num_frames` reference at `fps`: every fps / sample_fps-th frame,
    rounded, without repeats (the rule of diffusers' MiniMax-H3 text encoder step)."""
    step, at, out = fps / sample_fps, 0.0, []
    while round(at) < num_frames:
        if not out or round(at) > out[-1]:
            out.append(round(at))
        at += step
    return out


def _or_resize(keep, height, width):
    """bool [h, w] -> bool [height, width]: an output cell is kept when any input cell it overlaps is kept (both grids
    span the same picture)."""
    def spans(n_out, n_in):  # input cells [lo, hi) under output cell i
        return [(i * n_in // n_out, max(-(-(i + 1) * n_in // n_out), i * n_in // n_out + 1)) for i in range(n_out)]
    rows = torch.stack([keep[lo:hi].any(0) for lo, hi in spans(height, keep.shape[0])])
    return torch.stack([rows[:, lo:hi].any(1) for lo, hi in spans(width, keep.shape[1])], 1)


def vision_token_keep(references, vlm_keeps, vision_inputs, processor, fps, sample_fps, grid=MINIMAX_H3):
    """bool over every vision token of the presentation, in its order (references in prompt order, a video block by
    block), from the references' text encoder masks (`Gaze.vlm_keep`, H3 token grid).

    Args:
        references: the normalized references (diffusers MiniMax-H3 references), in prompt order.
        vlm_keeps: bool [T', h, w] per reference.
        vision_inputs: the text encoder's vision inputs of the presentation (`image_grid_thw` / `video_grid_thw` in order).
        processor: the text encoder's processor (merge size, temporal patch).
        fps, sample_fps: the references' frame rate and the rate the text encoder reads videos at.
    """
    merge = processor.image_processor.merge_size
    pairs = processor.video_processor.temporal_patch_size
    images, videos = iter(vision_inputs.get("image_grid_thw", [])), iter(vision_inputs.get("video_grid_thw", []))
    out = []
    for ref, keep in zip(references, vlm_keeps, strict=True):
        keep = keep.bool().cpu()
        if ref.kind == "image":
            t, h, w = (int(v) for v in next(images))
            assert t == 1 and keep.shape[0] == 1, (t, keep.shape)
            out.append(_or_resize(keep[0], h // merge, w // merge).reshape(-1))
            continue
        blocks, h, w = (int(v) for v in next(videos))
        num_frames = len(ref.frames)
        starts = frame_groups(num_frames, grid)
        token_frame = [sum(s <= f for s in starts) - 1 for f in range(num_frames)]  # pixel frame -> H3 token frame
        assert keep.shape[0] == len(starts), (keep.shape, len(starts))
        picked = sampled_frames(num_frames, fps, sample_fps)
        assert blocks == -(-len(picked) // pairs), (blocks, len(picked))
        for b in range(blocks):
            first = 0 if b == 0 else picked[b * pairs - 1] + 1
            last = picked[min((b + 1) * pairs, len(picked)) - 1]
            frames = sorted({token_frame[f] for f in range(first, last + 1)})
            out.append(_or_resize(keep[frames].any(0), h // merge, w // merge).reshape(-1))
    return torch.cat(out)


@contextmanager
def drop_vision_rows(text_encoder, vision_keep):
    """While active, the text encoder's language model runs on the text tokens and the kept vision tokens only (`vision_keep`:
    bool over the vision tokens of the presentation, in order). Yields a dict whose "rows" is then the bool keep over
    every row of the presentation."""
    state = {"rows": None}

    def keep_rows(module, args, kwargs):
        vision = kwargs["visual_pos_masks"]  # [1, L]: the vision rows
        if int(vision.sum()) != vision_keep.numel():
            raise ValueError(f"{vision_keep.numel()} vision keeps for {int(vision.sum())} vision tokens")
        keep = vision_keep.to(vision.device)
        rows = torch.ones_like(vision[0])
        rows[vision[0]] = keep
        state["rows"] = rows.cpu()
        kwargs["inputs_embeds"] = kwargs["inputs_embeds"][:, rows]
        kwargs["position_ids"] = kwargs["position_ids"][..., rows]
        kwargs["attention_mask"] = kwargs["attention_mask"][:, rows]
        kwargs["visual_pos_masks"] = vision[:, rows]
        kwargs["deepstack_visual_embeds"] = [e[keep] for e in kwargs["deepstack_visual_embeds"]]
        return args, kwargs

    handle = text_encoder.model.language_model.register_forward_pre_hook(keep_rows, with_kwargs=True)
    try:
        yield state
    finally:
        handle.remove()
