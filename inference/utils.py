"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Helpers shared by the command lines: what codec pruning kept of each request, and the summary lines of a run.
"""

import torch


def track_drop(pipe):
    """A dict that holds, after each request, the reference tokens of that request (total), those codec pruning kept
    (kept) and its masks. The pipelines call `encode` once per request."""
    encode, counts = pipe.encode, {}

    def wrapped(*args, **kwargs):
        out = encode(*args, **kwargs)
        masks = out["masks"]
        counts.update(total=sum(m.numel() for m in masks), kept=sum(int(m.sum()) for m in masks), masks=masks)
        return out

    pipe.encode = wrapped
    return counts


def save_masks(path, masks):
    """The codec pruning masks of a request (bool, True = token kept, one array per reference) as a .npz."""
    import numpy as np
    np.savez_compressed(path, **{f"reference_{i + 1}": m.cpu().numpy() for i, m in enumerate(masks)})


def drop_line(counts):
    """'reference tokens dropped 12,345 of 20,000 (61.7%)', or 'every reference token' when codec pruning was off."""
    if not counts.get("total"):
        return "every reference token"
    return (f"reference tokens dropped {counts['total'] - counts['kept']:,} of {counts['total']:,} "
            f"({1 - counts['kept'] / counts['total']:.1%})")


def memory_line(seconds):
    """The run's time and its GPU memory peak."""
    return (f"generation {seconds:.1f} s (model loading excluded), peak GPU memory "
            f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB reserved, "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB allocated")
