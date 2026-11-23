"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Codec pruning is a contribution of this paper; please cite it if you use, modify, or port this code.

Tests of `codecopsd.codec_gazer.drop_mask` on synthetic inputs (CPU only, no model, no download, about 20 s):

    python tests/test_drop_mask.py        # or: pytest tests/test_drop_mask.py

Needs the H.264 reader (`bash scripts/build_codec_reader.sh`).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from codecopsd.codec_gazer import PRIORITY_DROP, drop_mask  # noqa: E402

T, H, W = 124, 480, 832  # a MiniMax-H3 video reference: 124 frames -> 37 token frames of 15 x 26 tokens
GRID = (37, H // 32, W // 32)


def moving_square():
    """A static random texture with a white square moving 4 pixels to the right per frame, in [-1, 1]."""
    g = torch.Generator().manual_seed(0)
    video = (torch.rand(1, 3, H, W, generator=g) * 2 - 1).repeat(T, 1, 1, 1)
    for t in range(T):
        video[t, :, 200:296, 64 + 4 * t:160 + 4 * t] = 1.0
    return video


def noise():
    """New random noise in every frame: the encoder spends bits on every token."""
    return torch.rand(T, 3, H, W, generator=torch.Generator().manual_seed(1)) * 2 - 1


def test_shape_and_type():
    dropped = drop_mask(moving_square())
    assert dropped.shape == GRID and dropped.dtype == torch.bool
    assert (~dropped).any(), "at least one token is kept"


def test_static_dropped_motion_kept():
    dropped = drop_mask(moving_square())  # the threshold decision
    square_rows, static_rows = ~dropped[1:, 6:10], ~dropped[1:, :5]  # token rows 6-9 hold the square (pixels 200-296)
    assert static_rows.float().mean() < 0.1, "a static background is mostly dropped"
    for t in range(square_rows.shape[0]):
        assert square_rows[t].any(), f"token frame {t + 1}: the moving square keeps tokens"
    assert dropped.float().mean() > 0.8


def test_priority_share():
    video = noise()
    assert not drop_mask(video).any(), "threshold decision: noise keeps every token"
    for share in (0.5, PRIORITY_DROP, 0.9):
        dropped = drop_mask(video, drop=share)
        assert abs(dropped.float().mean().item() - share) < 1e-3, share


def test_threshold_wins_when_it_drops_more():
    video = moving_square()
    threshold = drop_mask(video)
    assert torch.equal(drop_mask(video, drop=0.5), threshold)  # priority_drop.select: max


def test_uint8_input():
    video = noise()
    as_uint8 = video.add(1).mul(127.5).round().to(torch.uint8)
    assert torch.equal(drop_mask(as_uint8, drop=0.7), drop_mask(video, drop=0.7))


def test_picture():
    picture = torch.randint(0, 256, (3, 1536, 1536), dtype=torch.uint8, generator=torch.Generator().manual_seed(2))
    picture[:, :768] = 128  # flat top half, detailed bottom half
    dropped = drop_mask(picture)
    assert dropped.shape == (1, 48, 48)
    assert dropped[:, :24].all() and not dropped[:, 24:].any()


def test_masking_tokens():
    dropped = drop_mask(moving_square(), drop=0.7)
    tokens = torch.randn(*dropped.shape, 16)  # condition tokens on the same grid [T', H / 32, W / 32, C]
    kept, positions = tokens[~dropped], (~dropped).nonzero()
    assert kept.shape == (int((~dropped).sum()), 16) and positions.shape == (kept.shape[0], 3)
    assert torch.equal(tokens[tuple(positions.T)], kept)


if __name__ == "__main__":
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_")]
    for name, f in tests:
        f()
        print("ok", name, flush=True)
    print(f"{len(tests)} tests passed")
