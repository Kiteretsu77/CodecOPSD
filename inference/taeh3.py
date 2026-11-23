"""TAEH3, the encoder half of the Tiny AutoEncoder for MiniMax H3 (madebyollin/taehv, `taeh3`), for the extreme path's
reference encoding (inference/extreme.py `use_taeh3_encoder`).

The network code below is copied from taehv.py of https://github.com/madebyollin/taehv at commit
9055eee0391548d1787938d040a161cbe396bf98 (the encoder parts only, unchanged), under its MIT License:

    MIT License, Copyright (c) 2025 Ollin Boer Bohan. Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions: The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

taeh3 reads RGB in [0, 1] (frames N, T, C, H, W) and writes the normalized H3 latent ((z - latents_mean) / latents_std):
17-frame chunks with three leading zero frames, the last three latents dropped, as the released VAE chunks a video.
"""

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in * 2, n_out), nn.ReLU(inplace=True), conv(n_out, n_out), nn.ReLU(inplace=True),
                                  conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


def apply_model_with_memblocks_sequential_single_step(model, memory, work_queue):
    while work_queue:
        xt, i = work_queue.pop(0)
        if i == len(model):
            return xt.unsqueeze(1)
        b = model[i]
        if isinstance(b, MemBlock):
            xt_new = b(xt, xt * 0) if memory[i] is None else b(xt, memory[i])
            memory[i] = xt
            work_queue.insert(0, TWorkItem(xt_new, i + 1))
        elif isinstance(b, TPool):
            if memory[i] is None:
                memory[i] = []
            memory[i].append(xt)
            if len(memory[i]) > b.stride:
                raise ValueError(f"TPool memory overflow: {len(memory[i])} items for stride {b.stride}")
            elif len(memory[i]) == b.stride:
                N, C, H, W = xt.shape
                xt = b(torch.cat(memory[i], 1).view(N * b.stride, C, H, W))
                memory[i] = []
                work_queue.insert(0, TWorkItem(xt, i + 1))
        else:
            xt = b(xt)
            work_queue.insert(0, TWorkItem(xt, i + 1))
    return None


def apply_model_with_memblocks_sequential(model, x):
    """x: N, T, C, H, W; frame by frame through the blocks (memory O(1) in T)."""
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    out = []
    while work_queue:
        xt = apply_model_with_memblocks_sequential_single_step(model, memory, work_queue)
        if xt is not None:
            out.append(xt)
    return torch.cat(out, 1)


class TAEH3Encoder(nn.Module):
    """taehv's TAEHV encoder with the taeh3 settings (patch size 2, 24 latent channels, time downscale (True, True,
    False)), loading the `encoder.*` tensors of taeh3.safetensors."""

    def __init__(self, path):
        super().__init__()
        self.patch_size, latent_channels, image_channels = 2, 24, 3
        self.encoder = nn.Sequential(
            conv(image_channels * self.patch_size ** 2, 64), nn.ReLU(inplace=True),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 2), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            TPool(64, 1), conv(64, 64, stride=2, bias=False), MemBlock(64, 64), MemBlock(64, 64), MemBlock(64, 64),
            conv(64, latent_channels),
        )
        from safetensors.torch import load_file
        sd = {k: v for k, v in load_file(str(path)).items() if k.startswith("encoder.")}
        self.load_state_dict(sd)

    @torch.no_grad()
    def encode_video(self, x):
        """x: RGB in [0, 1], N, T, C, H, W -> normalized H3 latents N, T', 24, H / 16, W / 16 (taehv `_encode_h3_video`,
        sequential): the video cut into 17-frame chunks (the last frame repeated), three zero frames before each."""
        batch = x.shape[0]
        x = torch.cat([x, x[:, -1:].expand(-1, -x.shape[1] % 17, -1, -1, -1)], dim=1)
        x = F.pad(x.reshape(batch, -1, 17, *x.shape[2:]), (0, 0, 0, 0, 0, 0, 3, 0))
        x = F.pixel_unshuffle(x, self.patch_size)
        x = torch.cat([apply_model_with_memblocks_sequential(self.encoder, chunk) for chunk in x.unbind(1)], dim=1)
        return x[:, :-3]
