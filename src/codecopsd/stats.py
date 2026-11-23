"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Where the seconds of one request go, stage by stage: preparing the references, the VAE encode, codec pruning, the
text encoder, model moves between CPU and GPU memory, each denoising step, the VAE decode, and what the caller adds
(writing the video, drawing the kept tokens). Every stage is timed between two GPU synchronizations; the model moves
that happen inside a stage (pinned model offload, `codecopsd.offload.MoveLog`) are taken out of it and counted in their own
row, so the rows add up to the whole request. Each row also has the peak GPU memory reached during it.
"""

import time
from contextlib import contextmanager

import torch

MOVES = "model moves (CPU <-> GPU)"


class Clock:
    """The stage rows of one request: `rows` = [dict(stage, seconds, peak_gib, note)], in order of first use. While a
    stage runs, `now` is (its name, its start time), for a live display (the demo's progress line)."""

    def __init__(self, moves=None):
        self.rows, self._index = [], {}
        self.moves = moves  # an codecopsd.offload.MoveLog, or None
        self.t0 = time.perf_counter()
        self._peak = self._peak_reserved = 0.0
        self.now = None
        self._moved0 = moves.seconds if moves is not None else 0.0  # the moves before this request are not its own
        self._moved_in = 0.0  # seconds of moves made inside stages (counted in the MOVES row)

    def _row(self, stage):
        if stage not in self._index:
            self._index[stage] = len(self.rows)
            self.rows.append(dict(stage=stage, seconds=0.0, peak_gib=None, note=""))
        return self.rows[self._index[stage]]

    def add(self, stage, seconds, peak_gib=None, note=""):
        row = self._row(stage)
        row["seconds"] += seconds
        if peak_gib is not None:
            row["peak_gib"] = max(row["peak_gib"] or 0.0, peak_gib)
        if note:
            row["note"] = note
        return row

    @contextmanager
    def stage(self, name, note=""):
        """Time the block as stage `name` (GPU work included), without the model moves made inside it."""
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            self._keep_peaks()
            torch.cuda.reset_peak_memory_stats()
        moved = self.moves.seconds if self.moves is not None else 0.0
        t = time.perf_counter()
        self.now = (name, t)
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            self.now = None
            seconds = time.perf_counter() - t
            peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
            if cuda:
                self._keep_peaks()
            inside = (self.moves.seconds - moved) if self.moves is not None else 0.0
            self.add(name, seconds - inside, peak, note)
            if inside > 0:
                self._moved_in += inside
                self.add(MOVES, inside)

    def add_outside_moves(self):
        """Add to the MOVES row the moves made since the clock started outside every stage (diffusers' model offload
        sends every model back to the CPU between the denoising and the decode, and at the end of a call). Call once,
        when the request has ended. Returns the seconds added."""
        if self.moves is None:
            return 0.0
        outside = self.moves.seconds - self._moved0 - self._moved_in
        if outside > 0.01:
            self._moved_in += outside
            self.add(MOVES, outside)
            return outside
        return 0.0

    def total(self):
        return time.perf_counter() - self.t0

    def _keep_peaks(self):
        self._peak = max(self._peak, torch.cuda.max_memory_allocated() / 2**30)
        self._peak_reserved = max(self._peak_reserved, torch.cuda.max_memory_reserved() / 2**30)

    def peak_gib(self, reserved=False):
        """The peak GPU memory (GiB) of the request so far: allocated tensors, or with `reserved` the allocator's cache
        (what the process holds; `nvidia-smi` adds the CUDA context, about 0.5 GiB)."""
        if torch.cuda.is_available():
            self._keep_peaks()
        return self._peak_reserved if reserved else self._peak

    def table(self, total=None):
        """Plain-text table of the rows plus the rest of `total` (default: since the clock started)."""
        total = self.total() if total is None else total
        lines = [f"{'stage':<44} {'seconds':>8} {'share':>6} {'peak GB':>8}"]
        counted = 0.0
        for r in self.rows:
            counted += r["seconds"]
            peak = f"{r['peak_gib']:.1f}" if r["peak_gib"] is not None else ""
            lines.append(f"{r['stage']:<44} {r['seconds']:8.1f} {r['seconds'] / max(total, 1e-9):6.0%} {peak:>8}"
                         + (f"  {r['note']}" if r["note"] else ""))
        rest = total - counted
        if rest > 0.05:
            lines.append(f"{'other (Python, CPU work between stages)':<44} {rest:8.1f} {rest / max(total, 1e-9):6.0%}")
        lines.append(f"{'whole request':<44} {total:8.1f} {1:6.0%} {self.peak_gib():8.1f}")
        return "\n".join(lines)
