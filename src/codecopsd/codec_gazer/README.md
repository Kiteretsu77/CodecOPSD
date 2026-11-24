# Codec pruning (`codecopsd.codec_gazer`)

Codec pruning decides which reference tokens a CodecOPSD student drops. It asks a video codec: each reference is
re-encoded with x264, and a token is dropped when the codec spends neither motion nor residual on it, i.e. predicts it
for free from earlier frames (a static background, an unchanged region, a flat area). It reads only the reference's
pixels, so it needs no model, no training and no GPU. Settings: `src/codecopsd/configs/codec_gazer.yaml`. How to reuse it in
another model: `docs/codec_pruning.md`.

```python
from codecopsd.codec_gazer import drop_mask, keep_mask

dropped = drop_mask(video, drop=0.7)   # True = dropped; drop_mask = ~keep_mask, the pipelines index with keep_mask
mask = keep_mask(video)  # RGB video [T, 3, H, W] (uint8, or float in [-1, 1]) -> bool [T', H / 32, W / 32]
mask = keep_mask(image)  # RGB picture [3, H, W] -> bool [1, H / 32, W / 32]
mask = keep_mask(video, drop=0.7)               # priority drop: 70 % of the video's tokens dropped (recommended)
mask = keep_mask(track_video, kind="motion_track")  # a condition with settings of its own (table below)
g = gaze(video, drop=0.7, vlm=True)             # g.keep (transformer) and g.vlm_keep (the student's VLM text encoder)
```

## Decision

1. Re-encode the reference with x264 (settings below) and decode it with a patched FFmpeg (`codecopsd-codec`), which
   returns, per frame and 16 x 16 macroblock, the largest motion vector, the sum of |luma residual| and whether the
   macroblock was skipped.
2. A skipped macroblock counts as no motion and no residual.
3. A token is 2 x 2 macroblocks (32 x 32 pixels, VAE stride 16 x patch 2) in one token frame. Token frames follow the
   VAE: MiniMax-H3 (`MINIMAX_H3`) per 17-frame clip 1 frame and then 4 x 4 frames (17n + 5 frames -> 5n + 2 token
   frames); Cosmos 3 (`COSMOS3`) 1 frame and then 4 frames each (4n + 1 frames -> n + 1). A height or width that is not a
   multiple of 32 (Cosmos 3: 720, 1104, 540) is reflect-padded at the bottom / right to the next one. A token's
   motion is the largest motion over its pixels and frames, its residual the largest per-frame mean |residual|.
4. The token is dropped unless its motion reaches the motion threshold (given at 480 x 640, scaled with the square root
   of the area) or its residual exceeds the residual threshold (a picture's: given at short side 1024, scaled with the
   square root of the short side above it, see below). Never every token is dropped: at least one stays.

## Settings

The settings the CodecOPSD students were trained with (`SETTINGS`, from `configs/codec_gazer.yaml`):

| `kind` | References | Motion (px at 480 x 640) | Mean residual | x264 preset | crf | Keyframe every |
|---|---|---|---|---|---|---|
| `"video"` | RGB videos | 5.0 | 1.0 | ultrafast | 33 | 250 frames |
| `"image"` | RGB pictures (one intra frame) | 4.0 | 1.5 at short side 1024, x sqrt(1024 / short side) above | superfast | 35 | 1 frame |

Per-condition settings (`CONDITIONS`, `keep_mask(x, kind=<name>)`): a condition named here is gazed with the video
settings above, these entries replacing theirs. They are kept as a reference for other condition types; the
MiniMax-H3 students use only the plain `"video"` / `"image"` settings.

| `kind` | Condition | Motion (px at 480 x 640) | Mean residual | x264 preset |
|---|---|---|---|---|
| `"motion_track"` | sparse point tracks (small markers) | 2.0 | 0.5 | ultrafast |
| `"bbox"`, `"segmentation.layout_bbox_filled"`, `"segmentation.layout_bbox_outline"` | layout boxes (flat shapes; their corners move from 1 px / frame) | 1.0 | 0.1 | ultrafast |
| `"degradation"` | degraded real video (blur, low resolution) | 5.0 | 1.0 | veryfast |
| `"interpolation"` | given frames of a real video, grey in between | 5.0 | 1.0 | veryfast |
| `"inpainting"` | a real video with holes | 5.0 | 1.0 | veryfast |

All: rgb24 -> yuv444p, 16 fps, `scenecut=0:bframes=0:ref=1:aq-mode=0`. x264 threads = one per two macroblock rows
(x264's own choice on a machine with many CPUs): x264's decisions depend on its thread count, so it is fixed to make
the masks the same on every machine.

## Picture residual threshold over resolution

A token is kept when r > tau_r, with r the mean |luma residual| of the token (x264 superfast, crf 35, one intra frame) and

    tau_r(H, W) = tau_r0 * sqrt(s0 / max(s0, min(H, W))),   tau_r0 = 1.5,  s0 = 1024 (`residual_short_side`)

tau_r0 = 1.5 is the identity residual threshold set at 1024 x 1024. A picture of short side 1536 / 2048 uses 1.22 / 1.06;
at 1024 and below nothing changes. Videos keep one residual threshold at every size (`residual_short_side: null`). The
student text encoder's picture mask scales the same threshold (x 0.25).

Why: an enlarged picture spreads the same detail over more pixels, so a token's residual falls roughly in proportion to
the short side. A constant 1.5 then drops more and more as the picture grows; a threshold inversely proportional to the
short side keeps the drop flat, but pictures whose residual falls more slowly (dark, noisy backgrounds) then keep
background noise and drop less at a larger size. The square root lies between the two. Released MiniMax-H3 resizes
picture references to short side 2048; this repo keeps a picture's own short side within 1024-2048.

## Priority drop (videos; the pipeline's default, 70 %)

`drop_mask(video, drop=0.7)` (pipeline: `video_drop=0.7`) drops 70 % of the video's n tokens (all but
max(1, round((1 - 0.7) * n))), those of lowest priority r * (1 + m), r and m being the percentile ranks of the token's
residual and motion among all the video's tokens (ties at their mean rank; equal priorities: the later token is dropped
first). With `priority_drop.select: max` (the config's default), a video whose threshold decision already drops more
than the share keeps that decision; `select: exact` always drops exactly the share. Pictures keep the threshold
decision.

## Student text encoder mask (VLM drop)

`gaze(x, ..., vlm=True).vlm_keep`: the tokens of the reference the student's VLM text encoder reads
(`pipelines/minimax_h3_ref2va/vlm_drop.py` maps them onto its vision tokens). A picture: the threshold decision with both
image thresholds x 0.25 (motion 1.0 px, residual 0.375 at short side 1024, scaled with the short side as above; the text encoder
keeps more of a picture than the transformer does), OR
the transformer's mask. A video: the threshold decision OR the transformer's mask (with priority drop that is a superset
of either).

## Elastic drop (optional, videos)

`keep_mask(video, drop="elastic")` (pipeline: `video_drop="elastic"`) gives fewer tokens of a video the thresholds keep
densely, by how much the thresholds alone drop (d), with the numbers of `elastic_drop` in the config:

1. d >= 85 %: the threshold decision as is.
2. d >= 50 % (static, low motion): token frame 0's kept tokens + 50 % of the later frames' kept tokens.
3. otherwise (moving): token frame 0's kept tokens + 75 % of the later frames' kept tokens.

The total drop is then clamped into [50 %, 85 %] and never below d. The later tokens are taken by priority
r * (1 + m) (r, m = percentile ranks of residual and motion among the kept tokens); the result is always a subset of
the threshold decision, and token frame 0 is never thinned. Pictures keep the threshold decision. It is the same rule
for every RGB video reference.

## Sources

| File | Contents |
|---|---|
| `codec.py` | quantization, x264 call, token decision, priority / elastic drop, the text encoder mask |
| `codecopsd-codec` (prebuilt dependency) | the reader, a Python extension (abi3, CPython >= 3.10, Linux x86_64): FFmpeg 5.1 linked statically, with the H.264 IDCT writing 128 + residual instead of prediction + residual, uncoded blocks writing 128, and skipped macroblocks flagged in the exported motion vectors; returns the motion, residual and skip arrays per macroblock; runs without the interpreter lock, frame-threaded |

## Speed

The gaze must never add time. The pipeline starts it before anything else and collects the masks only after the
reference VAE encode:

1. Each reference moves to the GPU once (the VAE needs it there anyway); there it is already uint8 RGB.
2. The frames reach the host by an asynchronous copy into pinned memory; the caller does not wait for it.
3. Each reference is gazed in its own thread (`submit_gaze`). Threads are enough: x264 runs as a separate process and
   the reader runs without the interpreter lock.
4. Meanwhile the GPU encodes the references with the VAE.

The masks are ready long before the reference VAE encode finishes, so the pipeline never waits for them.

Rules for changing it:

- Any change must give the same masks bit for bit: compare against the current code on several sizes and lengths,
  also on a machine with few CPU cores.
- Do not change the x264 settings, its thread count, the FFmpeg patch or the token decision to gain speed: each of
  them changes the masks the students were trained on.
- Quantize where the reference lives (`to_rgb24`) and move only uint8 frames.
