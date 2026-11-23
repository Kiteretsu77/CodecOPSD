# Mask settings: which condition tokens to drop

Codec pruning gives every condition a token mask. It has two modes:

- **Threshold**: a token is kept when the codec spends motion or residual on it (largest motion vector at least
  `motion_px`, or mean |luma residual| above `residual`). There is no fixed budget: a static clip drops more than a
  moving one.
- **Priority drop** (videos only): drops a fixed share of a video's tokens, the lowest priority first (residual rank x
  (1 + motion rank)).

Pick the mode by the condition:

| Condition | Mode | Recommended |
|---|---|---|
| Pictures | threshold | the default thresholds; VLM at a quarter of them |
| General videos (RGB footage) | priority drop | drop 70 %; the VLM reuses the same mask |
| Specific condition videos | threshold | the per-condition thresholds below |

## Pictures: threshold

Every picture condition (identity, product, scene, a single given frame, ...) uses the threshold. A picture is one
intra frame with no motion, so the residual threshold decides alone: 1.5 at short side 1024 and below, and
1.5 x sqrt(1024 / short side) above (1536: 1.22, 2048: 1.06), so a larger picture does not lose more of its detail.

The VLM uses the same thresholds x 0.25 (residual 0.375 at short side 1024), together with the DiT's kept tokens. It
therefore reads more of each picture than the DiT, and we observed closer facial similarity.

| Flag (both models) | Default |
|---|---|
| `--picture-thresholds MOTION_PX RESIDUAL` | 4.0 1.5 |
| `--vlm-scale` | 0.25 |
| `--vlm-drop` | `auto` (`dit`: the VLM reads exactly the DiT's tokens; `off`: every vision token) |

```python
from codecopsd.codec_gazer import drop_mask, gaze

dropped = drop_mask(picture)                    # [3, H, W] -> bool [1, H / 32, W / 32], True = dropped
vlm_keep = gaze(picture, vlm=True).vlm_keep     # the VLM's kept tokens (thresholds x 0.25, or kept by the DiT)
```

## General videos: priority drop, 70 %

RGB reference videos (MiniMax-H3 Ref2VA, any real footage) use priority drop at 70 %, the default. Videos repeat
themselves over time, so 70 % is safe and the most stable choice. 66 to 75 % also work: lower for strong camera motion
or busy scenes, higher for calm ones. A video whose threshold decision already drops more than the share keeps that
decision (`--video-drop-select exact` drops exactly the share).

The VLM reuses the DiT's mask of each video (the default), which is the simplest setting.

| Flag (MiniMax-H3 Ref2VA) | Default |
|---|---|
| `--video-drop` | `0.7` (a share in (0, 1), or `threshold`) |
| `--video-drop-select` | `max` (`exact`: always the share) |

```python
dropped = drop_mask(video, drop=0.7)            # [T, 3, H, W] -> bool [T', H / 32, W / 32]
```

## Specific condition videos: threshold

Edge, sketch, depth, surface normal, segmentation, human pose, motion tracks, boxes, blur, grayscale, inpainting,
outpainting, given frames. Use the threshold when the drop must be almost lossless, i.e. the output stays as close to
the full-token output as a change of seed does. The drop then follows the content: sparse marks (tracks, boxes, pose)
lose most of their tokens, dense maps (depth, normal, edges) fewer.

Each condition has its own settings, picked by `kind`
([`codec_gazer.yaml`](../src/codecopsd/configs/codec_gazer.yaml), `conditions`):

| Condition | `kind` | x264 preset | Motion (px at 480 x 640) | Mean residual |
|---|---|---|---|---|
| Segmentation, human pose, depth, surface normal, sketch, edge, dense tracks, grayscale, interpolation and continuation, inpainting, outpainting | `"video"` | ultrafast | 5 | 1.0 |
| Motion tracks (small markers) | `"motion_track"` | ultrafast | 2 | 0.5 |
| Boxes (flat shapes) | `"bbox"` | ultrafast | 1 | 0.1 |
| Blur, low resolution | `"degradation"` | veryfast | 5 | 1.0 |

The motion threshold grows with the square root of the area (it is given at 480 x 640); the video residual threshold is
the same at every size. Lower thresholds keep the small changes that carry content (thin tracks, box corners) but drop
fewer tokens.

```python
dropped = drop_mask(depth_video)                         # no drop share: the threshold decision
dropped = drop_mask(track_video, kind="motion_track")    # a condition with settings of its own
```

On MiniMax-H3 Ref2VA, `--video-drop threshold` applies the plain `"video"` thresholds to every reference video
(`--video-thresholds MOTION_PX RESIDUAL`, default 5.0 1.0, changes them).

More on the codec settings: [`src/codecopsd/codec_gazer/README.md`](../src/codecopsd/codec_gazer/README.md).
