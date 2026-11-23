# Training CodecOPSD

> [!WARNING]
> **Need more check.** This guide and the training scripts are still being verified; commands, defaults, and numbers
> may change.

Sample training code for 4 GPUs. It is not the code the released LoRAs were trained with.

## The recipe

No target images or videos: a sample is a prompt and its references, and the teacher's prediction is the target. The
teacher is the released model with every condition token; the student is the same frozen model plus a LoRA on the
condition rows, given the tokens that codec pruning keeps. One update, on every GPU:

1. Draw a step k. The student takes k sampler steps from noise without gradients, so the state x_k is its own.
2. Teacher velocity at x_k (LoRA off, every condition row), without gradients.
3. Student velocity at x_k; loss = MSE to the teacher over the generated rows.
4. AdamW on the LoRA (lr 5e-5, gradient clipping 1.0); a checkpoint every 100 updates.

Each model trains in two commands: `encode` once (text embeddings, VAE latents, codec pruning masks), then `train`.
Run the same `train` command again to resume.

## 1. MiniMax-H3-Image (reference-to-image generation)

One sample per line of `samples.jsonl`, pictures in prompt order (output 2048 x 2048 by default):

```json
{"prompt": "Put the lamp <Picture 1> on the desk in <Picture 2>.", "images": ["lamp.png", "desk.png"], "height": 2048, "width": 2048}
```

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/train_minimax_h3_image.py encode --model models/MiniMax-H3-Image \
      --data samples.jsonl --out encoded --shard $i/4 &
done; wait
accelerate launch --num_processes 4 scripts/train_minimax_h3_image.py train --model models/MiniMax-H3-Image \
    --encoded encoded --out runs/codecopsd_image
```

The student rolls out on the released six-step grid. Settings: `src/codecopsd/configs/minimax_h3_image_train.yaml`.

## 2. MiniMax-H3 Ref2VA (reference-to-video generation)

One sample per line, references (pictures and videos) in prompt order; frames / height / width default to a draw from
the config's shapes:

```json
{"prompt": "...", "references": ["person.png", "motion.mp4"], "frames": 56, "height": 480, "width": 640}
```

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python scripts/train_minimax_h3_ref2va.py encode --model models/MiniMax-H3-Ref2VA \
      --data samples.jsonl --out encoded --shard $i/4 &
done; wait
accelerate launch --num_processes 4 scripts/train_minimax_h3_ref2va.py train --model models/MiniMax-H3-Ref2VA \
    --encoded encoded --out runs/codecopsd
```

The student rolls out on the 16-step grid of the released scheduler. Settings:
`src/codecopsd/configs/minimax_h3_ref2va_train.yaml`.

## Options of `train`

| Option | Meaning |
|---|---|
| `--fsdp auto` / `on` / `off` | shard the frozen transformer, the LoRA, and its optimizer states over the GPUs (auto: on below 120 GB per GPU) |
| `--live-gaze on` / `off` | the DataLoader workers run codec pruning again on every reference, checked against the encoded masks (default on) |
| `--workers N` | DataLoader workers per GPU (default 2) |
| `--deterministic` | deterministic attention backward: two runs repeat bit for bit (slower backward) |
| `--init-lora x.safetensors` | start from a LoRA |
| `--config my.yaml` | settings (defaults: the two config files above) |

Checkpoints (`<out>/lora_<update>.safetensors`) load into the pipelines with `load_condition_lora`; `<out>/log.jsonl`
records every update (loss, gradient norm, time per stage, peak memory).

## Notes

The DataLoader prepares each sample on the CPU ahead of its update, so the GPUs do not wait for data. Update 1 gives
the same loss on 1, 2, and 4 GPUs, with and without FSDP, and a run killed in the middle of an update resumes with the
same data order and noise.
