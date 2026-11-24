# Consumer GPUs on Ref2VA

MiniMax-H3 Ref2VA on a 12 to 32 GB GPU, optimized for the RTX 3090, RTX 4090 and RTX 5090. The standard app
(`python app.py ref2va`, 80 GB GPUs and more) stays as it is; everything below concerns the extreme app only.

```bash
python app.py ref2va_extreme                                   # plans for the first GPU's memory
python app.py ref2va_extreme --gpu-memory 24                   # plans for a 12, 16, 24 or 32 GB card
python inference/generate.py h3_flash_ref2va --gpu-memory 24   # command line: a bundled example
python -m inference.hardware                                   # this machine: GPU, CPU memory, kernels, warnings
```

## General setting

The seven points of the README, in detail. Everything else is as in the standard app: codec pruning, VLM pruning,
the CodecOPSD LoRA and the released VAE decoder.

1. **Transformer.** [HyperFlow H3 Flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) replaces the
   released MiniMax-H3 transformer. It is Video Rebirth's compressed MiniMax-H3 (13.9B parameters, INT8 and low rank,
   16 GB) with its own sampler: 4 steps by default (`--steps 6` or `8`) instead of the 6 of Pika H3 Flex. The
   CodecOPSD LoRA runs on it unchanged, except on block 9, which Flash removed.
2. **Reference encoder.** TAEH3, a small encoder, encodes the reference pictures and videos instead of the released
   VAE encoder. The released VAE still decodes the video. `--vae-encoder released` brings the released encoder back.
3. **VLM.** Downloaded already quantized to FP8 (31.7 GB), so no machine has to read and convert the 66.7 GB bf16
   release. The download is pinned to one revision of a public Hub copy whose weights equal our own FP8 conversion; it
   keeps the 50 decoder layers MiniMax-H3 reads, one fewer than converting the release gives. If that copy cannot be
   had, the app falls back to a folder saved by `python -m inference.vlm_fp8`, then to the bf16 release, downloaded
   and converted at load as in the standard app (on the command line, `--vlm-fp8 none` forces this). Every source
   gives the same video, and the run prints which one it uses.
4. **GPU memory.** One plan per card size, picked from the GPU's memory or from `--gpu-memory` (on a larger GPU this
   also caps the process to that size, so it shows what that card does).

   | GPU | On the GPU | Waits in CPU memory |
   |---|---|---|
   | 64 GB and more | every model, all the time | nothing |
   | 32 GB | the model that runs; Flash whole while it denoises | the other models; the VLM, streamed layer by layer |
   | 24 GB | the same, with 33 of Flash's 49 blocks | the same, and 16 Flash blocks, streamed in at every step |
   | 12 and 16 GB | the same, with no Flash block | the same, and all 49 Flash blocks, streamed in at every step |

   Flash runs its 49 blocks in order at every step. A streamed block is copied to the GPU just before it runs (the
   next one is copied ahead, on a side stream) and dropped after it runs, so at most two streamed blocks are on the
   GPU at a time. Flash alone is 16.4 GB, more than a 16 GB card holds next to the activations, so the 16 and 12 GB
   plans stream every block. Every plan computes the same products from the same weights and gives the same video.
5. **CPU memory.** The models that wait in CPU memory are page-locked, so they cross the PCIe link at full speed. On a
   12 to 32 GB plan this keeps about 54.5 GB while generating: the FP8 VLM (25.7 GB), Flash (16.4 GB) and the VAE
   (5.2 GB), plus Python, PyTorch and CUDA. With less than 64 GB of CPU memory available, the VLM is not locked but
   read from the page cache at each request: about 29 GB kept, and a slower VLM step. `python -m inference.hardware`
   prints what this machine gets.
6. **Kernels.** The attention, INT8 and FP8 kernels follow the GPU generation (sections below). Flash's INT8 kernel
   is chosen per generation because the faster one differs; both kernels give the same video.
7. **Output and references.** 480p by default (`480 x 832`; portrait and larger sizes are offered, and a larger size
   needs more GPU memory), 56, 90 or 124 frames. A reference picture is read at its own short edge, kept between
   1024 and 2048 px: a smaller picture is enlarged to 1024, a larger one reduced to 2048, one in between is read as
   uploaded (resized, never cropped; sides rounded to multiples of 32). The standard app reads every picture at a
   2048 short edge, as released, so a 1024 px picture there takes 4 times as many tokens with no new detail. Reference
   videos are read as in the standard app. The bundled examples (ten cases of the standard app and cases published by others) use smaller
   reference files than the standard app: pictures with a short edge of 1024 or 1536 px, videos at 480p.

**Requirements:**
- Linux x86_64 and an NVIDIA GPU with 12 GB or more.
- 64 GB of CPU memory for the fastest path; less works, with the VLM read from the page cache.
- 62 GB of disk, all downloaded on the first launch: the FP8 VLM (31.7 GB), `models/MiniMax-H3-Ref2VA/` without its
  transformer and bf16 text encoder (11.1 GB), `models/hyperflow-h3-flash/ref2va_int8_convrot/` (16.3 GB) and the
  LoRA (2.6 GB).

## RTX 30 series (Ampere)

- **Attention:** PyTorch SDPA, as cuDNN attention is slower on this generation.
- **INT8 GEMMs:** cuBLAS, which is faster here than Flash's own Triton kernel and gives the same output.
- **VLM:** these GPUs have no FP8 tensor cores, so the FP8 weights are multiplied in bf16. The weights still take only
  FP8 memory.
- **Plan:** 24 GB on an RTX 3090.

## RTX 40 series (Ada Lovelace)

- **Attention:** PyTorch SDPA.
- **INT8 GEMMs:** cuBLAS, as on the RTX 30 series.
- **VLM:** FP8 tensor cores.
- **Plan:** 24 GB on an RTX 4090. Flash's streamed blocks cross the PCIe link at every step, so a host with PCIe 4.0 or
  newer runs faster than one with PCIe 3.0.

## RTX 50 series (Blackwell)

- **Attention:** cuDNN.
- **INT8 GEMMs:** Flash's Triton kernel, which is faster here than its default cuBLASLt and gives the same output.
- **VLM:** FP8 tensor cores.
- **Plan:** 32 GB on an RTX 5090. Flash stays whole on the GPU while it runs, so no block crosses the PCIe link during
  the denoising steps.
