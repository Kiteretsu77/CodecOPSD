# Releasing CodecOPSD

For maintainers: the steps of each release, and how we count use of the code and the LoRAs. Use is counted only
through the download statistics of GitHub and Hugging Face.

## Versions

- Each release is a tag `vX.Y.Z` on `main` with two assets: the small wheel `codecopsd-codec` for Linux x86_64 and for
  ARM (aarch64), which holds only the H.264 reader of codec pruning. Its source, its build script and its packaging
  are not in this repo; they stay in our internal repo (`scripts/build_codec_wheel.sh` there writes
  `dist/codecopsd_codec-X.Y.Z-cp310-abi3-linux_<arch>.whl`; build the ARM one with gcc in a manylinux_2_28 aarch64
  container, never with a Cray `cc` wrapper).
- The reader links FFmpeg statically and uses the stable ABI of CPython 3.10, so one wheel per CPU architecture serves
  Python 3.10 and later with glibc 2.28 or newer. Both give bit-identical outputs.
- `pyproject.toml` of this repo depends on those asset URLs, so every `pip install -e .` downloads it and its download
  count is the number of installs; pip picks the wheel of the machine's architecture. Update both URLs there with
  each new version of the reader.
- Any code change, the codec included, ships as a new tag. Nothing downloads code at run time. Only data (example
  archives, LoRA weights) is fetched, always at a pinned version.

## Example media and stored results

- `python scripts/pack_examples.py` writes two archives at the repo root (ignored by git):
  `CodecOPSD_examples_video_<date>.zip` (the video page and its 12-32 GB page) and
  `CodecOPSD_examples_image_<date>.zip`, paths as in the repo (unzip at the repo root).
- Upload each to `assets/` of its LoRA repo on Hugging Face, set the names in `EXAMPLE_ARCHIVES` and the new commits
  in `LORA_REPOS` (`src/codecopsd/download.py`). Every install downloads them once before the demo starts; a new
  name makes every install download them again.

## LoRAs on Hugging Face

Each released CodecOPSD LoRA has its own model repo, which holds:

- the LoRA `.safetensors`;
- a small `config.json` with the base model, LoRA rank and alpha, and the version. The Hub counts GET and HEAD
  requests to `config.json` as downloads;
- the example archive of its model under `assets/`;
- a model card without `library_name`, with the tag `minimax-h3`, from which the Hub files the repo under the
  `minimax-h3` library. Do not set `diffusers`: that library has its own counting rule, which may not count this
  `config.json`.

Set the repo and its pinned revision in `src/codecopsd/download.py` (`LORA_REPOS`). The repos are
`HikariDawn/CodecOPSD-H3-Ref2VA` and `HikariDawn/CodecOPSD-H3-Image`.

## Download count of every LoRA (required)

Every load of a CodecOPSD LoRA counts once on the Hub. The Hub files both repos under the `minimax-h3` library (from
the `minimax-h3` tag), whose count rule takes requests for `.safetensors` files and for `config.json`.

- Every load goes through `load_condition_lora()` of the pipelines (command lines, the app pages including the Flash
  page, the Python API). For a local file of ours it calls `count_lora_load()` (`src/codecopsd/download.py`), which
  requests the repo's `config.json` with `hf_hub_download`. A cached copy still sends one HEAD request, so every load
  counts.
- A LoRA that `ensure_lora()` has just downloaded is not counted again: the download itself counted.
- Given a repo id, `load_condition_lora()` requests the `.safetensors` file, a HEAD when it is cached, and that
  request counts.
- Training that starts from a released LoRA (`--init-lora`) calls `count_lora_load()` once, on rank 0.
- Failures are ignored, so offline runs still work. `HF_HUB_OFFLINE=1` turns counting off.
- Each count prints a `[download] counted: ...` line.

Any new entry point that loads a LoRA must go through `load_condition_lora()`. If the repos ever lose the
`minimax-h3` library, a download no longer counts by itself: then call `count_lora_load()` after downloads too.

## GitHub traffic

GitHub keeps clone and view counts for 14 days only. Archive them every day, together with stars, forks, the
download counts of the release assets and the Hub download counts of the LoRA repos. Start within 14 days of the
repository going public.
