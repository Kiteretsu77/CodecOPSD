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

- `python scripts/pack_examples.py` writes the archives of every model to `release_assets/` (ignored by git).
- Upload them to a release tagged `examples-vN` and set `EXAMPLES_TAG` in `src/codecopsd/download.py` to that tag.
  Every install downloads them once before the demo starts. A new tag makes every install download them again.

## LoRAs on Hugging Face

Each released CodecOPSD LoRA has its own model repo, which holds:

- the LoRA `.safetensors`;
- a small `config.json` with the base model, LoRA rank and alpha, and the version. The Hub counts GET and HEAD
  requests to `config.json` as downloads. Requests for the weights alone are not counted.
- a model card without `library_name`. In particular, do not set `diffusers`: that library has its own counting
  rule, which may not count this `config.json`.

Set the repo in `src/codecopsd/download.py` (`LORA_REPO`) and pin its revision.

## Download count of every LoRA (required)

Every code path that loads a CodecOPSD LoRA fetches the `config.json` of its repo with `hf_hub_download`. It does so
on every load, also when the LoRA is already on disk. Failures are ignored, so offline runs still work. With a cached
`config.json`, `huggingface_hub` still sends one HEAD request, so each launch counts once. `HF_HUB_OFFLINE=1` turns
this off.

The entry points that must do it:

- `ensure_lora()` in `src/codecopsd/download.py`, before its early return for a LoRA that is already on disk;
- `inference/generate.py` (command lines);
- the app pages (`inference/demo/main.py`, the Flash page included);
- `load_condition_lora()` of the pipelines (Python API).

Any new entry point that loads a LoRA must do it as well. After a LoRA repo goes up, load the LoRA once and check
that the repo's download count on the Hub goes up.

## GitHub traffic

GitHub keeps clone and view counts for 14 days only. Archive them every day, together with stars, forks, the
download counts of the release assets and the Hub download counts of the LoRA repos. Start within 14 days of the
repository going public.
