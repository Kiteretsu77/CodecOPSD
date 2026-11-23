# Releasing CodecOPSD

For maintainers: the steps of each release, and how we count use of the code and the LoRAs. Use is counted only
through the download statistics of GitHub and Hugging Face.

## Versions

- Each release is a tag `vX.Y.Z` on `main`. `version` in `pyproject.toml` matches the tag.
- Build the codec reader first (`bash scripts/build_codec_reader.sh`). The wheel must contain
  `codecopsd/codec_gazer/_codec_reader.abi3.so` (listed in `package-data` of `pyproject.toml`); without it, codec
  pruning stops with an ImportError. The reader links FFmpeg statically, so the wheel is a Linux x86_64 wheel.
- Build the wheel with `pip wheel . --no-deps -w dist/` and attach it to the GitHub Release of that tag.
- The README installs the library from that asset, so its download count is the number of installs:
  `pip install https://github.com/Kiteretsu77/CodecOPSD/releases/download/vX.Y.Z/<wheel file>`
- The demo and the command lines need `app.py` and `inference/`, which are not in the wheel. They clone the same tag
  (`git clone --branch vX.Y.Z https://github.com/Kiteretsu77/CodecOPSD.git`).
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
