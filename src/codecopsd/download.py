"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Everything the demo and the command line read, downloaded on first use into `models/` (the released models and the
CodecOPSD LoRAs from the Hugging Face Hub at pinned revisions), `assets/examples/refs/` (the media of the bundled
examples) and, for the demo, `assets/examples/cache/` (its stored example results); the last two come as one zip
archive per LoRA repository on the Hub (`EXAMPLE_ARCHIVES`).

    python -m codecopsd.download                       # both models + the example media
    python -m codecopsd.download --results             # + the demo's stored example results
    python -m codecopsd.download minimax_h3_image      # one model
    python -m codecopsd.download --dry-run             # what would be downloaded, and its size

Older layouts keep working: a model folder at the repository root (`MiniMax-H3-Ref2VA/`) or a LoRA in `weights/` is
used when `models/` has none. MiniMax-H3 and MiniMax-H3-Image ship the same text encoder (63 GB): it is downloaded once
and linked into the second folder when the Hub files are identical.

The text encoder (VLM) is downloaded already converted to FP8 (`ensure_vlm_fp8`, 29.6 GB), and the bf16 one of the
releases only when that fails or a run reads the bf16 VLM (the teacher of `--mode baseline`, `--bf16-vlm` here).
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
EXAMPLES = ROOT / "assets" / "examples"
# The example media (assets/examples/refs/) and the demo's stored example results (assets/examples/cache/) stay out of
# git. They come as one zip archive per LoRA repository on the Hugging Face Hub (`LORA_REPOS`), its paths as in this
# repository (assets/examples/...), downloaded once per install (the demo unpacks it before it starts). New data = a new
# archive name here (`python scripts/pack_examples.py` writes them at the repository root; upload them to assets/ of
# the repository), which every install downloads again.
EXAMPLE_ARCHIVES = {"minimax_h3_ref2va": "CodecOPSD_examples_video_20261125.zip",  # the video page and its 12-32 GB page
                    "h3_flash_ref2va": "CodecOPSD_examples_video_20261125.zip",
                    "minimax_h3_image": "CodecOPSD_examples_image_20261123.zip"}

_META = ["processor/*", "tokenizer/*", "scheduler/*", "model_index.json", "modular_model_index.json"]
# folder under models/: (Hub repo, pinned revision, files). MiniMax-H3-Ref2VA/ is MiniMaxAI/MiniMax-H3 with only its
# Ref2VA transformer (transformer_ref/), its VAE, its text encoder and its audio VAE and audio scheduler (0.6 GB, read
# only by a request with audio on).
REPOS = {
    "MiniMax-H3-Ref2VA": ("MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe",
                          ["transformer_ref/*", "vae/*", "text_encoder/*", "audio_vae/*", "audio_scheduler/*"]
                          + _META),
    "Pika-H3-Flex": ("Pika-Labs/Pika-H3-Flex", "0ddf55ad7067ca1e76812253bd7b4600df87f759", ["ref2va/adapter.safetensors"]),
    "MiniMax-H3-Image": ("MiniMaxAI/MiniMax-H3-Image", "492fd8c4eca666f1085f57e30738c1e0d86b3439",
                         ["transformer_diffusers/*", "vae_diffusers/*", "text_encoder/*"] + _META),
    "hyperflow-h3-flash": ("VideoRebirthAI/hyperflow-h3-flash", "6834923e1a9ca4d15a2349a622b6677bea1404cb",
                           ["ref2va_int8_convrot/*", "LICENSE", "NOTICE"]),
    # The FP8 text encoder (VLM), already converted: a third-party ModelOpt copy of the MiniMax-H3 VLM (the same VLM as
    # MiniMax-H3-Image's), only its text encoder folder. Its FP8 tensors equal our own conversion byte for byte
    # (codecopsd.models.fp8_linear). Third-party base weights, so nothing of ours is counted on download.
    "MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8": ("feizhai123/MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8",
                                               "98f87ad983dbcd6a956b6623308626b2245b5098",
                                               ["Ref2VA/text_encoder/*", "LICENSE", "NOTICE"]),
}
# model: (folders it reads, its CodecOPSD LoRA file). "h3_flash_ref2va" (`python app.py ref2va_extreme`): HyperFlow H3
# Flash in place of the released Ref2VA transformer, with the same LoRA, text encoder and VAE.
NEEDS = {
    "minimax_h3_ref2va": (["MiniMax-H3-Ref2VA", "Pika-H3-Flex"], "codecopsd_minimax_h3_ref2va.safetensors"),
    "h3_flash_ref2va": (["MiniMax-H3-Ref2VA", "hyperflow-h3-flash"], "codecopsd_minimax_h3_ref2va.safetensors"),
    "minimax_h3_image": (["MiniMax-H3-Image"], "codecopsd_minimax_h3_image.safetensors"),
}
# model: {folder: patterns of REPOS it does not read} (Flash: the 62 GB released Ref2VA transformer)
SKIP = {"h3_flash_ref2va": {"MiniMax-H3-Ref2VA": ["transformer_ref/*"]}}
# model: the model whose example folders and media it reuses (the extreme page runs the video examples at 480p)
EXAMPLES_OF = {"h3_flash_ref2va": "minimax_h3_ref2va"}
# The CodecOPSD LoRAs: their file under models/ -> (Hugging Face repository, pinned revision). Each repository also
# holds a small config.json (counted downloads, `count_lora_load`) and the example archive of its model under assets/.
LORA_REPOS = {
    "codecopsd_minimax_h3_ref2va.safetensors": ("HikariDawn/CodecOPSD-H3-Ref2VA", "98e6ba85446430541f6f578bfa3a2c18ae395324"),
    "codecopsd_minimax_h3_image.safetensors": ("HikariDawn/CodecOPSD-H3-Image", "944e980d9c9cba38571b4de4e82d91a0ae535e85"),
}

# The text encoder (VLM) of every model: why it comes already converted to FP8. Converting the bf16 release at load
# reads all 66 GB of it into CPU memory first (and, under strict overcommit, needs that much commit headroom), more
# than many machines with a 12-32 GB GPU have. So the FP8 VLM is downloaded pre-quantized and read as file views.
# Sources, in this order (`vlm_fp8_source`): the downloaded ModelOpt copy (VLM_FP8, its text encoder folder
# VLM_FP8_SUBFOLDER), then an FP8 VLM saved once by `python -m inference.vlm_fp8` (SAVED_VLM_FP8), then the bf16
# release converted at load (the old path, kept as the fallback when the download fails).
VLM_FP8 = "MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8"
VLM_FP8_SUBFOLDER = "Ref2VA/text_encoder"
SAVED_VLM_FP8 = "MiniMax-H3-VLM-FP8"


def model_dir(folder):
    """models/<folder>, or <folder> at the repository root when only that exists (older layout). Pika H3 Flex and
    HyperFlow H3 Flash sit next to the MiniMax-H3 folder in use (the pipeline looks for them there)."""
    if folder in ("Pika-H3-Flex", "hyperflow-h3-flash"):
        return model_dir("MiniMax-H3-Ref2VA").parent / folder
    new, old = MODELS_DIR / folder, ROOT / folder
    return old if not new.exists() and old.is_dir() else new


def lora_path(name):
    """models/<LoRA file>, or weights/<LoRA file> when only that exists (older layout)."""
    file = NEEDS[name][1]
    new, old = MODELS_DIR / file, ROOT / "weights" / file
    return old if not new.exists() and old.is_file() else new


def _hub_files(repo, revision, patterns):
    """{path: (size, sha256)} of the Hub files a download fetches."""
    from fnmatch import fnmatch
    from huggingface_hub import HfApi

    files = {}
    for f in HfApi().list_repo_tree(repo, revision=revision, recursive=True, expand=True):
        if getattr(f, "size", None) is not None and any(fnmatch(f.path, p) for p in patterns):
            files[f.path] = (f.size, f.lfs.sha256 if getattr(f, "lfs", None) else None)
    return files


PLANNED = set()  # --dry-run: the folders this run would download


def _shared_text_encoder(folder):
    """The other MiniMax folder whose downloaded text encoder is byte-identical on the Hub to the one `folder` needs."""
    other = {"MiniMax-H3-Ref2VA": "MiniMax-H3-Image", "MiniMax-H3-Image": "MiniMax-H3-Ref2VA"}.get(folder)
    if other is None or not ((model_dir(other) / "text_encoder").is_dir() or other in PLANNED):
        return None
    mine = {p: v for p, v in _hub_files(*REPOS[folder][:2], ["text_encoder/*"]).items()}
    theirs = {p: v for p, v in _hub_files(*REPOS[other][:2], ["text_encoder/*"]).items()}
    return model_dir(other) / "text_encoder" if mine and mine == theirs else None


def ensure_folder(folder, dry_run=False, skip=()):
    """Download `folder` (a key of REPOS) into models/ unless it is already there, without the patterns in `skip`;
    a folder that lacks a subfolder a pattern names gets those files. Returns its path."""
    path = model_dir(folder)
    repo, revision, patterns = REPOS[folder]
    patterns = [p for p in patterns if p not in skip]
    if path.is_dir() and any(path.iterdir()):
        patterns = [p for p in patterns if p.endswith("/*") and p.count("/") == 1 and not (path / p[:-2]).is_dir()]
        if not patterns:
            return path
    shared = _shared_text_encoder(folder) if "text_encoder/*" in patterns else None  # not when the FP8 VLM is used
    if shared is not None:
        patterns = [p for p in patterns if not p.startswith("text_encoder")]
    size = sum(s for s, _ in _hub_files(repo, revision, patterns).values())
    note = f" (+ the text encoder linked from {shared.parent.name}/)" if shared is not None else ""
    print(f"[download] {repo} -> {path.relative_to(ROOT)}/: {size / 1e9:.1f} GB{note}", flush=True)
    if dry_run:
        PLANNED.add(folder)
        return path
    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=revision, local_dir=path, allow_patterns=patterns, max_workers=16)
    if shared is not None:
        os.symlink(os.path.relpath(shared, path), path / "text_encoder")
    return path


def vlm_fp8_dir():
    """models/MiniMax-H3-ModelOpt-Mixed9-Dynamic-FP8/Ref2VA/text_encoder: the downloaded FP8 VLM."""
    return model_dir(VLM_FP8) / VLM_FP8_SUBFOLDER


def _complete(folder):
    """Whether a text encoder folder holds its config, its index and every shard the index names."""
    import json

    folder = Path(folder)
    try:
        shards = set(json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"].values())
    except (OSError, ValueError, KeyError):
        return False
    return (folder / "config.json").is_file() and all((folder / s).is_file() for s in shards)


_VLM_FP8_TRIED = {}  # one download attempt per process (the command line and the demo both ask)


def ensure_vlm_fp8(dry_run=False):
    """Download the FP8 VLM (`VLM_FP8`: its text encoder folder, 29.6 GB) into models/ unless it is complete there.
    Returns its folder, or None (with a note) when it cannot be downloaded: the caller falls back to a saved FP8 VLM
    or to the bf16 release (`vlm_fp8_source`). Third-party base weights: no download counter of ours."""
    path = vlm_fp8_dir()
    if _complete(path):
        return path
    if "result" in _VLM_FP8_TRIED:
        return _VLM_FP8_TRIED["result"]
    repo, revision, patterns = REPOS[VLM_FP8]
    result = None
    try:
        size = sum(s for s, _ in _hub_files(repo, revision, patterns).values())
        print(f"[download] FP8 VLM {repo}/{VLM_FP8_SUBFOLDER} -> {model_dir(VLM_FP8).relative_to(ROOT)}/: "
              f"{size / 1e9:.1f} GB", flush=True)
        if dry_run:
            return path
        from huggingface_hub import snapshot_download

        snapshot_download(repo, revision=revision, local_dir=model_dir(VLM_FP8), allow_patterns=patterns,
                          max_workers=16)
        result = path if _complete(path) else None
        if result is None:
            print(f"[download] FP8 VLM: {path} is incomplete after the download", flush=True)
    except Exception as e:  # no network, the Hub down, a full disk: run with the fallbacks
        print(f"[download] FP8 VLM failed ({e!r}); falling back to a saved FP8 VLM or the bf16 release", flush=True)
    _VLM_FP8_TRIED["result"] = result
    return result


def vlm_fp8_source():
    """The FP8 VLM on disk, in the order of preference: ("downloaded", folder) for the ModelOpt copy, ("saved",
    folder) for one saved by `python -m inference.vlm_fp8`, or (None, None): the bf16 release is converted at load."""
    from .models.fp8_linear import fp8_folder_format

    if _complete(vlm_fp8_dir()) and fp8_folder_format(vlm_fp8_dir()) == "modelopt":
        return "downloaded", vlm_fp8_dir()
    saved = model_dir(SAVED_VLM_FP8)
    if _complete(saved) and fp8_folder_format(saved) == "saved":
        return "saved", saved
    return None, None


_FETCHED = set()  # LoRA files this process downloaded (`ensure_lora`): the Hub counted that download already


def count_lora_load(path):
    """Count one load of a CodecOPSD LoRA in its repository's Hugging Face download statistics (docs/releasing.md): one
    request for the repository's config.json, which the Hub counts (a cached copy still sends a HEAD request). A LoRA
    this process just downloaded is not counted twice: the Hub counts that download itself (the count rule of the
    minimax-h3 library takes the .safetensors files too). Other LoRA files are not ours and are not counted. Failures
    (offline, no access, `HF_HUB_OFFLINE=1`) are ignored."""
    entry = LORA_REPOS.get(Path(path).name)
    if entry is None or str(Path(path).resolve()) in _FETCHED:
        return
    try:
        from huggingface_hub import constants, hf_hub_download

        if constants.HF_HUB_OFFLINE:  # a cached config.json would come back without any request: nothing counted
            print(f"[download] not counted: one load of {entry[0]} (HF_HUB_OFFLINE=1)", flush=True)
            return
        hf_hub_download(entry[0], "config.json")
        print(f"[download] counted: one load of {entry[0]} (config.json request)", flush=True)
    except Exception as e:
        print(f"[download] not counted: one load of {entry[0]} ({type(e).__name__})", flush=True)


def ensure_lora(name, dry_run=False):
    """The CodecOPSD LoRA of `name`: its path in models/, downloaded from its Hugging Face repository (`LORA_REPOS`) when
    it is not there, or None (with a note) when it cannot be had. Every load is counted where the LoRA is loaded
    (`count_lora_load`, called by the pipelines' `load_condition_lora`)."""
    path = lora_path(name)
    if path.is_file():
        return path
    repo, revision = LORA_REPOS[path.name]
    print(f"[download] CodecOPSD LoRA {repo} -> {path.relative_to(ROOT)}", flush=True)
    if dry_run:
        return path
    try:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, path.name, revision=revision, local_dir=MODELS_DIR))
    except Exception as e:  # no network, no access: the teacher still runs
        print(f"[download] CodecOPSD LoRA {path.name}: failed ({e!r}); without it, only the teacher runs", flush=True)
        return None
    _FETCHED.add(str(path.resolve()))
    print(f"[download] counted: one load of {repo} (the download of {path.name})", flush=True)
    return path


def example_files(names=None):
    """The media files the bundled examples of `names` (keys of NEEDS; None = every model) refer to."""
    import json

    files = []
    for name in names or NEEDS:
        for spec in sorted((EXAMPLES / EXAMPLES_OF.get(name, name)).glob("*/example.json")):
            files += [Path(a).name for a in json.loads(spec.read_text())["args"] if str(a).startswith("../../refs/")]
    listed = [n.strip() for n in (EXAMPLES / "references.txt").read_text().splitlines() if n.strip()]
    return [n for n in dict.fromkeys(files) if n in listed]


def archive_of(name):
    """(Hub repository, pinned revision, path in it) of the example archive of model `name`."""
    repo, revision = LORA_REPOS[NEEDS[name][1]]
    return repo, revision, f"assets/{EXAMPLE_ARCHIVES[name]}"


def fetch_archive(name, results):
    """Download the example archive of model `name` and unpack its example media into assets/examples/refs/ and, with
    `results`, its stored results of `name` into assets/examples/cache/<name>/ (marked with the archive's name in
    cache/<name>/.release). Returns True when it is in place; a failure is reported and the caller goes on without
    it. The archive stays in the Hugging Face cache, so another model of the same archive does not download it again."""
    import zipfile

    repo, revision, file = archive_of(name)
    wanted = ("assets/examples/refs/",) + ((f"assets/examples/cache/{name}/",) if results else ())
    try:
        from huggingface_hub import hf_hub_download

        print(f"[download] {repo}/{file} -> {EXAMPLES.relative_to(ROOT)}/", flush=True)
        archive = hf_hub_download(repo, file, revision=revision)
        root = ROOT.resolve()
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if m.filename.startswith(wanted) and not m.is_dir()]
            for m in members:  # only files under refs/ or this model's cache folder, inside the repository
                if not (ROOT / m.filename).resolve().is_relative_to(root):
                    raise zipfile.BadZipFile(f"unexpected member {m.filename!r}")
            for m in members:
                if m.filename.startswith("assets/examples/refs/") and (ROOT / m.filename).exists():
                    continue  # media already here (from another archive, or placed by hand)
                z.extract(m, ROOT)
        if results:
            (EXAMPLES / "cache" / name).mkdir(parents=True, exist_ok=True)
            (EXAMPLES / "cache" / name / ".release").write_text(EXAMPLE_ARCHIVES[name] + "\n")
        return True
    except Exception as e:  # no network, no access, a broken archive
        print(f"[download] {file}: {e!r}", flush=True)
        return False


def results_wanted(name):
    """Whether the demo's stored results of `name` should be downloaded: none there yet, or those of an older archive
    (cache/<name>/.release names it). Results computed here (no .release file) are kept."""
    folder = EXAMPLES / "cache" / name
    tag = folder / ".release"
    if tag.is_file():
        return tag.read_text().strip() != EXAMPLE_ARCHIVES[name]
    return not any(folder.glob("*/meta.json"))


def ensure_examples(names=None, results=False, dry_run=False):
    """The media of the bundled examples of `names` (None = every model) into assets/examples/refs/ and, with
    `results`, the demo's stored example results into assets/examples/cache/, from the example archives on the Hub
    (`fetch_archive`). What cannot be downloaded is reported and skipped (the examples without media then fail alone)."""
    out = EXAMPLES / "refs"
    for name in names or NEEDS:
        needed = example_files([name])
        missing = [n for n in needed if not (out / n).exists()]
        wants = results and results_wanted(name)
        if missing:
            print(f"[download] example media of {name}: {len(missing)} of {len(needed)} files missing", flush=True)
        if wants:
            print(f"[download] stored example results of {name}", flush=True)
        if (missing or wants) and not dry_run and fetch_archive(name, wants):
            still = [n for n in needed if not (out / n).exists()]
            if still:
                print(f"[download] {len(still)} example files are not in the archive; their examples will not run",
                      flush=True)
    return out


def ensure(names, examples=True, dry_run=False, results=False, bf16_vlm=False):
    """Everything `names` (keys of NEEDS) read: the FP8 VLM first (`ensure_vlm_fp8`), model folders, LoRAs, the
    example media and, with `results`, the demo's stored example results. The bf16 text encoder of the releases
    (text_encoder/, 66 GB) is downloaded only with `bf16_vlm` (a run that reads the bf16 VLM) or when no FP8 VLM can
    be had (`vlm_fp8_source`): then it is converted at load, as before."""
    fp8 = ensure_vlm_fp8(dry_run) is not None or vlm_fp8_source()[0] is not None
    for name in names:
        for folder in NEEDS[name][0]:
            skip = list(SKIP.get(name, {}).get(folder, ()))
            if fp8 and not bf16_vlm and "text_encoder/*" in REPOS[folder][2]:
                skip.append("text_encoder/*")
            ensure_folder(folder, dry_run, skip)
        ensure_lora(name, dry_run)
    if examples:
        ensure_examples(names, results, dry_run)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help=f"any of {', '.join(NEEDS)} (default: every model)")
    p.add_argument("--no-examples", action="store_true", help="skip the media of the bundled examples")
    p.add_argument("--results", action="store_true", help="also the demo's stored example results")
    p.add_argument("--dry-run", action="store_true", help="print what would be downloaded and its size")
    p.add_argument("--bf16-vlm", action="store_true", help="also the bf16 text encoder of the releases (66 GB), which "
                   "the teacher of --mode baseline reads; default: the FP8 VLM only, the bf16 one when that fails")
    args = p.parse_args()
    unknown = [m for m in args.models if m not in NEEDS]
    if unknown:
        p.error(f"unknown model {unknown[0]!r}: one of {', '.join(NEEDS)}")
    ensure(args.models or list(NEEDS), not args.no_examples, args.dry_run, args.results, args.bf16_vlm)


if __name__ == "__main__":
    sys.exit(main())
