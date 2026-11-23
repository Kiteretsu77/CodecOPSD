"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Everything the command line reads, downloaded on first use into `models/`: the released models from the Hugging Face
Hub at pinned revisions and the CodecOPSD LoRAs.

    python -m codecopsd.download                       # both models
    python -m codecopsd.download minimax_h3_image      # one model
    python -m codecopsd.download --dry-run             # what would be downloaded, and its size

Older layouts keep working: a model folder at the repository root (`MiniMax-H3-Ref2VA/`) or a LoRA in `weights/` is
used when `models/` has none. MiniMax-H3 and MiniMax-H3-Image ship the same text encoder (63 GB): it is downloaded once
and linked into the second folder when the Hub files are identical.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"

_META = ["processor/*", "tokenizer/*", "scheduler/*", "model_index.json", "modular_model_index.json"]
# folder under models/: (Hub repo, pinned revision, files). MiniMax-H3-Ref2VA/ is MiniMaxAI/MiniMax-H3 with only its
# Ref2VA transformer (transformer_ref/), its VAE and its text encoder.
REPOS = {
    "MiniMax-H3-Ref2VA": ("MiniMaxAI/MiniMax-H3", "42ed227ee7df40d41602854ae760620d6eb651fe",
                          ["transformer_ref/*", "vae/*", "text_encoder/*"] + _META),
    "Pika-H3-Flex": ("Pika-Labs/Pika-H3-Flex", "0ddf55ad7067ca1e76812253bd7b4600df87f759", ["ref2va/adapter.safetensors"]),
    "MiniMax-H3-Image": ("MiniMaxAI/MiniMax-H3-Image", "492fd8c4eca666f1085f57e30738c1e0d86b3439",
                         ["transformer_diffusers/*", "vae_diffusers/*", "text_encoder/*"] + _META),
}
# model: (folders it reads, its CodecOPSD LoRA file)
NEEDS = {
    "minimax_h3_ref2va": (["MiniMax-H3-Ref2VA", "Pika-H3-Flex"], "codecopsd_minimax_h3_ref2va.safetensors"),
    "minimax_h3_image": (["MiniMax-H3-Image"], "codecopsd_minimax_h3_image.safetensors"),
}
# TODO: the CodecOPSD LoRAs on the Hugging Face Hub; until then they are placed by hand in models/
LORA_REPO = None


def model_dir(folder):
    """models/<folder>, or <folder> at the repository root when only that exists (older layout). Pika H3 Flex sits
    next to the MiniMax-H3 folder in use (the pipeline looks for it there)."""
    if folder == "Pika-H3-Flex":
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


def ensure_folder(folder, dry_run=False):
    """Download `folder` (a key of REPOS) into models/ unless it is already there; a folder that lacks a subfolder
    a pattern names gets those files. Returns its path."""
    path = model_dir(folder)
    repo, revision, patterns = REPOS[folder]
    if path.is_dir() and any(path.iterdir()):
        patterns = [p for p in patterns if p.endswith("/*") and p.count("/") == 1 and not (path / p[:-2]).is_dir()]
        if not patterns:
            return path
    shared = _shared_text_encoder(folder) if "text_encoder/*" in patterns else None
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


def ensure_lora(name, dry_run=False):
    """The CodecOPSD LoRA of `name`: its path, or None (with a note) when it is not there and cannot be downloaded."""
    path = lora_path(name)
    if path.is_file():
        return path
    if LORA_REPO is None or dry_run:
        print(f"[download] CodecOPSD LoRA {path.relative_to(ROOT)}: not on the Hub yet; place it there by hand "
              "(without it, only the teacher runs)", flush=True)
        return None
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(LORA_REPO, path.name, local_dir=MODELS_DIR))


def ensure(names, dry_run=False):
    """Everything `names` (keys of NEEDS) read: model folders and LoRAs."""
    for name in names:
        for folder in NEEDS[name][0]:
            ensure_folder(folder, dry_run)
        ensure_lora(name, dry_run)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help=f"any of {', '.join(NEEDS)} (default: every model)")
    p.add_argument("--dry-run", action="store_true", help="print what would be downloaded and its size")
    args = p.parse_args()
    unknown = [m for m in args.models if m not in NEEDS]
    if unknown:
        p.error(f"unknown model {unknown[0]!r}: one of {', '.join(NEEDS)}")
    ensure(args.models or list(NEEDS), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
