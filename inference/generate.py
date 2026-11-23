"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

One-line inference on a bundled example, or on your own inputs (run from the repository root).

    python inference/generate.py minimax_h3_ref2va                     # MiniMax-H3 Ref2VA, compress mode -> outputs/
    python inference/generate.py minimax_h3_ref2va --mode baseline     # the teacher: every token, 49 calls
    python inference/generate.py minimax_h3_ref2va --mode extreme      # the most aggressive token drop
    python inference/generate.py h3_flash_ref2va                       # a 12-32 GB GPU: HyperFlow H3 Flash + the same LoRA, 480p
    python inference/generate.py minimax_h3_image                      # MiniMax-H3-Image, compress mode: line art to a photo
    python inference/generate.py minimax_h3_image --prompt "..." --image a.png --image b.png --out out.png
    python inference/generate.py minimax_h3_ref2va --help              # every flag of a model
    python inference/generate.py cosmos3 --model Cosmos3-Nano ...      # Cosmos 3 (Nano): every flag given

The released model and the example media are downloaded on first use (`models/`, `assets/examples/refs/`; README: Quick
Start; `--no-download` turns it off); `--model` / `--lora` override the paths. The text encoder (VLM) is downloaded
already converted to FP8 (`codecopsd.download.ensure_vlm_fp8`, also with `--model`); the 66 GB bf16 one is downloaded
and read only when that fails or the run reads the bf16 VLM (`--mode baseline`, `--text-encoder bf16`). Every other
flag goes to the model's command line (`inference/cli/<model>.py`). Without `--prompt` / `--prompt-file` a bundled
example runs (`assets/examples/<model>/<name>/example.json` = the model's flags, paths relative to that folder).
"""

import importlib
import sys
from pathlib import Path


def flag_value(flags, name, default=None):
    """The last value of `name` in `flags` ("--name value" or "--name=value"; argparse keeps the last), or `default`."""
    value = default
    for i, a in enumerate(flags):
        if a == name and i + 1 < len(flags):
            value = flags[i + 1]
        elif a.startswith(name + "="):
            value = a.split("=", 1)[1]
    return value


def main(argv=None):
    from inference.examples import ROOT, example_args  # first: `inference` sets the CUDA allocator option before torch
    from inference.load import MODELS

    from codecopsd.download import ensure, ensure_vlm_fp8, lora_path, model_dir

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "cosmos3":  # no bundled examples: every flag given (inference/cli/cosmos3.py)
        return importlib.import_module("inference.cli.cosmos3").main(argv[1:])
    if not argv or argv[0] not in MODELS:
        raise SystemExit(__doc__ + f"\nmodels: {', '.join(MODELS)}, cosmos3")
    name, user = argv[0], argv[1:]
    cfg = MODELS[name]
    cli = importlib.import_module(f"inference.cli.{cfg['cli']}")
    if any(a in ("-h", "--help") for a in user):
        return cli.main(["--help"])
    example = cfg["example"]
    if "--example" in user:
        i = user.index("--example")
        example, user = user[i + 1], user[:i] + user[i + 2:]
    download = "--no-download" not in user
    user = [a for a in user if a != "--no-download"]
    mode = user[user.index("--mode") + 1] if "--mode" in user else "compress"
    own_inputs = any(a in ("--prompt", "--prompt-file") or a.startswith("--prompt=") for a in user)
    # the VLM the run reads: bf16 for the teacher of --mode baseline or --text-encoder bf16, else FP8 (cli/<model>.py)
    bf16_vlm = flag_value(user, "--text-encoder", "bf16" if mode == "baseline" else "fp8") == "bf16"
    if download and "--model" not in user:
        ensure([name], examples=not own_inputs, bf16_vlm=bf16_vlm)  # first use: the FP8 VLM, the model, the media
    elif download and not bf16_vlm and flag_value(user, "--vlm-fp8") is None:
        ensure_vlm_fp8()  # your own release folder: the FP8 VLM still comes from models/ (cli/<model>.py --vlm-fp8)
    args = [] if own_inputs else example_args(name, example)
    defaults = ["--model", str(model_dir(cfg["release"]))] + cfg.get("args", [])
    if mode != "baseline" and not any(a == "--lora" or a.startswith("--lora=") for a in user):
        lora = lora_path(name)
        if not lora.exists():
            raise SystemExit(f"{lora.relative_to(ROOT)} not found (README: Quick Start); --mode baseline runs the teacher")
        defaults += ["--lora", str(lora)]
    if not any(a == "--out" or a.startswith("--out=") for a in user):
        out = ROOT / "outputs" / f"{name}_{'own' if own_inputs else example}_{mode}{cfg['ext']}"
        out.parent.mkdir(exist_ok=True)
        defaults += ["--out", str(out)]
    # argparse keeps the last value of a flag: the user's flags come last and win
    flags = defaults + args + user
    print("[inference]", name, " ".join(flags), flush=True)
    cli.main(flags)


if __name__ == "__main__":
    sys.path[0] = str(Path(__file__).resolve().parents[1])  # run as a file: the `inference` package from the repository root
    main()
