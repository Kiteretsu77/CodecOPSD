"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

One-line inference on your own inputs (run from the repository root).

    python inference/generate.py minimax_h3_ref2va --prompt-file prompt.txt --reference a.png --reference b.mp4
    python inference/generate.py minimax_h3_ref2va --mode baseline ...     # the teacher: every token, 49 calls
    python inference/generate.py minimax_h3_ref2va --mode extreme ...      # the most aggressive token drop
    python inference/generate.py minimax_h3_image --prompt "..." --image a.png --image b.png --out out.png
    python inference/generate.py minimax_h3_ref2va --help                  # every flag of a model

The released model is downloaded on first use into `models/` (README: Inference; `--no-download` turns it off);
`--model` / `--lora` override the paths. Every other flag goes to the model's command line (`inference/cli/<model>.py`).
"""

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    from inference.load import MODELS  # first: `inference` sets the CUDA allocator option before torch

    from codecopsd.download import ensure, lora_path, model_dir

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in MODELS:
        raise SystemExit(__doc__ + f"\nmodels: {', '.join(MODELS)}")
    name, user = argv[0], argv[1:]
    cfg = MODELS[name]
    cli = importlib.import_module(f"inference.cli.{cfg['cli']}")
    if any(a in ("-h", "--help") for a in user):
        return cli.main(["--help"])
    download = "--no-download" not in user
    user = [a for a in user if a != "--no-download"]
    mode = user[user.index("--mode") + 1] if "--mode" in user else "compress"
    if download and "--model" not in user:
        ensure([name])  # first use: the released model and the LoRA
    defaults = ["--model", str(model_dir(cfg["release"]))]
    if mode != "baseline" and not any(a == "--lora" or a.startswith("--lora=") for a in user):
        lora = lora_path(name)
        if not lora.exists():
            raise SystemExit(f"{lora.relative_to(ROOT)} not found (README: Inference); --mode baseline runs the teacher")
        defaults += ["--lora", str(lora)]
    if not any(a == "--out" or a.startswith("--out=") for a in user):
        out = ROOT / "outputs" / f"{name}_{mode}{cfg['ext']}"
        out.parent.mkdir(exist_ok=True)
        defaults += ["--out", str(out)]
    # argparse keeps the last value of a flag: the user's flags come last and win
    flags = defaults + user
    print("[inference]", name, " ".join(flags), flush=True)
    cli.main(flags)


if __name__ == "__main__":
    sys.path[0] = str(ROOT)  # run as a file: the `inference` package from the repository root
    main()
