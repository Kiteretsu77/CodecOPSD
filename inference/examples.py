"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The bundled examples, shared by the command line and the demos: `assets/examples/<model>/<name>/example.json` holds the
command-line flags of `generate.py` (paths relative to that folder), its label and its place on the demo page.
"""

import json
from pathlib import Path

from codecopsd.download import EXAMPLES_OF

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "assets" / "examples"
PATH_FLAGS = {"--prompt-file", "--image", "--reference"}
EXAMPLE_NAMES = {"deep_sea": "Deep sea: 3 pictures + 3 videos"}  # labels of examples without one in their example.json


# TODO: Be careful to check, when choosing a bundled example, whether its teacher fits in the 96 GB of an RTX PRO 6000,
# our main target GPU. If it runs out of memory there, record the teacher as OOM for that example.
def example_folder(name):
    """The folder of assets/examples/ that holds the examples of `name`: the extreme page reuses the video examples."""
    return EXAMPLES_OF.get(name, name)


def examples_of(name, chosen=None):
    """[(folder, example.json)] of the examples of `name`: every folder of assets/examples/<name>/ in their "order"
    (without those marked "demo": false, which only the command line runs), or the folders `chosen`, in that order."""
    def order(folder):  # "order" in example.json, else the place in EXAMPLE_NAMES
        spec = json.loads((folder / "example.json").read_text()) if (folder / "example.json").exists() else {}
        return spec.get("order", list(EXAMPLE_NAMES).index(folder.name) if folder.name in EXAMPLE_NAMES else 999), folder.name
    folders = ([EXAMPLES / example_folder(name) / ex for ex in chosen] if chosen is not None
               else sorted((EXAMPLES / example_folder(name)).glob("*"), key=order))
    out = []
    for folder in folders:
        spec = json.loads((folder / "example.json").read_text()) if (folder / "example.json").exists() else None
        if spec and (chosen is not None or spec.get("demo", True)):
            out.append((folder, spec))
    return out


def example_args(name, example):
    """The flags of a bundled example, with its file paths made absolute."""
    folder = EXAMPLES / example_folder(name) / example
    if not (folder / "example.json").exists():
        found = sorted(p.name for p in (EXAMPLES / example_folder(name)).glob("*") if (p / "example.json").exists())
        raise SystemExit(f"no example {example!r} for {name}; bundled: {', '.join(found)}")
    flags = json.loads((folder / "example.json").read_text())["args"]
    args = [str((folder / a).resolve()) if i and flags[i - 1] in PATH_FLAGS else a for i, a in enumerate(flags)]
    missing = [a for i, a in enumerate(args) if i and flags[i - 1] in PATH_FLAGS and not Path(a).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} example files missing (e.g. {missing[0]}): python -m codecopsd.download")
    return args


def flag_values(args, flag):
    """Every value given to `flag` in a command line (a repeatable flag such as --reference has several)."""
    return [args[i + 1] for i, a in enumerate(args[:-1]) if a == flag]


def flag_values_n(args, flag, n):
    """The n values after `flag` in a command line (e.g. --video-thresholds 5.0 1.0), or None."""
    return [args[args.index(flag) + 1 + k] for k in range(n)] if flag in args else None
