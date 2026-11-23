"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

What the demo process holds: its folders, the page it serves and its options (set once by `main`, read as `state.ARGS`
and so on), the pipelines, what each viewer's page shows, and the names the page gives the models.
"""

from ..examples import EXAMPLES, ROOT
from ..load import MODELS

OUT = ROOT / "outputs" / "app"  # every result, reference view and stats.jsonl
CACHE = EXAMPLES / "cache"  # the stored results of the bundled examples: <model>/<example>/

ARGS = None  # the parsed options of the page (main)
PAGE = None  # the page this app serves (main): dict(model, title, heading, sizes, frames, examples, ...)
DEMO = None  # the Blocks, to see whether the page of a running request is still open
VIDEO = "minimax_h3_ref2va"  # the model key of the video page: "minimax_h3_ref2va" (H3 + Pika H3 Flex) or "h3_flash_ref2va" (HyperFlow H3 Flash)

STATE = {"name": None, "pipe": None}  # the pipeline on the GPU, the denoise timer and the hooks of the running request
PIPES, PARKED = {}, {}  # --offload none: every pipeline, and its transformer's pinned CPU copy
VIEW = {}  # (session, model) -> example clicks so far; a request of that tab stops when it changes
PAGES = {}  # (session, model) -> the results on this viewer's page: {"key": their inputs, "fresh": {who: run},
#             "running": {who: request number}} (a Teacher run next to Ours of the same inputs keeps Ours, and so on)
EXAMPLE_KEYS = {}  # (model, request_key) -> example folder
STATS = []  # every finished run since the start: dict(page, who, example, stages {name: s}, total, peak, process)

STEPS = {"pika": 6, "fasth3": 8, "flash": 6, "base": 49, "minimax_h3_image": 6}  # transformer calls per request
LABELS = {name: m["label"] for name, m in MODELS.items()}
VIDEO_MODELS = ("minimax_h3_ref2va", "h3_flash_ref2va")
WHO = {"student": "Student", "released": "Teacher"}
NAME = {"student": "Student (CodecOPSD)", "released": "Teacher (every token)"}
SHORT = {"student": "Student", "released": "Teacher"}
SAMPLER = {"minimax_h3_ref2va": "MiniMax H3 with Pika H3 Flex, 6 steps",
           "h3_flash_ref2va": "HyperFlow H3 Flash (a compressed MiniMax H3), 4 steps",
           "minimax_h3_image": "MiniMax H3 Image, 6 steps"}  # both columns run this distilled model; the Student adds the LoRA


def budget_gib():
    """The GPU memory budget of this page in GB (the extreme page: 32), or None."""
    return getattr(ARGS, "gpu_memory", None)


def new_page():
    """The page state of a request that no viewer's page holds (`requests.page_for`)."""
    return {"key": None, "fresh": {}, "running": {}}
