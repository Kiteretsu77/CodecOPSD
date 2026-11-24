"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

`main(page)`: the options of a demo page, the first-use downloads, then the page served (`app.py` at the repository
root picks the page).
"""

import argparse
import threading
from pathlib import Path

from codecopsd.download import ensure, lora_path, model_dir

from ..hardware import announce, vlm_files_gb
from ..load import MODELS, vlm_notice, vlm_source
from . import state
from .cache import example_rows, request_key, upgrade_stored_refs, warm_light
from .pipelines import load, preload
from .requests import fill_cache
from .state import CACHE, EXAMPLE_KEYS, EXAMPLES, OUT, PIPES, STEPS, VIDEO_MODELS
from .style import CSS, HEAD, THEME
from .ui import build


class NoTransformStreams:
    """ASGI middleware: the page's live streams (text/event-stream: progress, results, the heartbeat) are marked
    `Cache-Control: no-transform`. A proxy that compresses responses (code-server's port proxy, used to reach the demo
    from VS Code) otherwise holds the small stream messages in its compression buffer: the page then sees no progress
    while a request runs, and its heartbeat stops, so it takes the connection for lost and reloads."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def sending(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                if any(k.lower() == b"content-type" and v.lower().startswith(b"text/event-stream") for k, v in headers):
                    cache = [v for k, v in headers if k.lower() == b"cache-control"]
                    headers = [(k, v) for k, v in headers if k.lower() != b"cache-control"]
                    value = b", ".join(cache) if cache else b"no-cache"
                    if b"no-transform" not in value.lower():
                        value += b", no-transform"
                    headers.append((b"cache-control", value))
                    message = {**message, "headers": headers}
            await send(message)

        return await self.app(scope, receive, sending)


def main(page):
    """Run the demo page `page`: dict(model ("minimax_h3_ref2va", "h3_flash_ref2va" or "minimax_h3_image"), title (browser tab),
    heading (one line under the summary), sizes and frames (video pages), size (a fixed output size), examples (the
    example folders a page reuses, in order), doc)."""
    state.PAGE = page
    name = page["model"]
    state.VIDEO = name if name in VIDEO_MODELS else state.VIDEO
    p = argparse.ArgumentParser(description=page["doc"], formatter_class=argparse.RawDescriptionHelpFormatter)
    folder = MODELS[name]["release"]  # the extreme page reads the same release folder and LoRA as the video page
    p.add_argument("--model", default=str(model_dir(folder)), help=f"the released {folder} folder")
    p.add_argument("--lora", default=str(lora_path(name)), help="the CodecOPSD LoRA")
    if name == "minimax_h3_ref2va":
        p.add_argument("--sampler", choices=["pika", "fasth3", "base"], default=None,
                       help="pika (default, 6 calls), fasth3 (8 calls) or base (the released 49 calls); README: "
                            "Acceleration")
    if name == "h3_flash_ref2va":  # the extreme path only (inference/extreme.py)
        from .. import extreme
        p.add_argument("--flash-weights", default=str(extreme.flash_weights()),
                       help="the ref2va_int8_convrot/ folder of VideoRebirthAI/hyperflow-h3-flash")
        p.add_argument("--gpu-memory", type=float, default=None,
                       help="the GPU memory in GB to plan for: 12, 16, 24 or 32 (default: the first GPU's own). It picks "
                            "what waits in CPU memory (README: Consumer GPUs); below the GPU's size the process is "
                            "also capped to it, CUDA context included")
        p.add_argument("--vae-encoder", choices=["taeh3", "released"], default="taeh3",
                       help="what encodes the references: taeh3 (default; the VAE encode stage about 50x faster, "
                            "inference/extreme.py) or the released VAE encoder")
        p.add_argument("--steps", type=int, choices=[4, 6, 8], default=4,
                       help="HyperFlow H3 Flash's sampling steps: 4 (default, the extreme page's), 6 or 8")
    offloads = ["auto", "pinned", "model", "none"] + (["stream"] if name == "minimax_h3_ref2va" else [])
    p.add_argument("--offload", choices=offloads, default="auto" if name == "h3_flash_ref2va" else "model",
                   help="auto (extreme page): every model resident from a 64 GB budget, else pinned; pinned: one "
                        "model on the GPU at a time from page-locked CPU memory, the text encoder streamed layer by "
                        "layer; model: one model on the GPU at a time (diffusers' offload); none: every model resident"
                        + ("; stream: the transformer streamed block by block (80 GB GPUs)" if name == "minimax_h3_ref2va" else ""))
    p.add_argument("--text-encoder", choices=["fp8", "bf16"], default="fp8", help="text encoder weights")
    p.add_argument("--no-download", action="store_true", help="do not download missing models and example media "
                   "(default: downloaded on first launch into models/ and assets/examples/refs/)")
    p.add_argument("--port", type=int, default=page.get("port", 7860))
    p.add_argument("--links", default="", help='links to the demo pages under the title: "label=url,label=self,label" (self: this page, '
                   "in bold; a label without a URL is shown as coming soon)")
    p.add_argument("--share", action=argparse.BooleanOptionalAction, default=True,
                   help="a public gradio.live link, printed at start (default; its page loads the Gradio frontend from "
                   "Gradio's CDN); --no-share: a local page only, behind your own port forward or tunnel, every file "
                   "comes from this server")
    p.add_argument("--build-cache", action="store_true", help="compute the results of every bundled example (Ours and "
                   "Teacher) that the demo shows when an example is clicked, then exit")
    p.add_argument("--fill-cache", action="store_true", help="compute the missing example results in the background "
                   "while the demo runs (requests wait for the example in progress)")
    args = state.ARGS = p.parse_args()
    if name == "h3_flash_ref2va":
        STEPS["flash"] = args.steps  # the live progress bar's count before the first run
    # First launch: the FP8 VLM (already converted, so the 66 GB bf16 one is neither downloaded nor read into CPU
    # memory; codecopsd.download.ensure_vlm_fp8), the models, the CodecOPSD LoRA (from its Hugging Face repository,
    # codecopsd.download.LORA_REPOS; every load is counted there), then the example media and the stored example
    # results (the example archive of that repository, out of git), unpacked into assets/examples/ before the demo
    # starts.
    if not args.no_download:
        try:
            ensure([name], results=True, bf16_vlm=args.text_encoder == "bf16")
        except Exception as e:  # e.g. no network: run with what is there
            print(f"[download] failed ({e!r}); continuing with the files already present", flush=True)
    # the VLM: the downloaded FP8 copy, else a saved FP8 VLM, else the bf16 release converted at load (load.vlm_source)
    vlm = vlm_source(args.text_encoder)
    args.vlm_fp8, args.stream_blocks, args.plan, args.pin_vlm = vlm["path"], 0, None, True
    if name == "h3_flash_ref2va":  # the memory cap, the plan, the VLM line, the hardware report (inference/extreme.py)
        run = extreme.setup(args.model, args.gpu_memory, args.offload, prefix="[app] ", vlm=vlm)
        args.gpu_memory, args.plan = run["budget"], run["plan"]  # the budget the page states
        args.offload, args.stream_blocks, args.pin_vlm = run["offload"], run["stream_blocks"], run["pin_vlm"]
    else:  # the hardware report and the warnings that apply (bf16 VLM files: mapped only when read)
        vlm_notice(vlm, args.offload, prefix="[app] ")
        announce(commit_gb=vlm_files_gb(args.model) if vlm["kind"] in ("converted", "bf16") else None)
    if not Path(args.model).is_dir():
        raise SystemExit(f"no model folder at {args.model}: run without --no-download (README: Quick Start) or pass "
                         "--model")
    OUT.mkdir(parents=True, exist_ok=True)
    if args.build_cache:
        for ex, _, values in example_rows(name):  # the keys that tie a request to its example
            EXAMPLE_KEYS[(name, request_key(values[:-1]))] = ex
        return fill_cache([name])
    if args.offload == "none":
        preload([name])
    load(name)  # loaded before the page opens: no request waits for the model to load
    demo = state.DEMO = build()
    if name == "minimax_h3_image" and name in PIPES:  # CPU only, a few seconds
        threading.Thread(target=upgrade_stored_refs, args=(name,), daemon=True).start()
    threading.Thread(target=warm_light, args=([name],), daemon=True).start()  # CPU: light copies of the examples
    if args.fill_cache:
        threading.Thread(target=fill_cache, args=([name],), daemon=True).start()
    from starlette.middleware import Middleware

    demo.queue(default_concurrency_limit=1).launch(app_kwargs={"middleware": [Middleware(NoTransformStreams)]},
                                                   server_name="0.0.0.0", server_port=args.port,
                                                   share=args.share, allowed_paths=[str(EXAMPLES), str(OUT), str(CACHE)],
                                                   theme=THEME, css=CSS, head=HEAD, max_file_size="1gb",
                                                   show_error=True)
