"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The bundled examples on the page and their stored results: the inputs and setting of each example (`example_rows`),
the results of a request with an example's exact inputs and setting, stored for it under
assets/examples/cache/<model>/<example>/ (`remember`, `store`), and the outputs of a click on an example (`cached`).
"""

import json
import re
import shutil
import threading
from pathlib import Path

import imageio.v3 as iio
import torch
from PIL import Image, ImageOps

from ..examples import EXAMPLE_NAMES, example_folder, examples_of, flag_values
from . import state
from .media import display_copy, fast_result, light_view, original, preview
from .outputs import outputs_of
from .settings import example_codec, parse_codec
from .state import CACHE, EXAMPLES, NAME, PIPES, VIDEO_MODELS, budget_gib, new_page
from .views import draw_references, gazed_pictures, stored_refs


def page_examples(name):
    """[(folder, example.json)] of the examples the page of `name` shows: the page's own list (`PAGE["examples"]`, in
    that order) of the folders it reuses, else every example of `name` in their "order"."""
    chosen = state.PAGE.get("examples") if state.PAGE and name == state.PAGE["model"] else None
    return examples_of(name, chosen)


# Every result of a request with the inputs of a bundled example is stored under assets/examples/cache/<model>/<example>/
# (one file per model + meta.json); a click on the example then shows it at once. `--fill-cache` computes the missing
# ones in the background after the demo opens, `--build-cache` computes them and exits.


def size_label(size):
    """The Settings label of an output size "H x W": "480p: 480 x 832", "480p portrait: 832 x 480"."""
    h, w = map(int, size.split(" x "))
    return f"{min(h, w)}p{' portrait' if h > w else ''}: {size}"


def example_size(name, args):
    """(height, width, frames) an example runs at on the page of `name`: its own flags; on a page with a GPU memory
    budget (`state.PAGE["size"]`) its own size only when the page offers it, else the page's size."""
    if state.PAGE and state.PAGE.get("size") and name == state.PAGE["model"]:
        own = f"{(flag_values(args, '--height') or [''])[0]} x {(flag_values(args, '--width') or [''])[0]}"
        size = tuple(map(int, own.split(" x "))) if own in state.PAGE["sizes"] else tuple(state.PAGE["size"])
        return (*size, int((flag_values(args, "--frames") or ["124"])[0]))
    return ((flag_values(args, "--height") or ["480"])[0], (flag_values(args, "--width") or ["832"])[0],
            int((flag_values(args, "--frames") or ["124"])[0]))


def example_rows(name):
    """[(folder name, label, input values in the order of the run function)] of the bundled examples of `name`."""
    rows = []
    for folder, spec in page_examples(name):
        a = spec["args"]
        seed = int((flag_values(a, "--seed") or ["42"])[0])
        if name in VIDEO_MODELS:
            h, w, frames = example_size(name, a)
            values = [(folder / flag_values(a, "--prompt-file")[0]).read_text().strip(),
                      [preview((folder / f).resolve()) for f in flag_values(a, "--reference")], f"{h} x {w}",
                      frames, *example_codec(spec), seed, False, "both"]  # False: audio off
        else:
            size = (f"{flag_values(a, '--height')[0]} x {flag_values(a, '--width')[0]}" if "--height" in a
                    else "first picture's aspect")
            values = [(folder / flag_values(a, "--prompt-file")[0]).read_text().strip(),
                      [preview((folder / f).resolve()) for f in flag_values(a, "--image")], size,
                      *example_codec(spec, video=False), seed, "both"]
        rows.append((folder.name, spec.get("label") or EXAMPLE_NAMES.get(folder.name, folder.name), values))
    return rows


def request_key(values):
    """What identifies a request: the prompt, the reference file names (Gradio keeps the name of every file it copies)
    and the settings, numbers written alike (124 == 124.0)."""
    def norm(v):
        try:
            return f"{float(v):g}"
        except (TypeError, ValueError):
            return str(v).strip()
    prompt, files, *settings = values
    names = tuple(Path(original(f[0] if isinstance(f, (list, tuple)) else f)).name for f in files or [])
    return (str(prompt).strip(), names, *(norm(v) for v in settings))


def read_meta(name, ex):
    """meta.json of an example's stored results ({} when none). Entries stored before the times were kept as numbers
    are read back from their label / info text (their total in whole seconds); Ours stored before the reference views
    had their numbers keeps its "cells" (`stored_refs`)."""
    path = CACHE / name / ex / "meta.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if meta.get("version") != 2:
        return {}
    for who in ("student", "released"):
        run = meta.get(who)
        if run and "denoise" not in run:
            label, info = run.get("label", ""), run.get("info", "")
            num = lambda pattern, text: float(re.search(pattern, text).group(1)) if re.search(pattern, text) else None
            run.update(denoise=num(r"denoise ([0-9.]+) s", label), total=num(r"([0-9.]+) s in total", info),
                       total_digits=0, drop=(num(r"drop ([0-9]+)%", label) or 0) / 100 if "drop" in label else None,
                       display=run["file"])
    return meta


CACHE_LOCK = threading.Lock()


def remember(name, ex, who, run):
    """Store one model's result of an example request when the example has none yet (`ex` None: not an example).
    Stored results stay as they are, so the times shown for an example do not change with every request."""
    with CACHE_LOCK:
        if ex is not None and not computed(read_meta(name, ex), who):
            store(name, ex, who, run)


def finish(name, ex, runs, order, extra=(), setting=None, final=True, page=None, me=None):
    """The outputs of a request (`me`: its number). Next to its own results: the other model's result of the same
    inputs from earlier on this page (`page_for`), the model another request of this page is running on them, and for
    a bundled example the stored result of a model not run (computed in advance), so the comparison stays on the
    page."""
    page = page or new_page()
    busy = set(runs) | (set() if final else set(order))
    earlier = {w: r for w, r in page["fresh"].items() if w not in busy}
    elsewhere = {w for w, rid in page["running"].items() if rid != me and w not in busy and w not in earlier}
    meta = read_meta(name, ex) if ex is not None else {}
    d = CACHE / name / str(ex)
    stored = {w: stored_run(meta, w, d) for w in ("student", "released")
              if w not in busy | set(earlier) | elsewhere and computed(meta, w)}
    return outputs_of({**stored, **earlier, **runs}, order, None, extra, stored=stored, model=name, setting=setting,
                      elsewhere=elsewhere)


def computed(meta, who):
    """Whether an example's stored results have `who`: a result, or a run out of GPU memory within the page's budget."""
    entry = meta.get(who, {})
    return entry.get("denoise") is not None or bool(entry.get("oom"))


def stored_run(meta, who, d):
    """A stored result as a run: its file, times and the GPU that measured them (older entries: the example's GPU),
    and for Ours its reference views. A stored out-of-memory run has no file."""
    if meta[who].get("oom"):
        return dict(meta[who])
    return {**meta[who], "gpu": meta[who].get("gpu") or meta.get("gpu", ""),
            "result": str(d / meta[who].get("display", meta[who]["file"])),
            **({"refs": stored_refs(meta, d)} if who == "student" else {})}


def store(name, ex, who, run):
    d = CACHE / name / ex
    d.mkdir(parents=True, exist_ok=True)
    meta = read_meta(name, ex) or {"version": 2}
    if run.get("oom"):  # out of GPU memory within the page's budget: stored as such, so the page says it at once
        meta[who] = dict(oom=True, denoise=None, total=run["total"], gpu=run.get("gpu"), budget_gb=budget_gib())
        (d / "meta.tmp").write_text(json.dumps(meta, indent=1))
        (d / "meta.tmp").replace(d / "meta.json")
        print(f"[cache] stored {name} {ex} {who}: out of GPU memory within {budget_gib():g} GB", flush=True)
        return
    result = run["result"]
    if name == "minimax_h3_image":
        file, display = f"{who}.png", f"{who}.jpg"
        result.save(d / file)
        display_copy(result, d / display)
    else:
        file = display = f"{who}.mp4"
        shutil.copy(result, d / file)
    meta[who] = dict(file=file, display=display, denoise=run["denoise"], total=run["total"], total_digits=1,
                     drop=run["drop"], calls=run["calls"], gpu=run.get("gpu") or torch.cuda.get_device_name(),
                     stats=run.get("stats"), process=run.get("process"), budget_gb=budget_gib())
    meta["gpu"] = torch.cuda.get_device_name()
    if who == "student":
        meta.pop("cells", None)
        meta["refs"] = []
        for r in run["refs"]:  # the reference views: their files next to meta.json
            r = dict(r)
            for k in ("tokens", "natural"):
                if r.get(k):
                    target = f"{r['kind']}{r['number']}_{k}{Path(r[k]).suffix}"
                    shutil.copy(r[k], d / target)
                    r[k] = target
            meta["refs"].append(r)
        if name == "minimax_h3_image":
            thumb = result.convert("RGB")
        else:
            frames = iio.imread(d / file, plugin="FFMPEG")
            spec = EXAMPLES / example_folder(name) / ex / "example.json"  # "thumb_frame": the gallery's frame
            pick = json.loads(spec.read_text()).get("thumb_frame") if spec.exists() else None
            thumb = Image.fromarray(frames[min(pick, len(frames) - 1) if pick is not None else len(frames) // 2])
        thumb.thumbnail((640, 640))
        thumb.save(d / "thumb.jpg", quality=88)
    (d / "meta.tmp").write_text(json.dumps(meta, indent=1))
    (d / "meta.tmp").replace(d / "meta.json")
    print(f"[cache] stored {name} {ex} {who}: denoise {meta[who]['denoise']:.1f} s", flush=True)


def example_setting(name, ex, d, meta):
    """(output size [video: and frames] with where the references come from (example.json "source";
    assets/examples/SOURCES.md has every file), the codec pruning setting of Ours) of an example's stored results."""
    spec_file = EXAMPLES / example_folder(name) / ex / "example.json"
    spec = json.loads(spec_file.read_text()) if spec_file.exists() else {}
    a = spec.get("args", [])
    src = f" (references: {spec['source']})" if spec.get("source") else ""
    if name in VIDEO_MODELS:
        h, w, frames = example_size(name, a)
        return f"output {h} x {w}, {frames} frames{src}", parse_codec(*example_codec(spec))["text"]
    strategy, pm, pr, vlm, scale = example_codec(spec, video=False)
    ours = parse_codec(strategy, None, None, None, pm, pr, vlm, scale, video=False)["text"]
    for who in ("student", "released"):  # the full-size result (PNG, or the full-size JPEG of a release) gives the size
        f = d / meta.get(who, {}).get("file", "")
        if f.suffix in (".png", ".jpg") and f.exists():
            with Image.open(f) as image:
                return f"output {image.height} x {image.width}{src}", ours
    return (f"output {flag_values(a, '--height')[0]} x {flag_values(a, '--width')[0]}{src}" if "--height" in a else None), ours


def cached(name, ex):
    """(thumbnail path or None, outputs or None) of the stored results of an example."""
    meta = read_meta(name, ex)
    whos = [w for w in ("student", "released") if computed(meta, w)]
    if not whos:
        return None, None
    d = CACHE / name / ex
    runs = {w: stored_run(meta, w, d) for w in whos}
    r = outputs_of(runs, whos, model=name, setting=example_setting(name, ex, d, meta), stored=whos)
    for who in ("student", "released"):
        if who not in whos:
            r["labels"][who] = f"{NAME[who]} · not computed yet"
    thumb = d / "thumb.jpg"
    return (str(thumb) if thumb.exists() else None), r


def upgrade_stored_refs(name):
    """Pictures of the image model's examples stored before the reference views had their numbers ("cells"): their
    views drawn again from the codec gazer alone (CPU; its masks depend on the pictures only, so they are the ones Ours
    read), checked against the stored drop and written next to the stored results, which stay as they are."""
    from concurrent.futures import ThreadPoolExecutor
    from codecopsd.codec_gazer import MINIMAX_H3, submit_gaze
    from codecopsd.pipelines.minimax_h3_image.pipeline_minimax_h3_image import CONFIG
    pipe = PIPES.get(name)
    if name != "minimax_h3_image" or pipe is None:
        return
    done = 0
    for ex, _, values in example_rows(name):
        meta = read_meta(name, ex)
        if "refs" in meta or not meta.get("student", {}).get("denoise"):
            continue
        try:
            images = [Image.open(path) for path in values[1]]
            pictures = gazed_pictures(pipe, images)
            with ThreadPoolExecutor(len(pictures)) as pool:
                futures = [submit_gaze(pool, torch.from_numpy(p).permute(2, 0, 1), CONFIG["references"]["gaze_kind"],
                                       MINIMAX_H3) for p in pictures]
                masks = [f.result().keep[0].numpy() for f in futures]
            sources = [f"{i.height} x {i.width}" for i in (ImageOps.exif_transpose(image) for image in images)]
            refs, share = draw_references([("picture", p, m, s) for p, m, s in zip(pictures, masks, sources)])
            if meta["student"].get("drop") is not None and abs((1 - share) - meta["student"]["drop"]) > 0.005:
                print(f"[cache] {name} {ex}: views not redrawn, drop {1 - share:.3f} vs stored "
                      f"{meta['student']['drop']:.3f}", flush=True)
                continue
            d = CACHE / name / ex
            with CACHE_LOCK:
                meta = read_meta(name, ex)
                meta["refs"] = []
                for r in refs:
                    target = f"{r['kind']}{r['number']}_tokens{Path(r['tokens']).suffix}"
                    shutil.copy(r["tokens"], d / target)
                    meta["refs"].append({**r, "tokens": target})
                meta.pop("cells", None)
                (d / "meta.tmp").write_text(json.dumps(meta, indent=1))
                (d / "meta.tmp").replace(d / "meta.json")
            done += 1
        except Exception as e:  # the stored "cells" stay
            print(f"[cache] {name} {ex}: views not redrawn ({e!r})", flush=True)
    print(f"[cache] reference views redrawn for {done} stored {name} examples", flush=True)


def gallery_items(name, rows):
    """Thumbnails (the result when computed, else the first reference) with the label and the speed-up."""
    items = []
    for ex, label, values in rows:
        thumb, r = cached(name, ex)
        meta = read_meta(name, ex)
        if meta.get("released", {}).get("oom") and meta.get("student", {}).get("denoise"):
            label = f"{label} · teacher out of memory"
        elif all(meta.get(w, {}).get("denoise") for w in ("student", "released")):
            label = f"{label} · {round(meta['released']['denoise'], 1) / round(meta['student']['denoise'], 1):.1f}x faster"
        elif meta.get("student", {}).get("teacher_denoise") and meta["student"].get("denoise"):
            # a precomputed (homepage) result: the Teacher's denoise seconds of the same run, no Teacher output stored
            s = meta["student"]
            label = f"{label} · {round(s['teacher_denoise'], 1) / round(s['denoise'], 1):.1f}x faster"
        items.append((thumb or values[1][0], label))
    return items


def warm_light(models):
    """Make the light copies of every bundled example in the background, so that the first clicks are light too."""
    for name in models:
        for _, _, values in example_rows(name):
            pass  # example_rows makes the input previews
        for d in sorted((CACHE / name).glob("*")):
            for f in sorted(d.glob("*.mp4")):
                if f.stem in ("student", "released"):
                    fast_result(f)
                elif not f.stem.endswith((".light", ".fast", ".part")):
                    light_view(f)
    print("[light] every example has its light copies", flush=True)
