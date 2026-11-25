"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

One request of the page: its inputs checked, then Ours and / or the Teacher, each alone on the GPU (`run_one`),
yielding the page outputs after each (`run_video_raw`, `run_image_raw`); the Gradio events that drive them
(`video_runner`, `image_runner`), and `fill_cache`, which computes the results the bundled examples do not have yet.
"""

import gc
import re
import time
from pathlib import Path

import gradio as gr
import torch
from diffusers.utils import export_to_video
from PIL import ImageOps

from . import state
from .cache import example_rows, remember, request_key, finish, computed, read_meta
from .inputs import check_tags, open_picture, open_reference, parse_seed, parse_size, source_of
from .media import display_copy, input_paths
from .outputs import page_outputs
from .pipelines import has_lora, load, set_lora
from .progress import GPU, Refused, Report, Superseded, drive, view_of
from .settings import parse_codec
from .state import EXAMPLE_KEYS, OUT, PAGES, STATE, STEPS, VIDEO_MODELS, VIEW, WHO, budget_gib, new_page
from .stats import CONTEXT_GIB, PAGE_ROWS, finish_stats
from .views import draw_references, read_references


# TODO: Be careful to check whether the teacher (Run: Teacher, every reference token) fits in the 96 GB of an RTX PRO
# 6000, our main target GPU, for every sample we choose. If it runs out of memory there, record the teacher as OOM for
# that sample.
def run_one(name, who, report, call, steps_key):
    """One model's run of a request, alone on the GPU (other requests may run between the Student and the Teacher).
    Returns dict(result, denoise, total, calls, refs, drop)."""
    report.check()
    asked = time.perf_counter()
    with GPU.use(report):
        report.check()
        waited = time.perf_counter() - asked
        report.live(WHO[who])  # the live progress line, until the result is on the page
        pipe = load(name, report)
        if who == "student" and not has_lora(pipe):
            raise gr.Error(f"no CodecOPSD LoRA at {state.ARGS.lora} (README: Model Zoo)")
        pipe._last_encode = pipe.timings = pipe.clock = None  # clock: the live line reads this call's only
        report.pipe = pipe
        set_lora(pipe, who == "student")
        STATE.update(denoise=0.0, calls=0, work=dict(enc=[0, 0], dec=[0, 0]))  # the progress bar's counts (`count_work`)
        key = steps_key(pipe)
        result = None
        try:
            report.start(WHO[who], STEPS.get(key, 6))
            resident = torch.cuda.memory_allocated() / 2**30
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            try:
                result = call(pipe)
            except torch.OutOfMemoryError:
                pass  # the frames of the failed call are freed when this block ends
            total = time.perf_counter() - t
        finally:
            report.end()
            set_lora(pipe, True)
        if result is None:
            pipe._last_encode = None
            gc.collect()
            torch.cuda.empty_cache()
            if budget_gib():  # a page with a GPU memory budget says so under the outputs, the other result stays
                report.log(f"{WHO[who]}: out of GPU memory within the {budget_gib():g} GB budget")
                return dict(oom=True, result=None, denoise=None, total=total, calls=0, refs=[], drop=None,
                            gpu=torch.cuda.get_device_name())
            raise gr.Error("Out of GPU memory: use fewer or smaller references, fewer frames or a smaller output.")
        denoise, calls = STATE["denoise"], STATE["calls"]  # read before the next request resets them
        clock = getattr(pipe, "clock", None)
        if clock is not None:
            clock.add_outside_moves()  # diffusers' model offload between the stages (`codecopsd.offload.time_model_offload`)
        peak = clock.peak_gib() if clock is not None else torch.cuda.max_memory_allocated() / 2**30
        process = (clock.peak_gib(reserved=True) if clock is not None else torch.cuda.max_memory_reserved() / 2**30) \
            + CONTEXT_GIB
        if clock is None:  # a pipeline without stage timing: the denoising and the rest
            from codecopsd.stats import Clock
            clock = Clock()
            clock.add("denoise", denoise)
            clock.add("references, text encoder, decode", max(0.0, total - denoise))
        from codecopsd.pipelines.minimax_h3_ref2va import stage_times
        stages = stage_times(getattr(pipe, "timings", None))
        STEPS[key] = calls
        seen = read_references(pipe, name) if who == "student" else []  # CPU copies; drawn after the GPU is free
        pipe._last_encode = None
    if seen:
        report.stage("Finish: drawing the kept tokens")
    t = time.perf_counter()
    refs, share = draw_references(seen)
    drawn = time.perf_counter() - t
    report.log(f"{WHO[who]}: denoise {denoise:.1f} s, total {total:.1f} s"
               + ("" if share is None else f", {1 - share:.0%} of the reference tokens dropped")
               + f", GPU memory {resident:.1f} GB resident, peak {peak:.1f} GB; {stages}; progress counts "
               + ", ".join(f"{k} {d}/{n}" for k, (d, n) in STATE["work"].items()))
    if seen:
        clock.add(PAGE_ROWS[0], drawn)
    return dict(result=result, denoise=denoise, total=total, calls=calls, refs=refs, clock=clock, waited=waited,
                peak=peak, process=process, drop=None if share is None else 1 - share, gpu=torch.cuda.get_device_name())


def page_for(request, name, key):
    """The results on this viewer's page of `name`. A request with other inputs than those results starts a new page
    state and stops this viewer's running request of the tab (its results would no longer match the inputs); a request
    with the same inputs keeps the other model's result on the page (the Teacher run after Ours, or Ours again)."""
    k = (getattr(request, "session_hash", None), name)
    page = PAGES.get(k)
    if page is None or page["key"] != key:
        if page is not None and page["running"]:
            VIEW[k] = VIEW.get(k, 0) + 1
        page = PAGES[k] = {"key": key, "fresh": {}, "running": {}}
    return page


def image_runner(mode):
    """The run function of one hidden Generate event: "student" (the Student) or "released" (the Teacher)."""
    async def run_image(prompt, files, size, strategy, picture_motion, picture_residual, vlm, vlm_scale, seed,
                        request: gr.Request, progress=gr.Progress()):
        codec = [strategy, picture_motion, picture_residual, vlm, vlm_scale]  # the Settings panel's codec values
        page = page_for(request, "minimax_h3_image",
                        ("minimax_h3_image", request_key([prompt, input_paths(files), size, *codec, seed])))
        report = Report(progress, view_of(request, "minimax_h3_image"), "an image request")
        try:
            async for r in drive(run_image_raw(prompt, files, size, *codec, seed, mode, report, page), report):
                yield page_outputs(r)
        finally:
            report.in_line = True
            report.quiet()
    return run_image


def run_image_raw(prompt, files, size, strategy, picture_motion, picture_residual, vlm, vlm_scale, seed, mode,
                  report=None, page=None):
    """Yields the outputs: at once (the other model's result of the same inputs stays), then after each model (with
    "both" the Teacher first); the last yield holds every result. The results of a request with the inputs of a bundled
    example are stored for it when it has none yet. `page`: `page_for` of the viewer (None: no page)."""
    from codecopsd.pipelines.minimax_h3_image.pipeline_minimax_h3_image import reference_size
    from codecopsd.pipelines.minimax_h3_ref2va import map_references
    report = report or Report()
    paths = input_paths(files)
    report.log(f"image, Run {mode}: {len(paths)} pictures {[Path(p).name for p in paths]}, size {size!r}, seed {seed}, "
               f"prompt {str(prompt)[:120]!r}")
    prompt = (prompt or "").strip()
    try:
        if not prompt:
            raise Refused("Write a prompt; refer to the pictures as <Picture 1>, <Picture 2>, ...")
        if not paths:
            raise Refused("Add at least one reference picture: this model generates from references.")
        seed = parse_seed(seed)
        codec = parse_codec(strategy, None, None, None, picture_motion, picture_residual, vlm, vlm_scale, video=False)
        images = map_references(open_picture, paths, range(1, len(paths) + 1))  # one thread per picture
        check_tags(prompt, len(images))
        if str(size).startswith("first"):
            height, width = reference_size(images[0])
        else:
            height, width = parse_size(size, 64, 4096, 4096 * 4096)
    except Refused as e:
        report.log(f"not run: {e}")
        yield {"refused": str(e)}
        return
    ex = EXAMPLE_KEYS.get(("minimax_h3_image", request_key([prompt, paths, size, strategy, picture_motion,
                                                             picture_residual, vlm, vlm_scale, seed])))
    order = ["released", "student"] if mode == "both" else [mode]  # Teacher first: the GPU warm-up falls on it
    runs, page = {}, page if page is not None else new_page()
    setting = (f"output {height} x {width}", codec["text"])
    sources = [f"{i.height} x {i.width}" for i in (ImageOps.exif_transpose(image) for image in images)]
    page["running"].update({w: report.id for w in order})
    out = lambda final: finish("minimax_h3_image", ex, runs, order, setting=setting, final=final, page=page, me=report.id)

    def call(who):
        def go(pipe):
            STATE["images"], STATE["sources"] = images, sources
            # Ours: codec pruning with this request's thresholds and VLM mode; the Teacher (gaze off) reads every token
            return pipe(prompt, images, height=height, width=width, gaze=who == "student", vlm_drop=codec["vlm"] != "off",
                        vlm_picture_scale="dit" if codec["vlm"] == "dit" else codec["vlm_scale"],  # "off": vlm_drop False
                        gaze_overrides=codec["overrides"],
                        generator=torch.Generator().manual_seed(seed)).images[0]
        return go

    try:
        yield out(False)
        for who in order:
            run = run_one("minimax_h3_image", who, report, call(who), lambda pipe: "minimax_h3_image")
            path = OUT / f"h3_image_{who}_{int(time.time())}"
            report.stage("Finish: writing the image files")
            t = time.perf_counter()
            run["result"].save(path.with_suffix(".png"))  # the full result; the page shows a 1536 px JPEG copy
            display_copy(run["result"], path.with_suffix(".jpg"))
            run["clock"].add("write the image files", time.perf_counter() - t)
            finish_stats(run, "minimax_h3_image", who, ex)
            remember("minimax_h3_image", ex, who, run)
            run["result"] = str(path.with_suffix(".jpg"))
            runs[who] = page["fresh"][who] = run
            release(page, who, report.id)
            report.quiet()
            yield out(who == order[-1])
    except Superseded:
        report.log("stopped")
    finally:
        report.quiet()
        for who in order:
            release(page, who, report.id)


def release(page, who, me):
    """`who` is no longer run by request `me` on this page."""
    if page["running"].get(who) == me:
        page["running"].pop(who)


def video_runner(mode):
    """The run function of one hidden Generate event: "student" (the Student) or "released" (the Teacher)."""
    async def run_video(prompt, files, size, frames, strategy, video_drop, video_motion, video_residual, picture_motion,
                        picture_residual, vlm, vlm_scale, seed, audio, request: gr.Request, progress=gr.Progress()):
        # the Settings panel's codec pruning values, in the order of `parse_codec`
        codec = [strategy, video_drop, video_motion, video_residual, picture_motion, picture_residual, vlm, vlm_scale]
        page = page_for(request, state.VIDEO, (state.VIDEO, request_key([prompt, input_paths(files), size, frames, *codec,
                                                                         seed, bool(audio)])))
        report = Report(progress, view_of(request, state.VIDEO), "a video request")
        try:
            async for r in drive(run_video_raw(prompt, files, size, frames, *codec, seed, bool(audio), mode, report,
                                               page), report):
                yield page_outputs(r)
        finally:
            report.in_line = True
            report.quiet()
    return run_video


def run_video_raw(prompt, files, size, frames, strategy, video_drop, video_motion, video_residual, picture_motion,
                  picture_residual, vlm, vlm_scale, seed, audio, mode, report=None, page=None):
    """Yields the outputs: at once (the other model's result of the same inputs stays), then after each model (with
    "both" the Teacher first); the last yield holds every result. The results of a request with the inputs of a bundled
    example are stored for it when it has none yet. `page`: `page_for` of the viewer (None: no page). `audio`: the
    Settings switch, the soundtrack generated with the video and put into the mp4 (part of the request, so a request
    with audio never shows or stores an example's silent result)."""
    from codecopsd.pipelines.minimax_h3_ref2va import map_references
    report = report or Report()
    paths = input_paths(files)
    report.log(f"video, Run {mode}: {len(paths)} references {[Path(p).name for p in paths]}, size {size!r}, "
               f"{frames} frames, codec {strategy} {video_drop} {video_motion}/{video_residual} "
               f"{picture_motion}/{picture_residual} vlm {vlm} {vlm_scale}, seed {seed}, audio {bool(audio)}, "
               f"prompt {str(prompt)[:120]!r}")
    prompt = (prompt or "").strip()
    try:
        if not prompt:
            raise Refused("Write a prompt; refer to the references as <Picture 1>, ..., <Video 1>, ...")
        if not paths:
            raise Refused("Add at least one reference picture or video: this model generates from references.")
        seed = parse_seed(seed)
        height, width = parse_size(size, 256, 1568, 672 * 1568, multiple=32)
        if state.PAGE.get("size") and f"{height} x {width}" not in state.PAGE["sizes"]:  # a page with a GPU memory budget
            raise Refused(f"This page generates {' or '.join(state.PAGE['sizes'])} only (its GPU memory budget).")
        try:
            frames = int(frames)
        except (TypeError, ValueError):
            frames = -1
        top = max(state.PAGE["frames"])
        if not 22 <= frames <= top or (frames - 5) % 17:
            raise Refused(f"Frames: 17n + 5, from 22 to {top} (the Settings offer "
                           f"{', '.join(map(str, state.PAGE['frames']))}).")
        codec = parse_codec(strategy, video_drop, video_motion, video_residual, picture_motion, picture_residual, vlm,
                            vlm_scale)
        references = map_references(lambda path, i: open_reference(path, i, frames), paths, range(1, len(paths) + 1))
        kinds = [r.kind for r in references]
        check_tags(prompt, kinds.count("image"), kinds.count("video"), video_tab=True)
    except Refused as e:
        report.log(f"not run: {e}")
        yield {"refused": str(e)}
        return
    audio = bool(audio)
    ex = EXAMPLE_KEYS.get((state.VIDEO, request_key([prompt, paths, size, frames, strategy, video_drop, video_motion,
                                                       video_residual, picture_motion, picture_residual, vlm, vlm_scale,
                                                       seed, audio])))
    order = ["released", "student"] if mode == "both" else [mode]  # Teacher first: the GPU warm-up falls on it
    runs, page = {}, page if page is not None else new_page()
    sources = map_references(source_of, paths, references)
    asked = re.fullmatch(r"\s*(\d+)\s*[x×X*,]\s*(\d+)\s*", str(size))
    rounded = [] if (int(asked[1]), int(asked[2])) == (height, width) else [
        f"Output {asked[1]} x {asked[2]} rounded to {height} x {width} (multiples of 32)."]
    setting = (f"output {height} x {width}, {frames} frames{', with audio' if audio else ''}", codec["text"])
    page["running"].update({w: report.id for w in order})
    out = lambda final: finish(state.VIDEO, ex, runs, order, rounded, setting=setting, final=final, page=page,
                               me=report.id)

    def call(who):
        def go(pipe):
            STATE["sources"] = sources
            out = pipe(prompt, references, height=height, width=width, num_frames=frames,
                       # Ours: codec pruning with this request's setting (video_drop None = the threshold decision
                       # alone); the Teacher (gaze off) reads every token
                       gaze=who == "student", video_drop=codec["video_drop"], gaze_overrides=codec["overrides"],
                       vlm_drop={"dit": "dit", "off": False}.get(codec["vlm"], codec["vlm_scale"]),  # the mode the setting resolved
                       generator=torch.Generator().manual_seed(seed), **({"audio": True} if audio else {}),
                       **({"num_inference_steps": state.ARGS.steps} if getattr(pipe, "sampler", None) == "flash" else {}))
            return (out.frames[0], out.audio, out.sampling_rate) if audio else out.frames[0]
        return go

    try:
        yield out(False)
        for who in order:
            run = run_one(state.VIDEO, who, report, call(who), lambda pipe: getattr(pipe, "sampler", "base"))
            if run.get("oom"):  # a run past the GPU memory budget: said under the outputs, the other run stays
                remember(state.VIDEO, ex, who, run)
                runs[who] = page["fresh"][who] = run
                release(page, who, report.id)
                report.quiet()
                yield out(who == order[-1])
                continue
            path = str(OUT / f"{state.VIDEO}_{who}_{int(time.time())}.mp4")
            report.stage("Finish: writing the video file")
            t = time.perf_counter()
            if audio:  # the frames, then the soundtrack muxed in (AAC)
                from codecopsd.pipelines.minimax_h3_ref2va import add_audio_track
                frames_out, soundtrack, rate = run["result"]
                export_to_video(frames_out, path, fps=24)
                add_audio_track(path, soundtrack, rate)
            else:
                export_to_video(run["result"], path, fps=24)
            run["clock"].add("write the video file (mp4)", time.perf_counter() - t)
            run["result"] = path
            finish_stats(run, state.VIDEO, who, ex)
            remember(state.VIDEO, ex, who, run)
            runs[who] = page["fresh"][who] = run
            release(page, who, report.id)
            report.quiet()
            yield out(who == order[-1])
    except Superseded:
        report.log("stopped")
    finally:
        report.quiet()
        for who in order:
            release(page, who, report.id)


def fill_cache(models):
    """Compute the results the examples do not have yet, one request at a time like any other: the pictures first (Ours
    and Teacher), then the videos with Ours, then the videos with the Teacher."""
    plan = [("minimax_h3_image", ("student", "released"))] if "minimax_h3_image" in models else []
    plan += [(m, w) for m in VIDEO_MODELS if m in models for w in (("student",), ("released",))]
    for name, whos in plan:
        run = run_video_raw if name in VIDEO_MODELS else run_image_raw
        for ex, label, values in example_rows(name):
            needed = [w for w in whos if not computed(read_meta(name, ex), w)]
            if not needed:
                continue
            mode = "both" if len(needed) == 2 else needed[0]
            note = f"the example '{label}' ({' and '.join(WHO[w] for w in needed)}), computed in advance"
            print(f"[cache] run {name} {ex} {mode}", flush=True)
            try:
                for _ in run(*values[:-1], mode, Report(what=note)):
                    pass
            except Exception as e:  # keep going with the next example
                print(f"[cache] FAILED {name} {ex}: {e!r}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            time.sleep(3)  # a request that waits for the GPU gets it first
    print("[cache] every example computed", flush=True)
