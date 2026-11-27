"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The page layout: the header, the inputs and Settings, the Student / Teacher carousel, the examples gallery (a click shows
the example with its stored results), and `build`, which puts the page of `state.PAGE` together.
"""

import html as h

import gradio as gr

from . import state
from .cache import cached, example_rows, example_size, gallery_items, page_examples, request_key, size_label
from .media import add_files, captioned, paths_of
from .outputs import shown, stopped, ui_outputs
from .requests import image_runner, video_runner
from .settings import (IMAGE_STRATEGIES, STRATEGIES, STRATEGY_NOTES, THRESHOLDS, VIDEO_DROP, VLM_MODES, VLM_SCALE,
                       codec_defaults, codec_view)
from .state import EXAMPLE_KEYS, NAME, PAGES, VIEW
from .stats import stats_summary
from .style import ICONS, INTRO, PAPER, SUMMARY

SIZES_IMAGE = ["2048 x 2048", "1536 x 1536", "1024 x 1024", "1536 x 2720", "2720 x 1536", "first picture's aspect"]


LINKED = {"Pika H3 Flex": "https://huggingface.co/Pika-Labs/Pika-H3-Flex",
          "HyperFlow H3 Flash": "https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash"}


def linked(text):
    """A heading, escaped, with the model names of LINKED as links to their model pages."""
    out = h.escape(text)
    for name, url in LINKED.items():
        out = out.replace(h.escape(name), f'<a href="{h.escape(url, quote=True)}" target="_blank" rel="noopener">{h.escape(name)}</a>')
    return out


def examples_gallery(demo, name, inputs, outputs, last, run_event):
    """Clickable example thumbnails. A click shows the example at once, in one step: it stops this viewer's running
    request of the tab, then puts the example's inputs and its stored Student / Teacher results on the page. Every click
    counts ("always_last": a click while the previous one is still loading is not dropped, the last one wins), and the
    example is found by the clicked thumbnail's caption (its index only as a fallback), so a click never lands on the
    wrong example or fails."""
    rows = example_rows(name)
    for ex, _, values in rows:
        EXAMPLE_KEYS[(name, request_key(values[:-1]))] = ex
    by_label = {label: i for i, (_, label, _) in enumerate(rows)}
    gallery = gr.Gallery(gallery_items(name, rows), label="Examples: click one to load it with its result; Generate "
                         "runs it again", columns=6, height="auto", allow_preview=False, object_fit="cover",
                         show_label=True, elem_classes="examples")
    demo.load(lambda: gallery_items(name, rows), None, gallery)  # results computed since the start show on a reload
    empty = (gr.update(value=None, label=NAME["student"]), gr.update(value=None, label=NAME["released"]), "")
    keep = [gr.update()] * (len(inputs) + len(outputs) + 1)  # a click that names no example: the page stays as it is

    def which(evt):
        """The row of the clicked thumbnail: by its caption ("<label>" or "<label> · N x faster"), else by its index."""
        value = evt.value if isinstance(evt.value, dict) else {}
        caption = str(value.get("caption") or "")
        label = caption.rsplit(" · ", 1)[0] if caption.endswith("x faster") else caption
        if label in by_label:
            return by_label[label]
        index = evt.index[0] if isinstance(evt.index, (list, tuple)) and evt.index else evt.index
        return index if isinstance(index, int) and 0 <= index < len(rows) else None

    def pick(evt: gr.SelectData, request: gr.Request):
        key = (request.session_hash, name)
        VIEW[key] = VIEW.get(key, 0) + 1  # a running request of this tab stops at its next step
        PAGES.pop(key, None)
        index = which(evt)
        if index is None:
            print(f"[app] example click not matched: index {evt.index!r}, value {str(evt.value)[:120]!r}", flush=True)
            return keep
        ex, _, values = rows[index]
        r = cached(name, ex)[1]
        values = [values[0], gr.update(value=captioned(values[1]), selected_index=None), *values[2:-1]]
        if not r:
            return [*values, *empty, "Not computed yet: press Generate.", None]
        return [*values, *ui_outputs(r), shown(r)]

    gallery.select(pick, None, inputs + outputs + [last], cancels=run_event, show_progress="hidden",
                   concurrency_limit=None, trigger_mode="always_last")
    return gallery


# ---------------------------------------------------------------------------------------------------------------- UI

def bind_runs(kind, go, mode, runner, inputs, outputs, student, released):
    """Two hidden events, one per model, so that the progress bar shows on the panel of the model that runs. Generate
    starts the model chosen in `mode` (Student or Teacher) and shows its page of the carousel. Returns the events (Stop
    and a click on an example cancel them)."""
    events = []
    with gr.Row(elem_classes="hidden-run"):
        for who, panel in (("released", released), ("student", student)):
            hidden = gr.Button(who, elem_id=f"run-{kind}-{who}")
            events.append(hidden.click(runner(who), inputs, outputs, concurrency_limit=None, show_progress_on=[panel]))
    go.click(None, [mode], None, js=f"""(m) => {{ const who = m === 'released' ? 'released' : 'student';
        window.carouselGo && window.carouselGo('carousel-{kind}', who === 'released' ? 1 : 0);
        document.getElementById('run-{kind}-' + who).click(); return []; }}""")
    return events


def mode_radio():
    """Which model Generate runs: the Student (CodecOPSD) or the Teacher (every token, slower)."""
    return gr.Radio([("Student (CodecOPSD)", "student"), ("Teacher (every token, slower)", "released")],
                    value="student", label="Run", elem_classes="run-mode")


RESULT_HEIGHT = {"video": 480, "image": 520}  # px: the result next to the inputs, a little larger than them


def result_area(kind, view):
    """The right column: a two-page carousel (the Student fills the first page; a swipe, the arrow or the second tab
    shows the Teacher; Side by Side shows both), the core line (condition tokens, denoising time, speed-up), then the
    references as the model reads them, the stage table and the setting in small type. `view(label, elem_id,
    elem_classes)`: the result component. Returns (student, released, info, refs)."""
    with gr.Row(elem_id=f"carousel-{kind}", elem_classes="carousel"):
        with gr.Column(min_width=0, elem_classes="slide"):
            student = view(NAME["student"], f"view-{kind}-student", "ours-panel")
        with gr.Column(min_width=0, elem_classes="slide"):
            released = view(NAME["released"], f"view-{kind}-released", "teacher-panel")
    chevron = lambda d: (f'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="{d}" fill="none" stroke="currentColor" '
                         'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>')
    gr.HTML(f'<div class="car-nav" data-target="carousel-{kind}">'
            f'<button class="car-arrow" data-step="-1" aria-label="previous page">{chevron("M15 5l-7 7 7 7")}</button>'
            f'<button class="car-tab ours on" data-go="0"><span class="car-dot"></span>Student (CodecOPSD)</button>'
            f'<button class="car-tab teacher" data-go="1"><span class="car-dot"></span>Teacher (every token)</button>'
            f'<button class="car-arrow" data-step="1" aria-label="next page">{chevron("M9 5l7 7-7 7")}</button>'
            f'<button class="car-side" aria-pressed="false">Side by Side</button></div>',
            padding=False)
    info = gr.Markdown(elem_classes="core-info")
    refs = gr.HTML("", padding=False)
    return student, released, info, refs


def intro_html():
    import html as h
    links = []
    for icon, label, url in PAPER["links"]:
        inner = ICONS[icon] + (f"<span>{h.escape(label)}</span>" if label else "")
        links.append(f'<a class="hdr-link" href="{h.escape(url or "#", quote=True)}"'
                     + (' target="_blank">' if url else ' title="link coming soon" onclick="return false">') + f"{inner}</a>")
    demos = []
    for item in filter(None, (x.strip() for x in (state.ARGS.links or "").split(","))):
        label, _, url = item.partition("=")
        label, url = h.escape(label.strip()), url.strip()
        demos.append(f'<span class="hdr-demo on">{label}</span>' if url == "self" else
                     f'<a class="hdr-demo" href="{h.escape(url, quote=True)}" target="_blank">{label}</a>' if url else
                     f'<span class="hdr-demo soon">{label} (coming soon)</span>')
    demos.append('<span class="hdr-demo soon">Other models (coming soon)</span>')
    # The star request, worded as in the README (same as our earlier repos); it links to the same URL as the Code button.
    code = next((url for icon, _, url in PAPER["links"] if icon == "github"), "")
    repo = (f'<a href="{h.escape(code, quote=True)}" target="_blank" rel="noopener">GitHub repo</a>' if code
            else "GitHub repo")
    return (INTRO + f'<div class="hdr-links">{"".join(links)}</div>'
            + f'<p class="intro-line">{"<br>".join(h.escape(line) for line in SUMMARY.split(chr(10)))}</p>'
            + f'<p class="page-head"><b>{linked(state.PAGE["heading"])}</b></p>'
            + f'<p class="star-line">⭐ <b>If you like CodecOPSD, please help ⭐⭐star⭐⭐ the {repo}. Thanks!</b> 🤗</p>'
            + f'<div class="hdr-demos"><span class="hdr-demos-label">Demos:</span>{"".join(demos)}</div></div>')


def video_page(demo):
    """The video page (`state.VIDEO`): references, prompt, Run (Student / Teacher), Generate / Stop, Settings, the
    Student / Teacher carousel, the
    examples."""
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=420):
            files = gr.Gallery(label="1. References: pictures and videos, in prompt order (<Picture 1>, ..., "
                                     "<Video 1>, ...)", type="filepath", interactive=True,
                               file_types=["image", "video"], columns=3, height=RESULT_HEIGHT["video"] + 20,
                               object_fit="contain", elem_classes="ref-input")
            files.upload(lambda v: captioned(paths_of(v)), files, files, show_progress="hidden")
            with gr.Row():
                upload = gr.UploadButton("Upload your pictures or videos", file_types=["image", "video"],
                                         file_count="multiple", size="sm")
                clear = gr.Button("Clear the references", size="sm")
            upload.upload(add_files, [files, upload], files, show_progress="hidden")
            clear.click(lambda: [], None, files, show_progress="hidden")
            prompt = gr.Textbox(label="2. Prompt", lines=6, max_lines=14, placeholder="Refer to the references "
                                "as <Picture 1>, <Video 1>, ...")
            mode = mode_radio()
            with gr.Row():
                go = gr.Button("Generate video", variant="primary", elem_id="go-video", scale=4)
                stop = gr.Button("Stop", variant="secondary", elem_id="stop-video", scale=1, min_width=90)
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    sizes = state.PAGE["sizes"]
                    size = gr.Dropdown([(size_label(v), v) for v in sizes], value=sizes[0],
                                       label="Output, height x width", interactive=len(sizes) > 1)
                    # the page's lengths plus every length a bundled example runs at (e.g. the 73 frames of a
                    # 3 s line-art clip), so an example's own length is always a choice when it is run again
                    lengths = sorted(set(state.PAGE["frames"]) | {example_size(state.VIDEO, spec["args"])[2]
                                                                  for _, spec in page_examples(state.VIDEO)})
                    frames = gr.Dropdown([(f"{f} ({f / 24:.1f} s)", f) for f in lengths], value=124,
                                         label="Frames at 24 fps")
                # Codec pruning of Ours (the Teacher reads every token); an example fills these with its
                # own setting, clearing or uploading references puts the defaults back (`codec_defaults`)
                strategy = gr.Radio(STRATEGIES, value="default", label="Codec pruning", elem_classes="codec-strategy")
                note = gr.Markdown(STRATEGY_NOTES["default"], elem_classes="codec-note")
                with gr.Group(visible=False) as share:  # shown with Priority drop
                    drop = gr.Slider(0.05, 0.95, value=VIDEO_DROP, step=0.01,
                                     label="Share of each reference video's tokens to drop")
                with gr.Group(visible=False) as thresholds:  # shown with Threshold
                    with gr.Row():
                        v_motion = gr.Number(THRESHOLDS["video"][0], label="Videos: motion threshold (px)",
                                             minimum=0.1, maximum=64, step=0.5)
                        v_residual = gr.Number(THRESHOLDS["video"][1], label="Videos: residual threshold",
                                               minimum=0.05, maximum=64, step=0.1)
                    with gr.Row():
                        p_motion = gr.Number(THRESHOLDS["image"][0], label="Pictures: motion threshold (px)",
                                             minimum=0.1, maximum=64, step=0.5)
                        p_residual = gr.Number(THRESHOLDS["image"][1], label="Pictures: residual threshold",
                                               minimum=0.05, maximum=64, step=0.1)
                with gr.Row(visible=False) as vlm_row:  # hidden with Default
                    vlm = gr.Dropdown(VLM_MODES, value="threshold", label="Text encoder (VLM) tokens")
                    vlm_scale = gr.Number(VLM_SCALE, label="VLM threshold scale", minimum=0.01, maximum=1.0,
                                          step=0.05)
                seed = gr.Number(42, label="Seed", precision=0, visible=False)  # no seed control on the page: an example keeps its own seed, other runs use 42
                audio = gr.Checkbox(False, label="Audio: generate the soundtrack with the video (off: silent, as the "
                                    "CodecOPSD LoRA was trained; on: same speed, a slightly different video)",
                                    visible=state.VIDEO == "h3_flash_ref2va"
                                    or getattr(state.ARGS, "sampler", None) == "base")
            codec = [strategy, drop, v_motion, v_residual, p_motion, p_residual, vlm, vlm_scale]
            # the panel shows the controls of the chosen strategy (also when an example sets it)
            for c in (strategy, vlm):
                c.change(codec_view, [strategy, vlm], [share, thresholds, vlm_row, vlm_scale, note],
                         show_progress="hidden")
            clear.click(lambda: codec_defaults(True), None, codec, show_progress="hidden")
            upload.upload(lambda: codec_defaults(True), None, codec, show_progress="hidden")
            files.upload(lambda: codec_defaults(True), None, codec, show_progress="hidden")
        with gr.Column(scale=6, min_width=480):
            video_view = lambda label, elem_id, cls: gr.Video(
                label=label, autoplay=True, loop=True, height=RESULT_HEIGHT["video"], buttons=["download"],
                elem_id=elem_id, elem_classes=cls)
            student, released, info, cells = result_area("video", video_view)
    last = gr.State(None)
    outputs = [student, released, cells, info]
    run_event = bind_runs("video", go, mode, video_runner, [prompt, files, size, frames, *codec, seed, audio],
                          outputs + [last], student, released)
    stop.click(stopped, last, [student, released, info, last], cancels=run_event, show_progress="hidden")
    examples_gallery(demo, state.VIDEO, [prompt, files, size, frames, *codec, seed, audio], outputs, last, run_event)


def image_page(demo):
    """The image page: reference pictures, prompt, Run (Student / Teacher), Generate / Stop, Settings, the Student /
    Teacher carousel, the
    examples."""
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=420):
            files = gr.Gallery(label="1. Reference pictures, in prompt order (<Picture 1>, <Picture 2>, ...)",
                               type="filepath", interactive=True, file_types=["image"], columns=3,
                               height=RESULT_HEIGHT["image"], object_fit="contain", elem_classes="ref-input")
            files.upload(lambda v: captioned(paths_of(v)), files, files, show_progress="hidden")
            with gr.Row():
                upload = gr.UploadButton("Upload your pictures", file_types=["image"], file_count="multiple",
                                         size="sm")
                clear = gr.Button("Clear the references", size="sm")
            upload.upload(add_files, [files, upload], files, show_progress="hidden")
            clear.click(lambda: [], None, files, show_progress="hidden")
            prompt = gr.Textbox(label="2. Prompt", lines=4, max_lines=12, placeholder="The woman in <Picture 1> "
                                "is <Subject 1>. The target image shows <Subject 1> ... <Subject 1> is fully "
                                "preserved.")
            mode = mode_radio()
            with gr.Row():
                go = gr.Button("Generate image", variant="primary", elem_id="go-image", scale=4)
                stop = gr.Button("Stop", variant="secondary", elem_id="stop-image", scale=1, min_width=90)
            with gr.Accordion("Settings", open=False):
                with gr.Row():
                    size = gr.Dropdown(SIZES_IMAGE, value="first picture's aspect", label="Output, height x width",
                                       allow_custom_value=True)
                # Codec pruning of Ours (the Teacher reads every token); an example fills these with its
                # own setting, clearing or uploading references puts the defaults back (`codec_defaults`)
                strategy = gr.Radio(IMAGE_STRATEGIES, value="default", label="Codec pruning",
                                    elem_classes="codec-strategy")
                note = gr.Markdown(STRATEGY_NOTES["default"], elem_classes="codec-note")
                with gr.Row(visible=False) as thresholds:  # shown with Threshold
                    p_motion = gr.Number(THRESHOLDS["image"][0], label="Motion threshold (px)", minimum=0.1,
                                         maximum=64, step=0.5)
                    p_residual = gr.Number(THRESHOLDS["image"][1], label="Residual threshold", minimum=0.05,
                                           maximum=64, step=0.1)
                with gr.Row(visible=False) as vlm_row:  # hidden with Default
                    vlm = gr.Dropdown(VLM_MODES, value="threshold", label="Text encoder (VLM) tokens")
                    vlm_scale = gr.Number(VLM_SCALE, label="VLM threshold scale", minimum=0.01, maximum=1.0,
                                          step=0.05)
                seed = gr.Number(42, label="Seed", precision=0, visible=False)  # no seed control on the page: an example keeps its own seed, other runs use 42
            codec = [strategy, p_motion, p_residual, vlm, vlm_scale]
            # the panel shows the controls of the chosen strategy (also when an example sets it)
            for c in (strategy, vlm):
                c.change(lambda st, v: (gr.update(visible=st == "threshold"), *codec_view(st, v)[2:]),
                         [strategy, vlm], [thresholds, vlm_row, vlm_scale, note], show_progress="hidden")
            clear.click(lambda: codec_defaults(False), None, codec, show_progress="hidden")
            upload.upload(lambda: codec_defaults(False), None, codec, show_progress="hidden")
            files.upload(lambda: codec_defaults(False), None, codec, show_progress="hidden")
        with gr.Column(scale=6, min_width=480):
            image_view = lambda label, elem_id, cls: gr.Image(
                label=label, type="pil", height=RESULT_HEIGHT["image"], buttons=["download", "fullscreen"],
                elem_id=elem_id, elem_classes=cls)
            student, released, info, cells = result_area("image", image_view)
    last = gr.State(None)
    outputs = [student, released, cells, info]
    run_event = bind_runs("image", go, mode, image_runner, [prompt, files, size, *codec, seed],
                          outputs + [last], student, released)
    stop.click(stopped, last, [student, released, info, last], cancels=run_event, show_progress="hidden")
    examples_gallery(demo, "minimax_h3_image", [prompt, files, size, *codec, seed], outputs, last, run_event)


def build():
    with gr.Blocks(title=state.PAGE["title"]) as demo:
        gr.HTML(intro_html())
        (image_page if state.PAGE["model"] == "minimax_h3_image" else video_page)(demo)
        with gr.Accordion("Time per stage over every request since this demo started", open=False):
            summary = gr.Markdown(stats_summary)
            gr.Button("Refresh", size="sm").click(stats_summary, None, summary, show_progress="hidden")
    return demo
