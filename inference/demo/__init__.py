"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

The Gradio demo, one page per process (`app.py` at the repository root picks it), one module per part:

    main.py        the page's options, the first-use downloads, `main(page)`
    state.py       what the process holds: folders, options, pipelines, each viewer's page, model names
    pipelines.py   the pipelines of the page (built with `inference.load`), the Teacher as the LoRA switched off
    progress.py    one request at a time on the GPU, the live progress line, stopping a request
    requests.py    one request: inputs checked, Ours and / or the Teacher, the page outputs after each
    inputs.py      the checks of the inputs (size, seed, prompt tags, reference files)
    settings.py    the codec pruning setting of the Settings panel
    outputs.py     what the page shows of a request (results, core line, setting)
    views.py       the references as the model read them, kept tokens in colour
    stats.py       where the seconds went, stage by stage
    media.py       light copies of the bundled media, captions of the reference list
    cache.py       the bundled examples on the page and their stored results
    ui.py          the page layout and `build`
    style.py       the header text, theme, CSS, page script and icons

One request uses the GPU at a time; the others wait in line, in arrival order, and see what runs. A request stops at
its next denoise step when its page presses Stop, picks another example, or is closed or reloaded. Outputs are also
written to outputs/app/; every request is logged on stdout.
"""

import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # the page and the server call no outside service

from .main import main  # noqa: E402

__all__ = ["main"]
