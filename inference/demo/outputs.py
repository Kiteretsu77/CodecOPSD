"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

What the page shows of a request: the Student / Teacher results with their labels, the core line (condition tokens,
denoising time, speed-up; precomputed results marked), the reference views, the stage table folded under them and the
setting (`outputs_of`), as Gradio updates (`ui_outputs`), and the outputs after Stop (`stopped`).
"""

import html

import gradio as gr
import torch

from . import state
from .media import fast_result
from .state import NAME, SAMPLER, SHORT, budget_gib
from .stats import stats_html
from .views import drop_note, refs_html


def outputs_of(runs, order, refs=None, extra=(), stored=(), model=None, setting=None, elsewhere=()):
    """The outputs so far. `runs`: every result on the page (this request's, earlier ones of this page with the same
    inputs, and results computed in advance: `stored`); `order`: the models this request runs, the ones still to run
    marked running / next; `elsewhere`: models another request of this page is running on the same inputs. The labels
    name the models; under the results: the tokens the Student drops and the speed-up of the denoising (the
    transformer's steps, the part CodecOPSD speeds up), Teacher values in orange, Student values in blue; then the
    reference views, the stage table (folded) and the setting. `refs`: the reference views of the Student (default:
    those of the Student in `runs`)."""
    pending = [w for w in order if w not in runs]
    oom = {w for w, r in runs.items() if r.get("oom")}
    runs = {w: r for w, r in runs.items() if w not in oom}
    labels = {}
    for w in ("student", "released"):
        if w in oom:
            labels[w] = f"{NAME[w]} · out of GPU memory within the {budget_gib():g} GB budget"
        elif w in runs:
            labels[w] = NAME[w] + (" · precomputed" if w in stored else "")
        elif w in pending:
            labels[w] = NAME[w] + (" · running" if w == pending[0] else " · next")
        elif w in elsewhere:
            labels[w] = NAME[w] + " · running"
        else:
            labels[w] = NAME[w] + (" · not run for these inputs" if runs or pending else "")
    here = torch.cuda.get_device_name() if torch.cuda.is_available() else "this GPU"
    gpus = {w: (r.get("gpu") or here).replace("NVIDIA ", "") for w, r in runs.items()}  # where each time was measured
    compare = len(set(gpus.values())) <= 1
    s, t = runs.get("student"), runs.get("released")
    gpu = lambda w: f"1x {gpus[w]}"
    if refs is None:
        refs = (s or {}).get("refs") or []
    counted = [r for r in refs if r.get("total")]
    color = lambda w, text: f'<span class="v-{w}">{text}</span>'  # Teacher values orange, Student values blue
    parts = []  # ONE core line: condition tokens, then the denoising time (the part CodecOPSD speeds up)
    if s and counted:
        total, kept = sum(r["total"] for r in counted), sum(r["kept"] for r in counted)
        parts.append(f"Condition tokens: {color('released', thousands(total))} → {color('student', thousands(kept))} "
                     f"{drop_note(total, kept)}")
    elif s and s.get("drop"):
        parts.append(f"Condition tokens: {color('student', format(s['drop'], '.0%'))} dropped")
    if s and t and compare:
        od, td = round(s["denoise"], 1), round(t["denoise"], 1)
        parts.append(f"denoising: {color('released', f'{td:.1f} s')} → {color('student', f'{od:.1f} s')} "
                     f'({gpu("student")}), <span class="speedup">{td / od:.1f}x faster</span>')
    else:
        parts += [f"denoising ({SHORT[w]}): {color(w, format(runs[w]['denoise'], '.1f') + ' s')} ({gpu(w)})"
                  for w in ("student", "released") if w in runs]
    if oom:
        parts += [f"{SHORT[w]}: out of GPU memory within the {budget_gib():g} GB budget" for w in sorted(oom)]
    lines = [f'<span class="core">{"; ".join(parts)}</span>'] if parts else []
    shown_stored = [w for w in ("student", "released") if w in stored and w in runs]
    if shown_stored:  # results computed in advance for a bundled example
        lines.insert(0, f'<span class="precomputed">Precomputed</span> '
                        f'<span class="precomputed-note">{" and ".join(SHORT[w] for w in shown_stored)}, computed in '
                        "advance; Generate runs it again</span>")
    # the setting, at the bottom of the page in small type
    base, ours_setting = setting if isinstance(setting, tuple) else (setting, None)
    foot = [SAMPLER[model]] if model in SAMPLER else []
    foot += [base] if base else []
    if ours_setting and s:
        foot.append(ours_setting)
    if s and t and not compare:
        foot.append("the two times were measured on different GPUs")
    if budget_gib():
        plan = getattr(state.ARGS, "plan", None)
        foot.append(f"GPU memory budget {budget_gib():g} GB" + (f" ({plan['text']})" if plan else ""))
    note = ("Setting: " + "; ".join(foot) + ".") if foot else ""
    done = list(lines)
    if pending:
        lines.append(f"Running {NAME[pending[0]]} ...")
    return dict(student=fast_result((s or {}).get("result")), released=fast_result((t or {}).get("result")), refs=refs,
                setting=note, stats=stats_html(runs),
                info="  \n".join([*lines, *extra]), labels=labels, done="  \n".join(done))


def thousands(n):
    """A token count in thousands: 40184 -> "40.2k"."""
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def page_outputs(r):
    """The outputs of one yield of a request; a refused request leaves the outputs as they are and says why."""
    if "refused" in r:
        text = html.escape(r["refused"], quote=False)  # the warning and the Markdown read <Picture 1> as a tag otherwise
        gr.Warning(text)
        return gr.update(), gr.update(), gr.update(), f"**Not run**: {text}", gr.update()
    return (*ui_outputs(r), shown(r))


def shown(r):
    """What the Stop button needs of the outputs on the page: their labels and the finished lines of the info."""
    return {"labels": r.get("labels", {}), "done": r.get("done", "")}


def ui_outputs(r):
    """Gradio updates of (Student, Teacher, reference views + stage table + setting, core line). Both results are on
    the page (the carousel's two pages, or side by side)."""
    labels = r.get("labels", {})
    return (gr.update(value=r.get("student"), label=labels.get("student", NAME["student"])),
            gr.update(value=r.get("released"), label=labels.get("released", NAME["released"])),
            refs_html(r.get("refs"), r.get("setting")) + r.get("stats", ""),
            r.get("info") or "")


def stopped(last):
    """Stop: the outputs still marked running / next say stopped (the finished ones stay)."""
    if not last or not any("running" in l or "· next" in l for l in last["labels"].values()):
        return gr.update(), gr.update(), gr.update(), last
    halted = {w for w, l in last["labels"].items() if "running" in l or "· next" in l}
    labels = {w: (f"{NAME[w]} · stopped" if w in halted else l) for w, l in last["labels"].items()}
    done = "  \n".join(filter(None, [last["done"], "Stopped."]))
    return (*(gr.update(label=labels[w]) if w in labels else gr.update() for w in ("student", "released")), done,
            {**last, "labels": labels, "done": done})
