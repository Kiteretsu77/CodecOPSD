"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Where the seconds of every run went, stage by stage: under the results, in outputs/app/stats.jsonl, and summed over
every request in the panel at the bottom of the page.
"""

import html
import json
import re
import time

from .state import LABELS, OUT, SHORT, STATS, budget_gib


# Every run's seconds, stage by stage (`codecopsd.stats.Clock`: the pipeline's stages between GPU synchronizations, model
# moves in their own row, then what the page adds), shown under the results, appended to outputs/app/stats.jsonl, and
# summed over every request since the start in the "Time per stage" panel at the bottom of the page.

CONTEXT_GIB = 1.0  # the CUDA context and libraries on top of PyTorch's allocator, measured on an RTX PRO 6000


PAGE_ROWS = ("draw the kept tokens (page only)", "write the video file (mp4)", "write the image files")


def run_stats(run):
    """[(stage, seconds)] of a run: the pipeline's stages, the rest of the pipeline call as "other", then the page's
    own work after the call (`PAGE_ROWS`)."""
    rows = [(r["stage"], r["seconds"]) for r in run["clock"].rows]
    inside = [(n, v) for n, v in rows if n not in PAGE_ROWS]
    rest = run["total"] - sum(v for _, v in inside)
    if rest > 0.05:
        inside.append(("other (Python, CPU work between stages)", rest))
    return inside + [(n, v) for n, v in rows if n in PAGE_ROWS]


def finish_stats(run, name, who, ex):
    """Keep a finished run's stage rows on the run (`run["stats"]`, stored with an example's result) and log them."""
    import json
    run["stats"] = [[stage, round(seconds, 2)] for stage, seconds in run_stats(run)]
    entry = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), page=name, who=who, example=ex, gpu=run.get("gpu"),
                 budget_gb=budget_gib(), stages=run["stats"], total=round(sum(s for _, s in run["stats"]), 2),
                 waited=round(run.get("waited", 0.0), 2), peak_gib=round(run.get("peak", 0.0), 2),
                 process_gib=round(run.get("process", 0.0), 2))
    STATS.append(entry)
    try:
        with open(OUT / "stats.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# The rows of the table under the results (user 11/26): Student left, Teacher right, the same stages on the same row,
# seconds only; the denoising's token count in its own row under it (thousands), Ours with its cut against the Teacher.
# A stored stage name maps to its row by its start; a name not listed keeps a row of its own.
DISPLAY = (("references: ", "references and text encoder (VLM)"), ("text encoder", "references and text encoder (VLM)"),
           ("noise and packed layout", "references and text encoder (VLM)"), ("codec pruning", "codec pruning"),
           ("denoise", "denoise"), ("VAE decode", "VAE decode"), ("frames to the CPU", "save the result"),
           ("write the ", "save the result"), ("model moves", "model moves (CPU <-> GPU)"),
           ("draw the kept tokens", "other (CPU, page)"), ("other", "other (CPU, page)"))
ORDER = ("references and text encoder (VLM)", "codec pruning", "denoise", "VAE decode", "save the result",
         "model moves (CPU <-> GPU)", "other (CPU, page)")
TOKENS = re.compile(r"denoise \(([0-9,]+) tokens per step\)")


def display_rows(stats):
    """({row: seconds}, tokens per denoising step or None) of a run's stored stage rows."""
    rows, tokens = {}, None
    for stage, seconds in stats:
        row = next((name for start, name in DISPLAY if stage.startswith(start)), stage)
        rows[row] = rows.get(row, 0.0) + seconds
        m = TOKENS.match(stage)
        if m:
            tokens = int(m.group(1).replace(",", ""))
    return rows, tokens


def stats_html(runs, stored=()):
    """The stage table of the runs on the page (Ours left, Teacher right), folded under the results. Every column says
    where its numbers come from: "precomputed" (`stored`: stored with a bundled example) or "live run" (run on this
    page)."""
    have = [w for w in ("student", "released") if (runs.get(w) or {}).get("stats")]
    if not have:
        return ""
    table, tokens = {}, {}
    for w in have:
        table[w], tokens[w] = display_rows(runs[w]["stats"])
    table.get("released", {}).pop("codec pruning", None)  # the Teacher keeps every token: no codec pruning
    names = [r for r in ORDER if any(r in table[w] for w in have)]
    names += [r for w in have for r in table[w] if r not in names]
    totals = {w: sum(table[w].values()) for w in have}
    source = {w: "precomputed" if w in stored else "live run" for w in have}
    head = "".join(f"<th>{SHORT[w]} (s)<br><span class='stats-src stats-{source[w].split()[0]}'>{source[w]}</span></th>"
                   for w in have)
    body = []
    for row in names:
        cells = "".join(f"<td>{table[w][row]:.1f}</td>" if row in table[w] else "<td>&ndash;</td>" for w in have)
        body.append(f"<tr><td>{html.escape(row)}</td>{cells}</tr>")
        if row == "denoise" and any(tokens[w] for w in have):
            teacher = tokens.get("released")

            def cell(w):
                if not tokens[w]:
                    return "<td>&ndash;</td>"
                cut = (f" (\u2212{1 - tokens[w] / teacher:.0%})" if w == "student" and teacher and tokens[w] < teacher
                       else "")
                return f"<td>{tokens[w] / 1000:.1f}K{cut}</td>"
            body.append(f"<tr class='tokens'><td>total tokens</td>{''.join(cell(w) for w in have)}</tr>")
    body.append("<tr class='total'><td>whole run</td>" + "".join(f"<td>{totals[w]:.1f}</td>" for w in have) + "</tr>")
    memory = "; ".join(f"{SHORT[w]} peak {runs[w]['process']:.1f} GB" for w in have if runs[w].get("process"))
    budget = f" (budget {budget_gib():g} GB)" if budget_gib() else ""
    note = f'<p class="stats-note">GPU memory: {memory}{budget}, CUDA context included.</p>' if memory else ""
    where = " · ".join(f"{SHORT[w]}: {source[w]}" for w in have)
    return (f'<details class="stats"><summary>Where the time went, stage by stage ({where})</summary>'
            f'<table class="stats-table">'
            f'<tr><th>stage</th>{head}</tr>{"".join(body)}</table>{note}</details>')


def stats_summary():
    """Markdown: every stage's mean seconds and share over the runs since the start, per page and model."""
    if not STATS:
        return "No finished request yet."
    groups = {}
    for e in STATS:
        groups.setdefault((e["page"], e["who"]), []).append(e)
    out = []
    for (page, who), entries in groups.items():
        n = len(entries)
        sums = {}
        for e in entries:
            for stage, seconds in e["stages"]:
                sums[stage] = sums.get(stage, 0.0) + seconds
        total = sum(sums.values())
        out.append(f"**{LABELS.get(page, page)}, {SHORT[who]}**: {n} run{'s' if n > 1 else ''}, mean {total / n:.1f} s, "
                   f"GPU memory peak {max(e['process_gib'] for e in entries):.1f} GB")
        out.append("| stage | mean s | share |\n|---|---:|---:|")
        out[-1] += "".join(f"\n| {stage} | {seconds / n:.1f} | {seconds / max(total, 1e-9):.0%} |"
                           for stage, seconds in sorted(sums.items(), key=lambda kv: -kv[1]))
    return "\n\n".join(out)
