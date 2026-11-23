"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

Where the seconds of every run went, stage by stage: under the results, in outputs/app/stats.jsonl, and summed over
every request in the panel at the bottom of the page.
"""

import html
import json
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


def stats_html(runs, stored=()):
    """The stage table of the runs on the page (Ours and / or Teacher side by side), folded under the results. Every
    column says where its numbers come from: "precomputed" (`stored`: stored with a bundled example) or "live run"
    (run on this page)."""
    have = [w for w in ("student", "released") if (runs.get(w) or {}).get("stats")]
    if not have:
        return ""
    names = list(dict.fromkeys(stage for w in have for stage, _ in runs[w]["stats"]))
    table = {w: dict(runs[w]["stats"]) for w in have}
    totals = {w: sum(table[w].values()) for w in have}
    source = {w: "precomputed" if w in stored else "live run" for w in have}
    head = "".join(f"<th>{SHORT[w]} (s)<br><span class='stats-src stats-{source[w].split()[0]}'>{source[w]}</span></th>"
                   "<th>share</th>" for w in have)
    body = []
    for stage in names:
        cells = "".join(f"<td>{table[w][stage]:.1f}</td><td>{table[w][stage] / max(totals[w], 1e-9):.0%}</td>"
                        if stage in table[w] else "<td></td><td></td>" for w in have)
        body.append(f"<tr><td>{html.escape(stage)}</td>{cells}</tr>")
    body.append("<tr class='total'><td>whole run</td>" + "".join(f"<td>{totals[w]:.1f}</td><td>100%</td>" for w in have)
                + "</tr>")
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
