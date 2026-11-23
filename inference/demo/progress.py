"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

One request at a time on the GPU (`Line`), what each request shows while it waits and runs (`Report`: the live
progress line, stage by stage), and how a request stops: Stop, another example picked, or its page closed (`drive`).
"""

import asyncio
import contextvars
import itertools
import threading
import time
import traceback
from contextlib import contextmanager

import gradio as gr

from . import state
from .state import STATE, VIEW


class Superseded(Exception):
    """The request was stopped (Stop, another example picked, its page closed): it ends at its next check."""


class Refused(Exception):
    """An input the demo does not run: said in a warning and under the outputs, which stay as they are (a gr.Error
    would mark every output "Error")."""


REQUESTS = itertools.count(1)


PARTS = ("Encoding (VAE + VLM && Codec Pruning)", "Denoising", "Decoding")  # the progress bar's three stages
DECODE_STAGES = ("VAE decode", "frames to the CPU", "image to the CPU", "audio VAE decode")  # `codecopsd.stats` stages of part 3


class Report:
    """What a request is doing, shown on its outputs while it runs: waiting for the GPU, then (`live`) a progress bar
    drawn again every second, in three parts (`PARTS`: encode, denoising, decoding), each counting its own work. It
    ends (Superseded) at the next check once `cancel()` was called or `alive()` is False. Every request is logged with
    its number."""

    def __init__(self, progress=None, alive=None, what="a request"):
        # not `progress or ...`: an empty gr.Progress has len() 0, so it is falsy
        self._shown = progress is not None
        self._progress = progress if progress is not None else (lambda *args, **kwargs: None)
        self.alive = alive or (lambda: True)
        self.what, self.now, self.cancelled, self.id = what, "", False, next(REQUESTS)
        self.in_line, self.t0 = False, time.perf_counter()  # in_line: it has its place in the GPU line (or ended)
        self.who, self.steps = "", 6
        self.pipe = self.clock = None  # the pipeline of the model run now and its `codecopsd.stats.Clock` (this call's)
        self.phase = None  # (text, start time) of what the run does outside the pipeline's stages (loading, writing)
        self.called = False  # the pipeline call of the model run now has ended (the file is written)
        self.left = None  # seconds left of the run, once its denoising steps can tell (the queue shows it)
        self._stop, self._lock = None, threading.Lock()

    def log(self, text):
        print(f"[request {self.id}] {time.strftime('%H:%M:%S')} {text}", flush=True)

    def progress(self, value, desc, unit="steps"):
        self.now = desc
        if not self.cancelled:
            self._progress(value, desc=desc, unit=unit)

    def cancel(self):
        self.cancelled = True
        self.quiet()

    def check(self):
        if self.cancelled or not self.alive():
            raise Superseded

    def stage(self, text):
        self.phase = (text, time.perf_counter())
        if self._stop is not None:  # a model's run: on the live line
            self.redraw()
        else:
            self.progress(None, text)  # no bar, no "0.0%"

    def waiting(self, ahead, holder):
        left = getattr(holder, "left", None)
        text = "Queued behind another request" + (f" (about {left:.0f} s left)" if left else "")
        if ahead:
            text += f", {ahead} more before this one"
        self.progress(None, text)

    def live(self, who):
        """From now until `quiet()`: the progress line of one model's run (`redraw`) drawn again every second."""
        self.quiet()
        self.who, self.run_t0, self.pipe, self.clock, self.phase = who, time.perf_counter(), None, None, None
        self.called, self.left = False, None
        if not self._shown:
            return
        self._stop = stop = threading.Event()
        context = contextvars.copy_context()  # Gradio finds the event of a progress update in the context

        def loop():
            while not stop.wait(1.0):
                context.run(self.redraw)
        threading.Thread(target=loop, daemon=True, name=f"progress-{self.id}").start()

    def quiet(self):
        if self._stop is not None:
            self._stop.set()
            self._stop = None

    def redraw(self):
        """The progress line with its bar, the stage and its count only: "Denoising 3/6". Three stages, each with its
        own bar: the encode (the VAE's tiles and the VLM's layers, codec pruning on the CPU meanwhile), the denoising
        steps, and the decode (the VAE's tiles; then writing the file)."""
        with self._lock:
            if self.cancelled or self._stop is None:
                return
            if self.clock is None and self.pipe is not None:
                self.clock = getattr(self.pipe, "clock", None)  # set by the pipeline call (None before it)
            stage = self.clock.now[0] if self.clock is not None and self.clock.now is not None else ""
            work = STATE.get("work") or {}
            enc, dec = work.get("enc", (0, 0)), work.get("dec", (0, 0))
            k, steps = STATE.get("calls", 0), self.steps
            count = lambda done, total: (min(done, total), total) if total else None
            if self.called or dec[1] or stage.startswith(DECODE_STAGES):
                part, value, unit = 3, count(*dec) or ((1, 1) if self.called else None), ""
            elif k or stage.startswith("denoise"):
                part, value, unit = 2, (min(k, steps), steps), "steps"
                per = STATE.get("denoise", 0.0) / k if k else None
                self.left = per * max(steps - k, 0) + 5 if per else None  # the decode after it: a few seconds
            elif self.clock is not None or enc[1]:
                part, value, unit = 1, count(*enc), ""
            else:  # before the pipeline call: the model is loaded onto the GPU (first use of the tab)
                self.progress(None, self.phase[0] if self.phase is not None else "Starting")
                return
            done = f" {value[0]}/{value[1]}" if value else ""  # the bar itself shows a percentage
            self.progress(value, PARTS[part - 1] + done, unit)

    def start(self, who, steps):
        """The pipeline call starts: its stages show on the live line; a denoise step redraws it at once."""
        self.who, self.steps, self.phase = who, steps, None

        def encoded():
            self.check()
            self.redraw()
        STATE["on_encoded"] = encoded
        STATE["on_step"] = self.step
        self.redraw()

    def step(self, k):
        self.check()
        self.redraw()

    def end(self):
        self.called = True
        STATE.pop("on_step", None)
        STATE.pop("on_encoded", None)


class Line:
    """The GPU: one request at a time, the others wait in arrival order and see what runs and their place. A waiting
    request that is stopped leaves the line."""

    def __init__(self):
        self.cond, self.queue, self.holder = threading.Condition(), [], None

    @contextmanager
    def use(self, report):
        ticket = object()
        with self.cond:
            self.queue.append(ticket)
            report.in_line = True
            try:
                while self.holder is not None or self.queue[0] is not ticket:
                    report.waiting(self.queue.index(ticket), self.holder)
                    self.cond.wait(timeout=1.0)
                    report.check()
            except BaseException:
                self.queue.remove(ticket)
                self.cond.notify_all()
                raise
            self.queue.pop(0)
            self.holder = report
        try:
            yield
        finally:
            with self.cond:
                self.holder = None
                self.cond.notify_all()


GPU = Line()


def page_open(session):
    """False once the page of `session` was closed or reloaded: Gradio then marks its running events as no longer
    alive (it does not stop a function that runs). True when that cannot be read."""
    try:
        return all(job.alive for jobs in state.DEMO._queue.active_jobs if jobs for job in jobs if job.session_hash == session)
    except Exception:
        return True


def view_of(request, name):
    """`alive()` of a request: True until its page is closed or the same viewer picks another example in the tab."""
    session = getattr(request, "session_hash", None)
    key = (session, name)
    token = VIEW.get(key, 0)
    return lambda: VIEW.get(key, 0) == token and page_open(session)


DONE = object()


async def drive(gen, report):
    """Runs the blocking generator `gen` (one request) in a worker thread and yields what it yields. When Gradio stops
    this event (Stop, another example picked), the request leaves the GPU line or ends at its next denoise step. Errors
    reach the page as a message and the log with their traceback."""
    try:
        while True:
            item = await asyncio.to_thread(next, gen, DONE)
            if item is DONE:
                return
            yield item
    except (asyncio.CancelledError, GeneratorExit):
        report.cancel()
        report.log("stop asked")
        raise
    except gr.Error as e:
        report.log(f"error: {e.message}")
        raise
    except Exception as e:
        report.log("failed:\n" + traceback.format_exc())
        raise gr.Error(f"The request failed ({type(e).__name__}: {e}). The log has the details.") from None
