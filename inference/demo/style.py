"""On-Policy Codec Distillation for Efficient All-in-One Video Generation (CodecOPSD).

Authors: Boyang Wang, Kevin Blackburn-Matzen, Oindrila Saha, Matheus Gadelha, Zezhou Cheng

Copyright (c) 2026 The CodecOPSD Authors. All rights reserved.

How the page looks: the header text, the theme (system fonts), the CSS, the page script (`HEAD`: the carousel, the
paired reference videos, the number boxes) and the inline icons, so that the page loads nothing from outside.
"""

import gradio as gr


INTRO = """
<div class="hdr">
<h1 class="hdr-title">On-Policy Codec Distillation for Efficient All-in-One Video Generation</h1>
"""
# two lines on the page (split at "\n")
SUMMARY = ("We propose codec pruning, based on our modified video codec, to drop redundant condition tokens,\n"
           "and develop on-policy self-distillation (OPSD) to adapt the model to this pruning pattern for much faster "
           "generation.")

# System fonts: the page loads nothing from outside the demo (the default theme fetches Google Fonts).
THEME = gr.themes.Soft(primary_hue="blue", font=["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto",
                                                 "Helvetica Neue", "Arial", "sans-serif"],
                       font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"])

CSS = """
.gradio-container {max-width: 1880px !important; margin: auto !important; padding-left: 16px !important;
  padding-right: 16px !important}
.hdr {text-align: center; padding: 10px 0 14px 0; margin-bottom: 6px; border-bottom: 1px solid var(--border-color-primary)}
.hdr-name {font-size: 38px !important; font-weight: 800 !important; margin: 0 0 6px 0 !important; letter-spacing: -0.5px}
.hdr-title {font-size: 28px !important; font-weight: 700 !important; margin: 0 auto 14px auto !important; max-width: 900px; text-wrap: balance}
.intro-line {font-size: 16px; margin: 14px 0 0 0 !important; text-wrap: balance}
.star-line {font-size: 15px; margin: 10px 0 0 0 !important; text-wrap: balance}
/* colours: the Student and its numbers in blue, the Teacher and its numbers in orange, precomputed results in purple;
   red only for the speed-up */
.ours-panel [data-testid="block-label"] {background: #1d4ed8 !important; color: #ffffff !important; font-weight: 700 !important;
  font-size: 15px !important}
.ours-panel [data-testid="block-label"] svg {color: #ffffff !important}
.teacher-panel [data-testid="block-label"] {background: #c2410c !important; color: #ffffff !important; font-weight: 600 !important;
  font-size: 15px !important}
.teacher-panel [data-testid="block-label"] svg {color: #ffffff !important}
.speedup {color: #dc2626; font-weight: 800}
.drop-pct {font-weight: 700; color: #1d4ed8; white-space: nowrap}
.v-student {color: #1d4ed8; font-weight: 700} .v-released {color: #c2410c; font-weight: 700}
.precomputed {color: #7c3aed; font-weight: 800} .precomputed-note {color: #7c3aed; font-size: 15px}
.run-mode label {font-weight: 600}
#go-video, #go-image, #stop-video, #stop-image {min-height: 52px; font-size: 18px}
/* the results: a two-page carousel, the Student on the first page, the Teacher one swipe away */
.carousel {flex-wrap: nowrap !important; overflow-x: auto !important; scroll-snap-type: x mandatory; gap: 0 !important;
  scrollbar-width: none; overscroll-behavior-x: contain}
.carousel::-webkit-scrollbar {display: none}
.carousel > div {flex: 0 0 100% !important; min-width: 100% !important; max-width: 100% !important;
  scroll-snap-align: start; scroll-snap-stop: always}
.car-nav {display: flex; justify-content: center; align-items: center; gap: 8px; margin: 6px 0 0 0}
.car-arrow {display: inline-flex; align-items: center; justify-content: center; width: 38px; height: 38px; padding: 0;
  border-radius: 50%; cursor: pointer; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  color: var(--body-text-color); box-shadow: 0 1px 3px rgba(0, 0, 0, .12)}
.car-arrow svg {width: 20px; height: 20px; display: block}
.car-arrow:hover {background: var(--button-secondary-background-fill-hover)}
.car-tab {display: inline-flex; align-items: center; gap: 8px; padding: 7px 16px; border-radius: 999px; cursor: pointer;
  font-size: 15px; font-weight: 600; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  color: #64748b}
.car-tab .car-dot {width: 9px; height: 9px; border-radius: 50%; background: #cbd5e1}
.car-tab.ours.on {background: #1d4ed8; border-color: #1d4ed8; color: #ffffff}
.car-tab.teacher.on {background: #c2410c; border-color: #c2410c; color: #ffffff}
/* Side by Side: both results next to each other instead of the two pages */
.car-side {padding: 7px 16px; border-radius: 999px; cursor: pointer; font-size: 15px; font-weight: 600; margin-left: 10px;
  border: 1px solid #334155; background: var(--block-background-fill); color: #334155}
.car-side.on {background: #334155; color: #ffffff}
.carousel.side {overflow-x: hidden !important; scroll-snap-type: none; gap: 12px !important}
.carousel.side > div {flex: 1 1 0 !important; min-width: 0 !important; max-width: none !important}
.car-nav.side .car-arrow {visibility: hidden}
.car-nav.side .car-tab {background: var(--block-background-fill) !important; color: #64748b !important;
  border-color: var(--border-color-primary) !important}
.car-tab.on .car-dot {background: #ffffff}
.core-info p {font-size: 17px; margin: 4px 0}
.codec-note p {font-size: 13px; color: #4b5563; margin: 6px 0 0 0}
.codec-strategy label:first-of-type {font-weight: 700}  /* Default (recommended) */
.setting-note {font-size: 12.5px; color: #6b7280; margin: 10px 0 0 0; line-height: 1.5}
details.stats {margin: 10px 0 0 0; font-size: 13px}
details.stats summary {cursor: pointer; color: #374151; font-weight: 600}
table.stats-table {border-collapse: collapse; margin: 6px 0 0 0; font-variant-numeric: tabular-nums}
table.stats-table th, table.stats-table td {padding: 2px 10px 2px 0; text-align: right; border-bottom: 1px solid #e5e7eb}
table.stats-table th:first-child, table.stats-table td:first-child {text-align: left}
table.stats-table tr.total td {font-weight: 600}
.stats-note {font-size: 12.5px; color: #6b7280; margin: 6px 0 0 0}
.page-head {font-size: 15px; margin: 14px 0 0 0 !important}
/* the header: paper / code / model links, then the authors (small), the demo pages */
.hdr-authors {font-size: 13px; margin: 10px 0 0 0; line-height: 1.5} .hdr-author {white-space: nowrap}
a.hdr-author {color: #1d4ed8; text-decoration: none} a.hdr-author:hover {text-decoration: underline}
.hdr-authors sup {font-size: 10px; margin-left: 1px}
.hdr-links, .hdr-demos {display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap}
.hdr-links {margin: 14px 0 0 0}
.hdr-demos {margin: 14px 0 0 0}
.hdr-link {display: inline-flex; align-items: center; gap: 7px; height: 36px; padding: 0 16px; border-radius: 999px;
  font-size: 14.5px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  color: var(--body-text-color) !important; text-decoration: none !important; font-weight: 600;
  box-shadow: 0 1px 2px rgba(0, 0, 0, .06)}
.hdr-link svg {width: 20px; height: 20px; display: block} .hdr-link svg.logo-arxiv {width: auto; height: 26px}
.hdr-demos-label {font-size: 14px; color: #6b7280; margin-right: 2px}
.hdr-demo {padding: 5px 14px; border-radius: 8px; font-size: 14px; font-weight: 600; text-decoration: none !important;
  border: 1px solid #1d4ed8; color: #1d4ed8 !important}
.hdr-demo.on {background: #1d4ed8; color: #ffffff !important}
.hdr-demo.soon {border-color: var(--border-color-primary); color: #9ca3af !important; font-weight: 500}
.progress-text {display: none !important}
/* the progress line over a running result: bold and readable (Gradio shows it small, in a monospace font) */
.progress-level-inner {font-family: inherit !important; font-size: 16px !important; font-weight: 700 !important; color: #111827 !important}
.hidden-run {display: none !important}
[role=tablist] button {font-size: 19px !important; font-weight: 700 !important}
[role=tablist] button:nth-child(1) {color: #1d4ed8 !important}
[role=tablist] button:nth-child(2) {color: #dc2626 !important}
.examples .grid-wrap, .examples .grid-container {max-height: none !important; height: auto !important; overflow: visible !important}
/* thumbnail captions in full (the examples' speed-up, the references' sizes), on as many lines as they need */
.examples .caption-label, .ref-input .caption-label {white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important; max-width: calc(100% - 8px) !important; line-height: 1.3 !important}
.ref-input .caption-label {font-size: 12px !important}
.examples .caption-label {font-size: 11px !important; padding: 1px 5px !important; opacity: 0.92}
/* every example thumbnail numbered in its top-left corner, in page order (1, 2, ...), to refer to an example by number */
.examples .grid-container {counter-reset: example}
.examples .thumbnail-item {counter-increment: example}
.examples .thumbnail-item::before {content: counter(example); position: absolute; top: 5px; left: 5px; z-index: 5;
  min-width: 18px; padding: 1px 5px; border-radius: 5px; background: rgba(17, 24, 39, 0.6); color: #fff;
  font: 600 12px/1.35 ui-sans-serif, system-ui, sans-serif; text-align: center; pointer-events: none}
/* the gallery's own label sits over its top-left corner: the tiles start below it */
.examples .grid-wrap {padding-top: 30px !important}
/* the example gallery as a mosaic: the fine grid the script in HEAD places every thumbnail on (its own aspect ratio, no
   square crop); before the script runs, every thumbnail spans 6 of the 24 columns at about 16:9 */
.examples .grid-container {grid-template-columns: repeat(24, minmax(0, 1fr)) !important; grid-template-rows: none !important;
  grid-auto-rows: 8px !important; gap: 6px !important}
.examples .grid-container > * {grid-column: span 6; grid-row: span 12; min-width: 0; min-height: 0}
.examples .thumbnail-item {aspect-ratio: auto !important; width: 100% !important; height: 100% !important}
.examples .thumbnail-item > img, .examples .thumbnail-item video {width: 100% !important; height: 100% !important;
  object-fit: cover !important}
@media (max-width: 640px) {.examples .grid-container {grid-template-columns: repeat(12, minmax(0, 1fr)) !important}}
.stats-src {font-weight: 600; font-size: 11px}
.stats-live {color: #15803d}
.stats-precomputed {color: #7c3aed}
/* the references as the model reads them, under the results */
.refs {margin-top: 4px}
.refs-title {font-size: 16px; font-weight: 600; margin: 2px 0 8px 0}
.ref-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px; margin-bottom: 10px}
.ref-card, .ref-video {margin: 0; padding: 8px; border: 1px solid var(--border-color-primary); border-radius: 8px;
  background: var(--block-background-fill)}
.ref-video {margin-bottom: 10px}
.ref-card img {display: block; width: 100%; max-height: 300px; object-fit: contain}
.ref-pair {display: grid; grid-template-columns: 1fr 1fr; gap: 10px}
.ref-pair figure {margin: 0}
.ref-pair video {display: block; width: 100%; max-height: 300px; background: #000}
.ref-card figcaption, .ref-pair figcaption, .ref-lines {font-size: 14px; line-height: 1.5; margin-top: 6px}
.frame-readout {display: inline-block; min-width: 150px; font-weight: 600; font-variant-numeric: tabular-nums}
.frame-step {font-size: 13px; padding: 1px 10px; margin-right: 6px; border: 1px solid var(--border-color-primary);
  border-radius: 6px; background: var(--button-secondary-background-fill); cursor: pointer}
"""

# The two videos of a reference play together: the one the viewer last played, paused or moved leads, the other
# follows it; the counter shows the frame on screen, and the frame buttons step the pair one frame at a time.
HEAD = """
<script>
(() => {
  // number boxes (thresholds, shares, seed): digits and one point only; the browser would accept 1e3, +1, -1
  const isNum = t => t && t.tagName === 'INPUT' && t.type === 'number';
  document.addEventListener('keydown', e => { if (isNum(e.target) && ['e', 'E', '+', '-'].includes(e.key)) e.preventDefault(); }, true);
  document.addEventListener('paste', e => {
    if (!isNum(e.target)) return;
    const text = (e.clipboardData || window.clipboardData).getData('text').trim();
    if (!/^[0-9]*[.]?[0-9]*$/.test(text)) e.preventDefault();
  }, true);
  document.addEventListener('drop', e => { if (isNum(e.target)) e.preventDefault(); }, true);
})();
(() => {
  window.refSync = true;
  const now = () => performance.now();
  const lead = v => { const g = v && v.closest && v.closest('.ref-sync'); if (g && !(v._quiet > now())) g._lead = v; };
  for (const type of ['play', 'pause', 'seeking']) document.addEventListener(type, e => lead(e.target), true);
  const quiet = (v, act) => { v._quiet = now() + 400; act(); };
  document.addEventListener('click', e => {
    const b = e.target.closest && e.target.closest('.frame-step'); if (!b) return;
    const g = b.closest('.ref-sync'); const vs = [...g.querySelectorAll('video')]; if (!vs.length) return;
    const fps = +g.dataset.fps, n = +g.dataset.frames, l = g._lead && g.contains(g._lead) ? g._lead : vs[0];
    const frame = Math.min(n - 1, Math.max(0, Math.floor(l.currentTime * fps + 1e-3) + (+b.dataset.step)));
    for (const v of vs) quiet(v, () => { v.pause(); v.currentTime = (frame + 0.5) / fps; });
  });
  window.carouselGo = (id, i) => { const el = document.getElementById(id); if (el) el.scrollTo({left: i * el.clientWidth, behavior: 'smooth'}); };
  const sideBySide = (nav, on) => {  // both results next to each other, or the two pages again
    const el = document.getElementById(nav.dataset.target); if (!el) return null;
    el.classList.toggle('side', on); nav.classList.toggle('side', on);
    const s = nav.querySelector('.car-side'); if (s) { s.classList.toggle('on', on); s.setAttribute('aria-pressed', on); }
    el._page = undefined; el.scrollTo({left: 0});
    if (on) { syncAt(el, 0, true, true); follow(); } else leaveSide(el);
    return el;
  };
  // Side by Side: both results start from frame 0 at the same moment, wherever they were, and then show the same frame
  // while they play and loop. `syncAt` pauses both, seeks both to the same time, waits until both have data to play
  // on and starts both in one task. `follow` (every display frame) keeps the Teacher's video on the Student's: a small
  // playbackRate change (1 % per 4 ms of offset, at most 5 %); more than half a frame apart (a late start, as a video
  // with sound has, or a stall), the one ahead waits paused until the other reaches its frame; a seek only beyond
  // 0.5 s. The loop is ours (`loop` off, Gradio sets it again with a new value): once both have ended, both start from
  // 0 again. A play, pause or seek on either video, or a new result, moves both.
  const sideVideos = el => [...el.querySelectorAll('video')].filter(v => (v.currentSrc || v.src) && !v.error);
  let videoIds = 0;  // a replaced video element is a new result, even with the same src
  const sideKey = el => sideVideos(el).map(v => `${v._sideId = v._sideId || ++videoIds}:${v.currentSrc || v.src}`).join('|');
  const until = (v, type, ok, ms) => new Promise(done => {  // ok(), checked on the event and every 25 ms, or the time limit
    const t0 = performance.now(), check = () => {
      if (ok() || performance.now() - t0 > ms) { clearInterval(id); v.removeEventListener(type, check); done(); }
    };
    const id = setInterval(check, 25);
    v.addEventListener(type, check); check();
  });
  const loaded = v => {  // the rest of the video is in the browser: a new result still downloading would show frames late
    const b = v.buffered; if (!isFinite(v.duration)) return true;
    for (let i = 0; i < b.length; i++) if (b.start(i) <= v.currentTime + 0.1 && b.end(i) >= v.duration - 0.1) return true;
    return false;
  };
  // every video paused on time t (with the rest of it loaded if `load`: a new result), then all started in one task
  const syncAt = async (el, t, play, load) => {
    const token = el._sync = (el._sync || 0) + 1, live = () => el._sync === token && el.classList.contains('side');
    el._busy = true;
    try {
      for (let k = 0; live(); k++) {  // again (up to 5 times) if a video changed or started on its own meanwhile
        const vs = sideVideos(el), key = sideKey(el), at = v => Math.min(t, v.duration || t);
        for (const v of vs) { v._held = false; v._d = undefined; v.pause(); }  // the others wait while a new result loads
        await Promise.all(vs.map(v => until(v, 'loadedmetadata', () => v.readyState >= 1, 10000)));
        if (!live()) return;
        await Promise.all(vs.map(v => {
          if (v._loop === undefined) v._loop = v.loop;
          v.loop = false; v.pause(); v.playbackRate = 1; v.currentTime = at(v);
          return until(v, 'seeked', () => !v.seeking, 3000);
        }));
        if (load) await Promise.all(vs.map(v => until(v, 'progress', () => loaded(v), 5000)));
        await Promise.all(vs.map(v => until(v, 'canplay', () => v.readyState >= 3, 3000)));
        if (!live()) return;
        if (k < 4 && (sideKey(el) !== key || !vs.every(v => v.paused && Math.abs(v.currentTime - at(v)) < 0.01))) continue;
        el._key = sideKey(el); el._playing = play;
        if (play) {
          const now = sideVideos(el);
          for (const v of now) v._sidePlay = true;
          await Promise.race([Promise.all(now.map(v => v.play().catch(() => {}))), new Promise(r => setTimeout(r, 3000))]);
          for (const v of now) v._sidePlay = false;
        }
        return;
      }
    } finally { if (el._sync === token) el._busy = false; }
  };
  const leaveSide = el => {  // the two pages again: playbackRate 1 and the player's own loop back
    el._sync = (el._sync || 0) + 1; el._busy = el._playing = false; el._key = undefined;
    for (const v of el.querySelectorAll('video')) {
      v.playbackRate = 1; v._held = false; if (v._loop !== undefined) { v.loop = v._loop; v._loop = undefined; }
    }
  };
  let following = false;
  const follow = () => {
    if (following) return;
    following = true;
    const step = () => {
      const els = document.querySelectorAll('.carousel.side');
      for (const el of els) {
        if (el._busy) continue;
        if (sideKey(el) !== el._key) { syncAt(el, 0, true, true); continue; }  // a new result: both from 0
        const [l, ...fs] = sideVideos(el); if (!l) continue;
        for (const v of [l, ...fs]) if (v.loop) v.loop = false;
        for (const f of fs) {
          const held = l._held === true ? l : f._held === true ? f : null, other = held === l ? f : l;
          if (held) {  // it plays again once the other shows its frame (or has stopped)
            if (other.paused || other.currentTime >= held.currentTime - 0.002) {
              held._held = 'resume'; el._resumed = performance.now(); held.play().catch(() => {});
            }
            continue;
          }
          let rate = 1;
          if (el._playing && !l.paused && !f.paused) {
            const d = f.currentTime - l.currentTime;  // > 0: the follower is ahead
            if (Math.abs(d) > 0.5) { syncAt(el, l.currentTime, true); break; }
            // the one ahead waits, at most once a second: a slow restart after a wait is nudged away, not waited for
            if (Math.abs(d) > 0.025 && !(performance.now() - el._resumed < 1000)) {
              const a = d > 0 ? f : l; a._held = true; f._d = undefined; a.pause(); continue;
            }
            f._d = f._d === undefined ? d : f._d + 0.2 * (d - f._d);  // smoothed: a clock read can jitter by ms
            f._nudge = Math.abs(f._d) > (f._nudge ? 0.002 : 0.005);  // start beyond 5 ms, stop within 2 ms
            if (f._nudge) rate = 1 - Math.max(-5, Math.min(5, Math.round(f._d * 250))) / 100;
          } else f._d = undefined;
          if (f.playbackRate !== rate) f.playbackRate = rate;
        }
      }
      if (els.length) requestAnimationFrame(step); else following = false;
    };
    requestAnimationFrame(step);
  };
  // the viewer's play, pause or seek on either video moves both; both ended: both from 0
  for (const type of ['play', 'pause', 'seeking', 'ended']) document.addEventListener(type, e => {
    const v = e.target, el = v.tagName === 'VIDEO' && v.closest('.carousel.side');
    if (!el) return;
    if (el._busy) { if (type === 'play' && !v._sidePlay) v.pause(); return; }  // the player's autoplay waits too
    if (v._held) { if (type === 'play') v._held = false; return; }  // our own wait of a moment (`follow`)
    if (type === 'ended') { if (sideVideos(el).every(x => x.ended)) syncAt(el, 0, true); }
    else if (type === 'pause') { if (el._playing && !v.ended) syncAt(el, v.currentTime, false); }
    else if (type === 'play') { if (!el._playing) syncAt(el, v.currentTime, true); }
    else syncAt(el, v.currentTime, el._playing);  // seeking
  }, true);
  document.addEventListener('click', e => {
    const side = e.target.closest && e.target.closest('.car-side');
    if (side) { const nav = side.closest('.car-nav'); sideBySide(nav, !nav.classList.contains('side')); return; }
    const b = e.target.closest && e.target.closest('.car-arrow, .car-tab');
    if (b) {
      if (b.closest('.car-nav').classList.contains('side')) sideBySide(b.closest('.car-nav'), false);
      const el = document.getElementById(b.closest('.car-nav').dataset.target); if (!el) return;
      const page = Math.round(el.scrollLeft / Math.max(1, el.clientWidth));
      const i = b.dataset.go !== undefined ? +b.dataset.go : Math.max(0, Math.min(el.children.length - 1, page + (+b.dataset.step)));
      window.carouselGo(el.id, i); return;
    }
    if (e.target.closest && e.target.closest('.examples button')) {  // an example: its result from the top, Ours first
      for (const el of document.querySelectorAll('.carousel')) window.carouselGo(el.id, 0);
      const tab = document.querySelector('[role=tablist]'); if (tab) tab.scrollIntoView({behavior: 'smooth', block: 'start'});
    }
  }, true);
  setInterval(() => {
    for (const el of document.querySelectorAll('.carousel')) {
      if (el.classList.contains('side')) continue;  // Side by Side: `follow` keeps both aligned
      const i = Math.round(el.scrollLeft / Math.max(1, el.clientWidth));
      if (el._page === i) continue;
      el._page = i;
      const nav = document.querySelector(`.car-nav[data-target="${el.id}"]`);
      if (nav) nav.querySelectorAll('.car-tab').forEach((d, k) => d.classList.toggle('on', k === i));
      [...el.children].forEach((c, k) => { const v = c.querySelector('video'); if (v) { if (k === i) v.play().catch(() => {}); else v.pause(); } });
    }
  }, 200);
  setInterval(() => {
    for (const g of document.querySelectorAll('.ref-sync')) {
      const vs = [...g.querySelectorAll('video')]; if (vs.length < 2) continue;
      if (!g._started) {  // the page inserts these videos as HTML, where autoplay does not start them
        g._started = true; for (const v of vs) quiet(v, () => { v.muted = true; v.play().catch(() => {}); }); continue;
      }
      const l = g._lead && g.contains(g._lead) ? g._lead : vs[0];
      for (const v of vs) {
        if (v === l || v._quiet > now()) continue;
        if (l.paused !== v.paused) quiet(v, () => l.paused ? v.pause() : v.play().catch(() => {}));
        if (Math.abs(v.currentTime - l.currentTime) > (l.paused ? 0.01 : 0.08)) quiet(v, () => { v.currentTime = l.currentTime; });
      }
      const out = g.querySelector('.frame-readout'), fps = +g.dataset.fps, n = +g.dataset.frames;
      if (out) out.textContent = `Frame ${Math.min(n, Math.floor(l.currentTime * fps + 1e-3) + 1)} of ${n}`;
    }
  }, 150);
})();
// A reference view, larger: a click on a picture view opens it over the page (with its caption); Enlarge on a video
// moves its pair (it keeps playing in sync) over the page until the overlay closes (a click beside it, x or Escape).
(() => {
  let open = null;
  const close = () => {
    if (!open) return;
    if (open.moved) { if (open.placeholder.isConnected) open.placeholder.replaceWith(open.moved); else open.moved.remove(); }
    open.overlay.remove(); document.body.style.overflow = open.overflow; open = null;
  };
  const show = (content, moved, placeholder) => {
    close();
    const overlay = document.createElement('div'); overlay.className = 'ref-zoom-overlay';
    const x = document.createElement('button'); x.className = 'zoom-close'; x.setAttribute('aria-label', 'close'); x.textContent = '×';
    overlay.append(x, content);
    overlay.addEventListener('click', e => {
      if (e.target === overlay || e.target === x || e.target.tagName === 'IMG' || e.target.classList.contains('zoom-figure')) close();
    });
    document.body.append(overlay);
    open = {overlay, moved, placeholder, overflow: document.body.style.overflow};
    document.body.style.overflow = 'hidden';
  };
  document.addEventListener('click', e => {
    const t = e.target; if (!t.closest || t.closest('.ref-zoom-overlay')) return;
    const img = t.closest('.refs .ref-card img');
    if (img) {
      const fig = document.createElement('figure'); fig.className = 'zoom-figure';
      const big = document.createElement('img'); big.src = img.src; big.alt = img.alt; fig.append(big);
      const cap = img.closest('.ref-card').querySelector('figcaption');
      if (cap) { const c = document.createElement('figcaption'); c.innerHTML = cap.innerHTML; fig.append(c); }
      show(fig); return;
    }
    const b = t.closest('.ref-zoom');
    if (b) {
      const g = b.closest('.ref-video'); if (!g) return;
      const ph = document.createElement('div'); ph.style.height = g.offsetHeight + 'px';
      g.replaceWith(ph); show(g, g, ph);
    }
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
})();
(() => {
  // The example gallery as a mosaic (CSS ".examples .grid-container"): every thumbnail at its own aspect ratio, placed
  // on a fine grid (24 columns, 12 on a phone; 8 px rows) by a skyline: each tile goes into the lowest free stretch, at
  // a width from an uneven pattern (narrower for portrait and square pictures), or over the whole stretch when the rest
  // would be too narrow; a tile whose bottom lands within a quarter of its height of a neighbour's is snapped to it
  // (cover crop), and no tile may line every column up. So no gutter runs straight across (checked on the real
  // thumbnails from 700 to 1400 px wide). Placement only: the DOM order (the numbers in the corners, the clicks) stays
  // the page order.
  const ROW = 8, GAP = 6, PATTERN = [6, 7, 5, 6, 8, 5, 7, 6, 5, 7];  // 4 to 5 tiles a row (user 11/23: smaller tiles)
  const childOf = (grid, el) => { while (el && el.parentElement !== grid) el = el.parentElement; return el; };
  const layout = grid => {
    const tracks = getComputedStyle(grid).gridTemplateColumns.split(' ').map(parseFloat).filter(x => x > 0);
    const cols = tracks.length, colW = tracks[0];
    if (!cols || !colW) return;
    const scale = cols / 24, minW = Math.max(2, Math.round(4 * scale)), maxW = Math.max(minW + 1, Math.round(8 * scale));
    const top = new Array(cols).fill(0);
    grid.querySelectorAll('.thumbnail-item').forEach((thumb, i) => {
      const cell = childOf(grid, thumb);
      if (!cell) return;
      const media = thumb.querySelector('img, video');
      const mw = media ? (media.naturalWidth || media.videoWidth || 0) : 0;
      const mh = media ? (media.naturalHeight || media.videoHeight || 0) : 0;
      const ratio = mw && mh ? mw / mh : 16 / 9;
      // the leftmost stretch whose top is near the lowest one (within about half a tile): the tiles fill the page
      // left to right, band by band, so the numbers read in order; every stretch is filled at its own top (no holes)
      const floor = Math.min(...top), near = Math.round(colW * 3 / (ROW + GAP));
      const a = top.findIndex(t => t <= floor + near), low = top[a];  // the stretch [a, b)
      let b = a;
      while (b < cols && top[b] === low) b++;
      const seg = b - a, shape = ratio < 0.9 ? 0.6 : ratio < 1.2 ? 0.8 : 1;
      const want = Math.max(minW, Math.round(PATTERN[i % PATTERN.length] * scale * shape));
      // a too narrow rest joins the tile only while the tile stays at most maxW wide; else it is left for a small tile
      const w = Math.min(seg, seg - want >= minW ? want : seg <= maxW ? seg : Math.min(maxW, seg - minW));
      let h = Math.max(3, Math.round(((w * colW + (w - 1) * GAP) / ratio + GAP) / (ROW + GAP)));
      const x = a;  // always the stretch's left end: the numbers read left to right
      const tol = Math.max(1, Math.round(h * 0.25));
      for (const n of [x - 1, x + w]) {  // snap to a neighbour's bottom, unless that lines up every column
        if (n < 0 || n >= cols) continue;
        const d = top[n] - (low + h);
        if (d === 0 || Math.abs(d) > tol) continue;
        const next = top.map((t, c) => (c >= x && c < x + w ? low + h + d : t));
        if (new Set(next).size > 1) { h += d; break; }
      }
      if (top.every((t, c) => (c >= x && c < x + w ? low + h : t) === low + h)) h += h > 6 ? -2 : 2;  // no line across
      cell.style.gridColumn = `${x + 1} / span ${w}`;
      cell.style.gridRow = `${low + 1} / span ${h}`;
      for (let c = x; c < x + w; c++) top[c] = low + h;
    });
  };
  // Laid out again only when something it reads changed (the gallery's width, its tiles, their pictures' sizes): the
  // signature check keeps it from running on every update of the page (the progress line redraws every second).
  const signature = grid => grid.clientWidth + '|' + [...grid.querySelectorAll('.thumbnail-item img, .thumbnail-item video')]
    .map(m => (m.naturalWidth || m.videoWidth || 0) + 'x' + (m.naturalHeight || m.videoHeight || 0)).join(',');
  const done = new WeakMap(), watched = new WeakSet();
  let queued = false;
  const run = () => {
    queued = false;
    document.querySelectorAll('.examples .grid-container').forEach(grid => {
      watch(grid);
      const sig = signature(grid);
      const placed = [...grid.children].every(c => c.style.gridRow);  // tiles re-rendered by Gradio lose their place
      if (done.get(grid) === sig && placed) return;
      try { layout(grid); } catch (err) { console.warn('example mosaic', err); }
      done.set(grid, sig);
    });
  };
  const schedule = () => { if (!queued) { queued = true; setTimeout(run, 150); } };
  // each gallery watched on its own (its tiles added or replaced); the page only for a gallery that appears
  const watch = grid => {
    if (watched.has(grid)) return;
    watched.add(grid);
    new MutationObserver(schedule).observe(grid, {childList: true, subtree: true, attributes: true, attributeFilter: ['src']});
  };
  new MutationObserver(records => {
    for (const r of records) for (const n of r.addedNodes)
      if (n.nodeType === 1 && (n.matches('.examples, .examples *') || n.querySelector('.examples'))) return schedule();
  }).observe(document.body, {childList: true, subtree: true});
  document.addEventListener('load', e => { if (e.target.closest && e.target.closest('.examples')) schedule(); }, true);
  window.addEventListener('resize', schedule);
  schedule();
})();
</script>
<style>
.refs .ref-card img {cursor: zoom-in}
.ref-zoom {font-size: 13px; padding: 1px 10px; border: 1px solid var(--border-color-primary); border-radius: 6px;
  background: var(--button-secondary-background-fill); cursor: pointer}
.ref-zoom-overlay {position: fixed; inset: 0; z-index: 10000; display: flex; align-items: center; justify-content: center;
  padding: 48px 2vw 2vh 2vw; box-sizing: border-box; background: rgba(0, 0, 0, 0.88); cursor: zoom-out}
.ref-zoom-overlay .zoom-close {position: absolute; top: 6px; right: 14px; font-size: 34px; line-height: 1; color: #fff;
  background: none; border: 0; cursor: pointer}
.ref-zoom-overlay .zoom-figure {margin: 0; display: flex; flex-direction: column; align-items: center; max-width: 100%}
.ref-zoom-overlay .zoom-figure img {width: 96vw; height: calc(100vh - 110px); object-fit: contain}
.ref-zoom-overlay figcaption {color: #f3f4f6; font-size: 15px; margin-top: 8px; text-align: center}
.ref-zoom-overlay .ref-video {width: min(96vw, 2000px); max-height: calc(100vh - 60px); overflow: auto; margin: 0;
  cursor: default; background: #111; border-color: #333; color: #f3f4f6}
.ref-zoom-overlay .ref-video {padding: 8px; border: 1px solid #333; border-radius: 8px; box-sizing: border-box}
.ref-zoom-overlay .ref-pair {display: grid; grid-template-columns: 1fr 1fr; gap: 10px}
.ref-zoom-overlay .ref-pair figure {margin: 0}
.ref-zoom-overlay .ref-pair video {display: block; width: 100%; max-height: calc(100vh - 150px); background: #000}
.ref-zoom-overlay .ref-lines {font-size: 14px; line-height: 1.5; margin-top: 6px}
.ref-zoom-overlay .frame-readout {display: inline-block; min-width: 150px; font-weight: 600; font-variant-numeric: tabular-nums}
.ref-zoom-overlay .ref-zoom {display: none}
.ref-zoom-overlay .frame-step {font-size: 13px; padding: 1px 10px; margin-right: 6px; border: 1px solid #444;
  border-radius: 6px; background: #222; color: #f3f4f6; cursor: pointer}
@media (max-width: 640px) {.ref-zoom-overlay .ref-pair {grid-template-columns: 1fr}
  .ref-zoom-overlay .ref-pair video {max-height: calc(50vh - 80px)}}
</style>
"""


# The page header (the way paper demos on Hugging Face Spaces start): title, the paper / code / model / news links
# (placeholders until the URLs exist), the authors (small, under the links), and the demo pages (`--links`: this page
# highlighted, the others linked).
PAPER = dict(
    title="On-Policy Codec Distillation for Efficient All-in-One Video Generation",
    # (name, mark, homepage): each name links to the author's page (a new tab)
    authors=[("Boyang Wang", "", "https://kiteretsu77.github.io/BoyangWang/"),
             ("Kevin Blackburn-Matzen", "", "https://kmatzen.com/"),
             ("Oindrila Saha", "", "https://research.adobe.com/person/oindrila-saha/"),
             ("Matheus Gadelha", "", "https://mgadelha.me/"),
             ("Zezhou Cheng", "", "https://www.cs.virginia.edu/~zc3bp/")],
    links=[("arxiv", "", "https://arxiv.org/abs/2611.31184"), ("github", "Code", "https://github.com/Kiteretsu77/CodecOPSD"),
           ("huggingface", "Models", "https://huggingface.co/collections/HikariDawn/codecopsdopsd-6b04b49f98f5b4b68c5f7f9a"),
           ("x", "X", "https://x.com/BoyangWang7")])
ICONS = {  # inline, so the page loads nothing from outside
    "arxiv": '<svg class="logo-arxiv" aria-label="arXiv" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 246.98 111.12"><path d="M69.41,32.88c1.86,0,3.1,1.24,3.99,3.45,1.06-2.21,2.57-3.45,4.69-3.45h14.88c2.3,0,4.07,1.86,4.07,4.07v7.83c0,2.66-1.33,4.07-4.07,4.07s-4.07-1.42-4.07-4.07v-3.75h-10.54c-1.77,0-2.66.97-2.66,2.75v27.19h10.01c2.66,0,4.07,1.33,4.07,4.07s-1.42,4.07-4.07,4.07h-26.39c-2.66,0-3.99-1.33-3.99-4.07s1.33-4.07,3.99-4.07h8.24v-29.93h-7.26c-2.66,0-3.99-1.33-3.99-4.07s1.33-4.07,3.99-4.07h9.12Z" fill="#7c7469"/><path d="M181.07,32.88c2.66,0,4.07,1.42,4.07,4.07v34.01h10.1c2.75,0,4.07,1.33,4.07,4.07s-1.33,4.07-4.07,4.07h-28.6c-2.66,0-4.07-1.33-4.07-4.07s1.42-4.07,4.07-4.07h10.36v-29.93h-8.53c-2.74,0-4.07-1.33-4.07-4.07s1.33-4.07,4.07-4.07h12.61ZM185.29,15.27c0,3.19-2.71,5.82-5.82,5.82s-5.9-2.63-5.9-5.82,2.71-5.82,5.9-5.82,5.82,2.63,5.82,5.82Z" fill="#7c7469"/><path d="M246.98,36.96c0,.53-.09,1.15-.27,1.59l-14.88,38.08c-.71,1.68-1.95,2.48-3.72,2.48h-5.93c-1.86,0-3.19-.8-3.81-2.48l-15.05-38.08c-.27-.44-.36-1.06-.36-1.59,0-2.3,1.86-4.07,4.16-4.07,1.86,0,3.01.97,3.72,2.66l14.35,36.13,13.9-36.13c.71-1.68,1.86-2.66,3.72-2.66,2.3,0,4.16,1.77,4.16,4.07Z" fill="#7c7469"/><path d="M127.98,55.61l-32.74,38.85c-1.29,1.37-2.08,3.78-1.36,5.5.75,1.8,2.46,2.91,4.4,2.91,1.09,0,1.99-.38,3.16-1.56l40.19-42.71c1.6-1.69,1.62-4.33.04-6.04l-13.68,3.05Z" fill="#aa142d"/><path d="M127.98,55.61l31.19-38.27c1.49-1.99,2.2-3.03,1.49-4.72-.74-1.77-2.59-3.16-4.48-3.16h0c-1.06,0-1.72.09-3.01,1.11l-38.63,41.76c-1.72,1.84-1.71,4.7.02,6.53l47.79,51.07c1.02,1.05,2.05,1.19,3.14,1.19,1.93,0,3.19-1.14,4.03-2.82.72-1.73-.08-3.44-1.4-5.23l-40.12-47.46" fill="#7c7469"/><path d="M141.67,52.56L95,2.13S93.29.04,91.48,0s-3.6,1.02-4.34,2.79c-.7,1.69-.2,2.88,1.35,5.1l40.09,48.42" fill="#aa142d"/><path d="M32.45,32.88c5.02,0,8.21,3.31,8.21,8.4v37.83H5.14c-2.88,0-5.14-1.87-5.14-4.93v-13.03c0-4.07,2.83-7.35,7-8.15l25.51-3.57v-8.41H4.14c-2.05,0-4.14-1.45-4.14-4.07,0-2.87,2.14-4.07,4.36-4.07h28.09ZM32.5,70.97v-13.29l-24.35,3.4v9.9h24.35Z" fill="#7c7469"/></svg>',  # arXiv's own logo (static.arxiv.org design system)
    "github": '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
              "2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82"
              "-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2"
              "-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27"
              ".68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75"
              "-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8"
              '-8-8z"/></svg>',
    "huggingface": '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 95 88" aria-hidden="true" fill="none"><path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z" /><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z" /><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z" /><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z" /><path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd" /><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z" /><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z" /><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z" /><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z" /></svg> ',  # Hugging Face's own logo (huggingface.co)
    "x": '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M18.244 2.25h3.308l-7.227 8.26 8.502 '
         "11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 "
         '4.126H5.117z"/></svg>',
}
