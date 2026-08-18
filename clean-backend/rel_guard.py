# -*- coding: utf-8 -*-
r"""Per-page / per-job reliability guards for app.process() (2026-08-18).

WHY. On the single prod worker, one upload can run for many minutes and starve every
concurrent job because process()'s page loop has no time bound. Measured on real jobs
(rel2_real_*.json / rel2_census.json), three independent cost drivers:

  * PER-PAGE OCR ADMISSION — is_elevation_page(ocr_fallback=True) runs title-block OCR on
    every non-text-elevation page. 26-232: 128s of a 147s run (21 pages, 14.4s max) — and
    on a MARKED job that OCR result is never even used. (app.py fixes that one directly by
    not computing page_is_elev on the digitize path.)
  * DENSE-PAGE VECTOR/TEXTURE — vector_hatch.detect / texture.detect cost scales with the
    page's drawing-SEGMENT count. Legit dense elevations top out ~23s (26-130 p26, 94k
    segs); civil/detail mega-sheets run 120-213s (26-232 p8, 951k segs).
  * PAGE COUNT / INVERSION — a 60-pp set with 26 real elevations (26-130), or a
    text-unreliable set that admits ALL pages, simply has too many heavy pages.

MECHANISM. process() runs in a FastAPI background thread (Starlette run_in_threadpool),
NOT the main thread, so signal.alarm/SIGALRM is unavailable; and a Python thread cannot be
force-killed. So the PRIMARY per-page guard is a CHEAP PRE-CHECK on the segment count
(too_dense) — deterministic, spawns no thread, leaves no dangling CPU — that skips
auto-detect on a mega-dense page before the expensive call is made. call_with_timeout is a
BACKSTOP for cost drivers the segment count does not predict (e.g. an OCR blow-up inside
texture._read_scale): it runs the call in a daemon thread and abandons it after T_PAGE.
HONEST CAVEAT: the abandoned daemon is not killed — it runs its C call to completion — but
MuPDF/OpenCV/numpy/onnxruntime release the GIL so the caller proceeds, it is daemon=True
(never blocks shutdown), and the pre-check keeps it from ever being spawned on the pages
that actually hang. The per-JOB cap (enforced in app.py) is the ultimate backstop: once
exceeded, process() stops admitting pages and returns a partial takeoff.

THRESHOLDS (censused 2026-08-18; all env-overridable so prod retunes without a redeploy):
  T_PAGE_SECONDS   = 60    slowest LEGIT elevation page measured 22.8s (26-130 p26);
                           pathological 120-213s -> 60s clears legit 2.6x, well below path.
  PRECHECK_SEG_MAX = 250k  densest LEGIT admitted elevation ~95k segs; vector time is ~linear
                           in segs (95k->22s) so 250k ~ 58s ~ T_PAGE, and the 279k-951k
                           mega-sheets are skipped before the call.
  T_JOB_SECONDS    = 600   slowest LEGIT large job ~450s (26-130, 26 real elevations)
                           completes; unbounded/inverted docs return partial at 600s.
"""
import os
import threading

T_PAGE_SECONDS = float(os.environ.get("BFS_T_PAGE_SECONDS", "60"))
T_JOB_SECONDS = float(os.environ.get("BFS_T_JOB_SECONDS", "600"))
PRECHECK_SEG_MAX = int(os.environ.get("BFS_PRECHECK_SEG_MAX", "250000"))

# One line shown to the estimator on a page the guard abandoned. It books NO square footage.
PAGE_REVIEW_MSG = ("⏱ This sheet was too complex to auto-read within the time budget — "
                   "auto-detect was skipped and NO square footage was booked for it. Review "
                   "it manually (it may be a dense civil/detail sheet, not a wall elevation).")


class GuardTimeout(Exception):
    """A guarded call exceeded its per-page wall-clock budget (or failed pre-check) and was
    abandoned. app.py catches this and marks the page needs-review with zero booked SF."""


def page_segment_count(pg):
    """Total drawing-segment items on the page — the cost proxy for auto-detect. Uses the
    process-wide get_drawings cache (vector_hatch monkeypatch), so the vector reader that
    runs next reuses this parse for free. Returns -1 if it cannot be read."""
    try:
        dr = pg.get_drawings()
        return sum(len(d.get("items") or []) for d in dr)
    except Exception:
        return -1


def too_dense(pg):
    """PRE-CHECK. Returns (skip, nseg): skip=True when the page has more drawing segments
    than any legit elevation and would blow the per-page budget in vector/texture — so
    auto-detect is skipped on it entirely, with no worker thread spawned."""
    n = page_segment_count(pg)
    return (n > PRECHECK_SEG_MAX, n)


def call_with_timeout(fn, seconds=None):
    """Run fn() in a daemon thread; return its result, or raise GuardTimeout after `seconds`
    (default T_PAGE_SECONDS). The abandoned thread is not killed (Python cannot) — see the
    module docstring for why that is safe here. Use too_dense() first so this is never even
    reached for the pages that predictably hang."""
    if seconds is None:
        seconds = T_PAGE_SECONDS
    box = {}

    def _run():
        try:
            box["v"] = fn()
        except BaseException as e:      # let a real error surface if it finishes in time
            box["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise GuardTimeout("exceeded %.0fs" % seconds)
    if "e" in box:
        raise box["e"]
    return box.get("v")


def health():
    """Reported on /health so a deploy can confirm the running app.py imported this."""
    return {"t_page_s": T_PAGE_SECONDS, "t_job_s": T_JOB_SECONDS,
            "precheck_seg_max": PRECHECK_SEG_MAX}


def _selftest():
    import time
    ok = 0
    fail = []

    # too_dense boundary
    class _FakePg:
        def __init__(self, nseg):
            self._n = nseg

        def get_drawings(self):
            return [{"items": [0] * self._n}]
    if too_dense(_FakePg(PRECHECK_SEG_MAX + 1))[0] and not too_dense(_FakePg(PRECHECK_SEG_MAX))[0]:
        ok += 1
    else:
        fail.append("too_dense boundary")

    # a fast call returns its value
    if call_with_timeout(lambda: 42, seconds=5) == 42:
        ok += 1
    else:
        fail.append("fast call value")

    # a slow call raises GuardTimeout (and the caller continues immediately)
    t0 = time.time()
    try:
        call_with_timeout(lambda: time.sleep(10), seconds=1)
        fail.append("slow call did not time out")
    except GuardTimeout:
        if time.time() - t0 < 3:
            ok += 1
        else:
            fail.append("timeout did not return promptly")

    # an exception that finishes in time propagates (not swallowed as a timeout)
    try:
        call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("boom")), seconds=5)
        fail.append("exception not propagated")
    except ValueError:
        ok += 1
    except Exception:
        fail.append("wrong exception type propagated")

    print("rel_guard selftest: %d/4 ok; thresholds T_PAGE=%.0f T_JOB=%.0f SEG_MAX=%d"
          % (ok, T_PAGE_SECONDS, T_JOB_SECONDS, PRECHECK_SEG_MAX))
    if fail:
        print("  FAIL:", fail)
    return 0 if not fail else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
