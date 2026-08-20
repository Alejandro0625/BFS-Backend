"""
BFS Estimator — clean lightweight backend.  (deploy build: clean-backend v1)
Reads the estimator's Bluebeam markup polygons (digitize-markup) → exact SF per material.
Light deps only (PyMuPDF + OpenCV + numpy) → boots instantly, never OOMs.
Matches the existing React frontend contract:
  POST /analyze (multipart 'pdf') -> {jobId}
  GET  /status/{jobId}            -> {status, phase, log, progress, legend, takeoffData, scheduleData, error}
  GET  /polygons/{jobId}/{page}   -> {polygons, width, height}
  GET  /page-image/{jobId}/{page} -> PNG
  GET  /health
"""
import os, io, re, uuid, threading, json, time, shutil, hashlib
from collections import defaultdict
import fitz  # PyMuPDF
import texture  # classical-CV texture fallback for unmarked drawings
import model_infer  # trained cladding-extent model (ONNX) — the real auto-markup for RAW drawings
import vector_hatch  # reads the DRAWN pattern vectors (seam trains + gray fills) — exact, preferred on clean pages
import callouts  # reads the drawing's own text callouts + leader arrows -> names the regions
import density_reader  # dense-microtexture suggestions (Callahan class)
import ocr_text  # OCR fallback (onnxruntime RapidOCR) for FLATTENED sets — lazy, memory-safe
import titleblock  # title-block OCR (sheet no. + drawing title) — page-finding fallback for curve-title sheets
import snap_fill  # coloring-book BUCKET fill + corner-snap → exact SF from vector geometry (assist layer)
import material_groups  # within-job texture grouping → a selectable PREVIEW of material groups (assist layer)
import auto_trim as auto_trim_mod  # derive corner/base/opening LF from face geometry (blueprint 1c) — suggestions only
import dim_scale  # self-calibrate scale from the drawing's own dimension strings (blueprint 1b) — cross-check only
import dim_area  # second reader: independent area from printed W/H dimension strings (confidence signal; two-tier, moves no SF)
import admin_auth  # FAIL-CLOSED guard for /admin/* — no key configured means the door is shut, not open
# SELECT-TO-SUM FIX #1: job-level material GROUPING + NAMING. Imported DEFENSIVELY so a deploy
# that lands app.py without mat_canon.py degrades to today's per-name rollup instead of failing
# to boot — but NEVER silently: the reason is surfaced on /health as mat_canon, because a
# grouping that quietly stopped working looks exactly like a capability gap.
try:
    import mat_canon as _mc
    _MC_ERR = None
except Exception as _mce:      # pragma: no cover - deploy-shape guard
    _mc = None
    _MC_ERR = str(_mce)
# REVIEW RANKING (FIX_TWO_TIER_GATE): order the estimator's queue by the reader class's
# MEASURED confidently-wrong rate. Books nothing, moves no SF, asserts nothing — it only
# decides what she looks at first. Same defensive import as mat_canon above, for the same
# reason: `clean-backend/openings_convention.py` was once a 404 in the repo while app.py
# imported it, so a new module must degrade to today's behaviour instead of failing to
# boot — and must say so on /health, because a ranking that quietly stopped ranking looks
# exactly like a queue nobody prioritised.
try:
    import review_rank as _rr
    _RR_ERR = None
except Exception as _rre:      # pragma: no cover - deploy-shape guard
    _rr = None
    _RR_ERR = str(_rre)
# RELIABILITY GUARDS (2026-08-18): per-page/per-job wall-clock bounds so one dense upload
# cannot occupy the single worker for minutes and starve every concurrent job (measured #1
# blocker, ~35% of real jobs). Defensive import for the same reason as the two above —
# a new module was once a 404 in the repo while app.py imported it — but the deploy pushes
# rel_guard.py BEFORE app.py, and /health reports it. When present it supplies the per-page
# pre-check + timeout; the per-JOB cap and the big-PDF reject below need no module, so they
# still fire if it is somehow absent. See rel_guard.py for the censused thresholds.
try:
    import rel_guard as _rg
    _RG_ERR = None
    _GuardTimeout = _rg.GuardTimeout
except Exception as _rge:      # pragma: no cover - deploy-shape guard
    _rg = None
    _RG_ERR = str(_rge)
    class _GuardTimeout(Exception):
        """Never raised when rel_guard is absent — auto-detect runs unbounded, as it did
        before this change; the per-JOB cap and big-PDF reject still apply."""
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Where corrections accumulate as training data (mount a Railway volume here to persist).
CORR_DIR = os.environ.get("CORRECTIONS_DIR", "/data/corrections")
MATERIAL_LEDGER = os.environ.get("MATERIAL_LEDGER",
                                 os.path.join(os.path.dirname(CORR_DIR), "material_name_ledger.jsonl"))
# Durable job store — jobs are also written here so takeoffs survive a restart/redeploy.
JOBS_DIR = os.environ.get("JOBS_DIR", "/data/jobs")
MAX_MEM_JOBS = int(os.environ.get("MAX_MEM_JOBS", "20"))   # cap RAM: never OOM (evicted jobs live on disk)
MAX_DISK_JOBS = int(os.environ.get("MAX_DISK_JOBS", "200"))  # cap volume: prune oldest persisted jobs
MAX_AUTO_PAGES = int(os.environ.get("MAX_AUTO_PAGES", "30"))  # cap texture auto-detect work per PDF
# SITE/CIVIL SCALE CUT (2026-08-06, FIX_ENGINE_BLOWUP.md). A permit set contains civil,
# site, locus and grading sheets, and the engine has no page-selection stage, so it
# measures them as cladding at the scale THEY print. The scale is read correctly; the
# sheet is land, not a wall — 26-046 p1 prints 1"=400' and booked 58,468,346 SF, which is
# 2,500x an elevation at 1/8"=1'-0"; 26-199 p11 prints 1"=2000' (62,500x) and booked
# 3,136,600 SF. Census of every measured page in the 121-job archive (3,829 pages):
# the confirmed-scale ladder actually observed is 1, 1.33, 2, 2.67, 4, 5.33, 6, 8, 10,
# 10.67, 16, 20, 21.33, then a GAP, then 64, 400, 500, 2000. The cut sits in that gap and
# ABOVE every architectural scale anyone draws a building at (1/64"=1'-0" is 64 ft/in):
# 100 ft/in cannot come off the architectural ladder at all. At this cut 3 pages of 3,829
# are touched (0.08%) carrying 61,664,732 SF = 42.5% of all SF the archive run booked,
# and ZERO of them is a gold page — the coarsest confirmed scale carrying any gold
# anywhere in the corpus is 21.33 ft/in, 4.7x below the cut. Measured, not tuned:
# bfs_overnight/ENGSCALE_EXPOSURE.json carries the same table at 17/20/32/64/100/400,
# and 20 is NOT usable — it takes a gold page (26-234 p5) and four frozen-exam jobs.
SITE_SCALE_FT_PER_IN = float(os.environ.get("SITE_SCALE_FT_PER_IN", "100"))
# BLIND-BAND cut (WAVE 5 Item 2, 2026-08-17): a CONFIRMED site/civil sheet drawn coarser
# than any elevation but below the 100 ft/in site cut (e.g. an architectural SITE PLAN at
# 1"=30') slips under SITE_SCALE_FT_PER_IN and books phantom SF — measuring parking lots
# and courtyards as cladding. 24 ft/in sits ABOVE the coarsest CONFIRMED gold scale in the
# corpus (21.33 ft/in; r7x: 0 of 372 gold pages read >= 22 confirmed), so demoting on it can
# never touch a wall she marked. The scale alone does NOT demote — the page must ALSO name
# itself a site/civil sheet AND carry no wall-material keynote (_is_site_civil_sheet), so a
# real large-building elevation drawn coarse is never demoted.
SITE_BLINDBAND_FT_PER_IN = float(os.environ.get("SITE_BLINDBAND_FT_PER_IN", "24"))
# SIZE-OUTLIER GATE (Wave B Feature 3, so_sizegate_census.json): an auto region that covers more
# than this fraction of its page is almost always a page-spanning boundary phantom (a full-page
# gray box, a legend/whole-sheet smear), not a wall. Demote it to suggest_only (never delete).
# 0.50 is the census "bulletproof" T: it sits in the empirical 0.50-0.60 zero-gap and is 1.42x
# over the gold floor 0.3515 (largest fraction any REAL marked wall occupies, 26-070B p16), so at
# this T the gate demotes 0 IoU-matched-to-gold regions and 0 v13 regions => 0 gold SF lost.
SIZE_OUTLIER_FRAC = float(os.environ.get("SIZE_OUTLIER_FRAC", "0.50"))
SIZE_OUTLIER_GOLD_FLOOR = 0.3515

app = FastAPI(title="BFS Clean Backend")
# CORS: browsers may only call this API from the app's own origins (the Vercel prod
# domain + its preview builds + local dev). Non-browser clients (battery scripts,
# curl) send no Origin header and are unaffected — CORS is a browser-side gate.
app.add_middleware(CORSMiddleware,
                   allow_origins=["https://bfs-estimator.vercel.app",
                                  "http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_origin_regex=r"https://bfs-estimator[a-z0-9\-]*\.vercel\.app",
                   allow_methods=["*"], allow_headers=["*"])

# ---- APP LOCK (#32, ships DARK): set env BFS_APP_KEY on Railway to require
# X-BFS-Key on every request (health + CORS preflight exempt). Unset = open,
# exactly today's behavior. Flip = one env var, no deploy.
_APP_KEY = os.environ.get("BFS_APP_KEY", "")

@app.middleware("http")
async def _app_lock(request, call_next):
    if _APP_KEY and request.method != "OPTIONS" and request.url.path != "/health":
        supplied = request.headers.get("x-bfs-key") or request.query_params.get("key")
        if supplied != _APP_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "locked"}, status_code=401)
    return await call_next(request)
jobs = {}  # jobId -> dict  (in-memory hot cache, bounded to MAX_MEM_JOBS)

# ── durable, bounded job store ───────────────────────────────────────────────
# The old code kept every uploaded PDF in `jobs` forever → unbounded RAM → OOM.
# Now: keep a small hot set in RAM, persist each job to the volume, rehydrate on demand.
# -- JOB-ID GUARD (2026-08-06, FIX_ENDPOINT_SECURITY.md finding A) ------------
# `jid` is attacker-controlled on every job route and reached os.path.join with no
# sanitisation. MEASURED against the real routes, and wider than the first write-up:
#   * GET /status/{jid} leaks only through the BACKSLASH forms -- Starlette's path
#     convertor 404s '../' -- so that route alone is a Windows-only defect, which is
#     why the first probe read "not exploitable on the Railway Linux deployment".
#   * But /learn, /accept-suggestion and /split-suggest take the id from a JSON BODY,
#     where no path convertor exists. '../x', '..\\x' and a plain ABSOLUTE path each
#     returned HTTP 200 having PARSED a job.json outside JOBS_DIR, and
#     /accept-suggestion then REWROTE that outside file via _persist_job (proved by
#     content: the rewritten file carries the handler's own "AI wall (accepted)"
#     marker). posixpath resolves '../', and os.path.join(root, '/abs') discards root
#     on EVERY OS -- so those three routes are live in production, read AND write.
# One predicate at the two chokepoints covers all ~12 job routes instead of twelve
# per-route edits. A refused id gets 404, the same answer an unknown id gets, so the
# guard tells a prober nothing it did not already know.
_JID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

def _jid_ok(jid):
    """True only for a bare job id.

    The second clause is not decoration. `..` MATCHES the character class above -- the
    dot is inside it -- and _job_dir("..") is the PARENT of JOBS_DIR, so the pattern on
    its own still escapes one level. Every all-dots id is refused.

    Every real id passes: analyze()/compare() mint f"{ms}-{hex6}", and all 137 ids in
    the job store (including the 'coldrun-26-012' family) match, so no legitimate
    lookup changes behaviour.
    """
    s = str(jid)
    return bool(_JID_RE.match(s)) and s.strip(".") != ""

def _safe_jid(jid):
    """HARD wall: refuse before the id can become a path. _job_dir builds every path in
    the job store, so an untrusted id must never come back out of it, whoever calls it
    next."""
    if not _jid_ok(jid):
        raise HTTPException(404, "job not found")
    return str(jid)

def _job_dir(jid): return os.path.join(JOBS_DIR, _safe_jid(jid))

BIG_PDF_BYTES = int(os.environ.get("BIG_PDF_BYTES", str(200 * 1024 * 1024)))
# RELIABILITY thresholds mirrored here so the per-JOB cap + review message work even if
# rel_guard failed to import (the per-page pre-check/timeout below already no-op in that
# case). Values + rationale live in rel_guard.py; all env-overridable.
T_JOB_SECONDS = float(os.environ.get("BFS_T_JOB_SECONDS", "600"))    # whole-job wall-clock cap
T_PAGE_SECONDS = float(os.environ.get("BFS_T_PAGE_SECONDS", "60"))   # per-page auto-detect cap
# Cumulative per-JOB budget for title-block OCR admission (is_elevation_page's ocr_fallback).
# That OCR is the measured #1 hang on scanned/flattened sets (26-180A UNMARKED would spend
# 326s admitting 26 raster pages). Once spent, remaining pages fall back to TEXT-ONLY
# admission — real elevations that carry text are still admitted; only OCR-only discovery
# stops. 120s sits well above any NORMAL job's total run time (so it never fires on real
# work — identity-preserving) yet caps the scanned-set blow-up an order of magnitude below
# the job cap. On MARKED jobs no OCR runs at all (see page_is_elev below), so this is moot.
T_OCR_BUDGET_S = float(os.environ.get("BFS_T_OCR_BUDGET_S", "120"))
PAGE_REVIEW_MSG = (_rg.PAGE_REVIEW_MSG if _rg is not None else
                   "⏱ This sheet was too complex to auto-read within the time budget — "
                   "auto-detect was skipped and NO square footage was booked for it. Review it "
                   "manually (it may be a dense civil/detail sheet, not a wall elevation).")

def _load_pdf_bytes(path):
    """HEAVY-DOC SAFETY, phase 1: uploads now STREAM to disk (no doubled RAM while a
    giant uploads+parses). Processing still loads bytes — PyMuPDF rejects mmap objects
    ('bad stream: mmap.mmap', tested 2026-07-15), so true zero-copy needs the
    path-based refactor (every fitz.open(stream=...) site accepts a file path via a
    small _open() helper). That is phase 2 — its own gated cycle."""
    with open(path, "rb") as fh:
        return fh.read()

def _persist_job(jid):
    """Snapshot a job to the volume so it survives eviction + restarts. Best-effort (no-op if no volume)."""
    job = jobs.get(jid)
    if not job: return
    try:
        d = _job_dir(jid); os.makedirs(d, exist_ok=True)
        if job.get("pdf") and not os.path.exists(os.path.join(d, "input.pdf")):
            with open(os.path.join(d, "input.pdf"), "wb") as fh: fh.write(job["pdf"])
        # ⚠ THIS TUPLE IS A WHITELIST, and that is the persistence bug class: a key the
        # handlers write and this line forgets is silently lost on the next eviction or
        # redeploy — the job comes back looking untouched. `scopeMap` is the estimator's
        # ticklist, so losing it would un-park her out-of-scope groups behind her back
        # (the parked ZONES themselves ride inside takeoffData and the `out_of_scope`
        # piece flags inside polygons_by_page, both already here).
        # `reviewConfirmed` is her walk of the review queue (which regions she has already
        # eyeballed) and `netOverrides` is her per-zone NET number — the one STEP 4 says IS
        # the delivered quantity. Losing either behind her back is the same bug scopeMap
        # was added here to prevent: the job comes back looking untouched and she redoes
        # an hour of confirmation, or worse, the bid silently reverts to the gross.
        meta = {k: job.get(k) for k in ("status", "phase", "progress", "legend", "takeoffData",
                "scheduleData", "error", "polygons_by_page", "dims_by_page", "log", "projName", "pageCount",
                "coverageReport", "openingsConvention", "scopeMap", "reviewConfirmed", "netOverrides",
                "materialNameMap", "pageScopeMap")}
        with open(os.path.join(d, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except Exception:
        pass  # persistence is a bonus; the app still works from RAM without a volume

def _load_job(jid):
    d = _job_dir(jid)
    if not os.path.isdir(d): return None
    try:
        with open(os.path.join(d, "job.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        for k in ("polygons_by_page", "dims_by_page"):  # JSON stringifies int page keys → restore them
            meta[k] = {int(kk): vv for kk, vv in (meta.get(k) or {}).items()}
        p = os.path.join(d, "input.pdf")
        meta["pdf"] = _load_pdf_bytes(p) if os.path.exists(p) else None
        return meta
    except Exception:
        return None

def get_job(jid):
    """Hot cache first, else rehydrate the job from the durable store (survives restart/eviction)."""
    # SOFT wall, and it is needed HERE as well as in _job_dir: the hot cache is
    # checked first and never reaches _job_dir, and one successful traversal
    # leaves jobs[<traversal id>] populated (measured), so on a long-lived worker
    # the second identical request would be served from RAM past a _job_dir-only
    # guard. It returns None instead of raising so /autonomy-status -- which walks
    # jobIds out of correction records that /learn lets a caller write, outside any
    # try/except -- SKIPS a planted record rather than 404ing the whole endpoint.
    if not _jid_ok(jid): return None
    j = jobs.get(jid)
    if j is None:
        j = _load_job(jid)
        if j is not None: jobs[jid] = j
    return j

def _evict_mem():
    """Keep RAM bounded — drop oldest *finished* jobs (they persist on disk, rehydrate on access)."""
    if len(jobs) <= MAX_MEM_JOBS: return
    for jid in list(jobs.keys()):
        if len(jobs) <= MAX_MEM_JOBS: break
        if jobs[jid].get("status") in ("done", "error"):
            _persist_job(jid); jobs.pop(jid, None)

MAX_DISK_GB = float(os.environ.get("MAX_DISK_GB", "20"))   # heavy-doc safety: cap SIZE too

def _gc_disk():
    """Keep the volume bounded — prune oldest job folders by COUNT and by TOTAL SIZE
    (200 giant sets could fill the volume long before the count cap fires)."""
    try:
        if not os.path.isdir(JOBS_DIR): return
        dirs = [os.path.join(JOBS_DIR, d) for d in os.listdir(JOBS_DIR)]
        dirs = [d for d in dirs if os.path.isdir(d)]
        def _dsize(d):
            t = 0
            try:
                for f in os.listdir(d):
                    t += os.path.getsize(os.path.join(d, f))
            except Exception:
                pass
            return t
        dirs.sort(key=lambda d: os.path.getmtime(d))
        total = sum(_dsize(d) for d in dirs)
        budget = MAX_DISK_GB * 1024 ** 3
        while dirs and (len(dirs) > MAX_DISK_JOBS or total > budget):
            d = dirs.pop(0)
            total -= _dsize(d)
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

# ── BOUNDED JOB QUEUE (2026-08-19, PARALLEL_JOB_PROCESSING) ──────────────────
# The OLD path let every /analyze (and /compare) spawn its own Starlette background
# thread, so a whole daily batch ran AT ONCE. That is the measured #1 reliability
# blocker: each analysis peaks at ~5 GB RSS (v13 onnx session + zoom-2 pixmaps + cv2;
# rel2_census peak 5,045 MB) and the prod container is ~8 GB, so two concurrent jobs
# OOM-restart the single worker and LOSE every in-flight takeoff (a 442 MB set AND
# concurrent load both OOMed prod). And the engine's hot path (vector_hatch geometry)
# is pure-Python / GIL-bound, so those concurrent threads never truly ran in parallel
# anyway — they time-shared one core and STARVED each other. MAX SAFE CONCURRENCY =
# container / per-job = 8 / 5 -> 1, so true parallelism is not affordable in this RAM.
#
# Jobs now go through a BOUNDED FIFO QUEUE drained by exactly BFS_MAX_CONCURRENCY worker
# thread(s) (default 1). Excess uploads wait cleanly as "queued" (visible on /status and
# /health) instead of contending, so RAM stays at 1x per-job and nothing OOMs; each job
# runs to completion in order; and process()'s existing per-JOB cap (T_JOB_SECONDS=600s)
# is the real kill that stops one hung job from holding the slot forever — after the cap
# it returns a partial takeoff and the queue drains to the next job. BFS_MAX_CONCURRENCY
# is env-tunable so a Railway RAM UPGRADE lets the owner raise concurrency with NO
# redeploy. Worker THREADS (not processes) so every worker shares the single v13 onnx
# session; at N=1 the GIL is moot and the queue's role is admission control + reliability,
# not speedup. process()/process_compare are UNCHANGED — only how they are launched is.
import queue as _job_queue_mod
MAX_CONCURRENCY = max(1, int(os.environ.get("BFS_MAX_CONCURRENCY", "1")))
_JOB_QUEUE = _job_queue_mod.Queue()
_QUEUE_LOCK = threading.Lock()
_workers_started = False
_queue_depth = 0   # jobs currently waiting or running (reported on /health)

def _job_worker():
    global _queue_depth
    while True:
        fn, args = _JOB_QUEUE.get()
        try:
            fn(*args)
        except Exception:
            import traceback; traceback.print_exc()
        finally:
            with _QUEUE_LOCK:
                _queue_depth -= 1
            _JOB_QUEUE.task_done()

def _ensure_workers():
    global _workers_started
    if _workers_started:
        return
    with _QUEUE_LOCK:
        if _workers_started:
            return
        for _ in range(MAX_CONCURRENCY):
            threading.Thread(target=_job_worker, name="bfs-job-worker", daemon=True).start()
        _workers_started = True

def _enqueue_job(fn, *args):
    """Hand a heavy job (process / process_compare) to the bounded worker pool. Replaces
    background_tasks.add_task so a whole batch cannot run at once and OOM the worker; the
    job waits in the FIFO as `queued` until a worker is free."""
    global _queue_depth
    _ensure_workers()
    with _QUEUE_LOCK:
        _queue_depth += 1
    _JOB_QUEUE.put((fn, args))

# ── helpers ────────────────────────────────────────────────────────────────
def shoelace(pts):
    n = len(pts); a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def scale_ft_per_in(s):
    if not s: return 8.0
    m = re.search(r'(\d+)\s*/\s*(\d+)', s)
    if m and int(m.group(1)) > 0: return int(m.group(2)) / int(m.group(1))
    m = re.search(r'([\d.]+)', s)
    if m:
        v = float(m.group(1))
        if 0 < v < 1: return 1.0 / v
    return 8.0

def _flag_scale_outliers(td, jlog=None, job=None):
    """CROSS-PAGE SCALE-OUTLIER FLAG (review-only; additive; books/removes NOTHING).

    Most elevation pages in ONE set share a single drawing scale. A page read at a
    CONFIDENTLY-WRONG scale (scale_confirmed, so the existing default-scale warning never
    fires) inflates SF by scale-squared -- the "85x" total. After every page is processed,
    each CONFIRMED-scale page that carries booked cladding is compared against the set's
    shared scale (the median of those pages); when a clear shared scale exists (a MAJORITY
    agree), any page whose scale departs by >=2x or <=0.5x is FLAGGED -- never has its SF
    touched -- so the estimator can verify/calibrate. It only ADDS `scaleOutlier`/
    `scaleOutlierNote` to the page payload; no total, zone or polygon moves, so it can never
    book or remove SF (confidently-wrong impossible). Returns the count of pages flagged.

    Pages read at a DEFAULT/unread scale (verifiedScale falsy) are skipped -- they already
    carry the existing calibrate-before-trusting warning. A job with fewer than 3 confirmed
    booked pages, or with no dominant shared scale, is left untouched (no reference set)."""
    OUT_HI, OUT_LO = 2.0, 0.5
    cand = []   # (entry, ft_per_in) for confirmed, booked-cladding pages
    for e in (td or []):
        if not e.get("verifiedScale"):
            continue          # unread/default pages already get the calibrate warning
        if not any(float(z.get("netArea") or 0) > 0 for z in (e.get("zones") or [])):
            continue          # no booked cladding on this page -- not part of the scale set
        m = re.search(r"=\s*([0-9]*\.?[0-9]+)", e.get("scale") or "")  # "1\"=8.0'" -> 8.0
        if not m:
            continue
        try:
            fpi = float(m.group(1))
        except ValueError:
            continue
        if fpi > 0:
            cand.append((e, fpi))
    if len(cand) < 3:
        return 0              # too few confirmed pages to define a "set" -- no reference
    vals = sorted(v for _, v in cand)
    n = len(vals)
    med = vals[n // 2] if (n % 2) else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    if med <= 0:
        return 0
    # only speak when a MAJORITY agree on one scale (a real shared "set" scale); a job with
    # no dominant scale has no reference against which to call any page an outlier.
    near = sum(1 for v in vals if OUT_LO * med < v < OUT_HI * med)
    if near <= n / 2.0:
        return 0
    def _g(x):
        return "%g" % round(float(x), 2)
    flagged = 0
    for e, fpi in cand:
        r = fpi / med
        if r >= OUT_HI or r <= OUT_LO:
            fac = r if r >= 1.0 else 1.0 / r
            e["scaleOutlier"] = True
            e["scaleOutlierNote"] = (
                "⚠ Scale 1\"=%s' differs from this set's 1\"=%s' by %sx -- SF on this "
                "page could be ~%sx off. Verify or calibrate this sheet before trusting its SF."
                % (_g(fpi), _g(med), _g(round(fac, 1)), _g(round(fac * fac))))
            flagged += 1
            if jlog and job is not None:
                jlog(job, "Page %s: scale 1\"=%s' is an outlier vs this set's shared 1\"=%s' -- "
                          "flagged for review (SF unchanged)" % (e.get("pageNumber"), _g(fpi), _g(med)),
                     "warn")
    return flagged

def categorize(subject):
    s = (subject or "").lower()
    if any(k in s for k in ["acm", "mcm", "composite"]): return "ACM/Composite Panel"
    if any(k in s for k in ["lap", "cementitious", "fiber cement", "fibercement", "nichiha"]): return "Fiber Cement / Lap"
    if any(k in s for k in ["soffit", "fascia", "trim", "return"]): return "Soffit/Trim"
    if any(k in s for k in ["brick", "masonry", "cmu"]): return "Masonry"
    if any(k in s for k in ["standing seam", "metal", "pnl", "panel", "alum"]): return "Metal Wall Panel"
    if any(k in s for k in ["shingle", "shake"]): return "Shingle/Shake"
    if "pvc" in s or "azek" in s: return "PVC/Trim"
    return subject or "Other"

def _reader_from_name(m, src9=""):
    """Which reader produced a region, from the RAW reader name it was given.

    This used to live inside /evidence-pdf and read the zone's displayed materialName. Once
    FIX #1 renames a zone to its material GROUP those prefixes are gone, so every row of the
    evidence receipt would have collapsed to the "Drawing geometry" default — a provenance
    column that silently stops being provenance. The zone now carries `reader`, derived HERE
    from the dominant raw name, and evidence-pdf prefers it."""
    m = str(m or "")
    if "bluebeam" in str(src9 or ""):
        return "Your markup (exact)"
    for pre, nm in (("Hatched area", "Hatch reader"), ("Color fill", "Drawn color"),
                    ("Rendered", "Rendered reader"), ("Wall area (AI boundary", "AI boundary model"),
                    ("Wall band", "Story-band"), ("Wall area", "Structural flood"),
                    ("Panel wall", "Drawn fill")):
        if m.startswith(pre):
            return nm
    return "Drawing geometry"


def _review_fields(classes, auto=True, src9="", raw_name=""):
    """The REVIEW-QUEUE fields for one zone, from the reader classes of the pieces that
    built it (`FIX_TWO_TIER_GATE`: ship the ranking, not an assert tier).

    Additive by construction — every value here is derived, none of them is read back by
    any SF, grouping, total, Excel or evidence quantity. If `review_rank` failed to import
    the zone simply carries no review fields, exactly as it did before this shipped.

    A MARKED page is not ranked with the readers: its SF is the estimator's own polygon,
    exact by geometry, so it takes the `markup` class and sorts last."""
    if _rr is None:
        return {"reader": _reader_from_name(raw_name, src9)}   # exactly today's behaviour
    cls = _rr.MARKUP_CLASS if not auto else _rr.combine(classes or [])
    if cls is None:
        # an auto zone whose pieces carry no stamp — a job persisted before this shipped.
        # Fall back to the NAME, which is what the receipt used to do, and leave the risk
        # fields None: unmeasured is the truth here, and a guessed rank is worse than none.
        return {"readerClass": None, "reviewRank": _rr.UNMEASURED_RANK,
                "reviewRisk": None, "reviewRate": None,
                "reviewWhy": _rr.why(None), "reader": _reader_from_name(raw_name, src9)}
    return {"readerClass": cls,
            "reviewRank": _rr.rank(cls),
            "reviewRisk": _rr.risk(cls),
            "reviewRate": _rr.rate(cls),
            "reviewWhy": _rr.why(cls),
            "reader": _rr.label(cls, src9)}


# ── THE SCOPE GATE (STEP 1, PERFECT_ACCURACY_BLUEPRINT) ──────────────────────
# 70.8% of every SF this engine draws lands where the estimator marked NOTHING
# (FIX_SCOREBOARD_DIRECTION, 3 independent runs): she bids a SCOPED subset —
# brick, EIFS, masonry and glazing belong to other trades by charter — and the
# drawing never says which subset. That is the single biggest source of wasted
# estimator time on this product, and it is NOT a detection error: the engine
# drew the building, she bid a part of it.
#
# Every autonomous attempt to GUESS the subset has died. The last one, 10f76e3,
# auto-demoted the fill reader on a `material`/`group` string prefix and turned
# the LIVE battery red: `callouts.name_regions` (callouts.py:560/588/655)
# overwrites material, category AND group 74 lines before the predicate read
# them, so it missed every piece the drawing had named and removed 5,339 SF of
# real cladding instead — 14,201 -> 8,862 against her 11,843 (FIX_FILL_DEMOTE_
# PREDICATE). So this does not guess. The human ticks, once, in ~30 seconds.
#
# THREE RULES MAKE THAT SAFE
#  1. IDENTITY BY DEFAULT. `scopeMap` starts EMPTY and an absent group is IN
#     SCOPE. `process()` never writes `out_of_scope` — only POST /set-scope
#     does, and only after a takeoff already exists — so the guards added at the
#     booking sites below are provable no-ops until she clicks, and a job nobody
#     touches yields byte-identical zones, totals, Excel and evidence PDF.
#  2. PARKED, NEVER DELETED. An unticked group's zones MOVE, whole, into
#     `parkedZones` and move back on a re-tick. Nothing is recomputed, so no
#     field can be dropped in a rebuild — the /accept-suggestion mirror bug
#     class (a field the analyze path writes and a rebuild forgets vanishes from
#     exactly the pages she touched).
#  3. THE KEY IS THE STAMP, NEVER A NAME PREDICATE. `material_group` is written
#     once, AFTER every naming pass (~app.py:875), and the zone and its polygons
#     carry the same value — a later rename cannot desynchronise them. This is
#     the `reader_class` lesson applied to scope.
#
# ⚠ `out_of_scope` is deliberately NOT `suggest_only`. suggest_only carries
# ACCEPT-FLOW semantics: /accept-suggestion flips it, renames the piece "AI wall
# (accepted)" and files it as boundary-model training gold. A parked group is
# none of those — it is real cladding that simply is not in THIS bid. Reusing
# the flag would have made "she restored it" indistinguishable from "the model
# was right", and poisoned the flywheel with it.

def _scope_group_of_zone(z):
    """The ticklist identity of a ZONE. Grouped (auto) pages emit one zone per
    material_group; MARKED pages are deliberately ungrouped (their names are the
    estimator's own markup labels) and answer with their own name."""
    return str(z.get("material_group") or z.get("materialName") or z.get("category") or "Unlabeled")


def _scope_group_of_poly(p):
    """The ticklist identity of a POLYGON — the same stamp its zone was built from
    (app.py:~875 writes material_group onto the piece and the zone from one pass)."""
    return str(p.get("material_group") or p.get("material") or p.get("category") or "Unlabeled")


def _scope_summary(j):
    """Every material group in the job with its SF, pages and tick — the panel's
    whole payload. `scopeMap` is the single source of truth for the tick; the
    zone's current list (zones vs parkedZones) only follows it."""
    smap = j.get("scopeMap") or {}
    rows = {}
    for el in (j.get("takeoffData") or []):
        pn = el.get("pageNumber")
        for z, is_parked in ([(z, False) for z in (el.get("zones") or [])] +
                             [(z, True) for z in (el.get("parkedZones") or [])]):
            g = _scope_group_of_zone(z)
            r = rows.setdefault(g, {"name": g, "sf": 0.0, "zones": 0, "pages": [],
                                    "category": z.get("category"), "family": z.get("family"),
                                    "reader": z.get("reader"), "readerClass": z.get("readerClass"),
                                    "inScope": bool(smap.get(g, True)), "parked": False})
            r["sf"] += float(z.get("netArea") or 0)
            r["zones"] += 1
            r["parked"] = r["parked"] or is_parked
            if pn and pn not in r["pages"]:
                r["pages"].append(pn)
    out = sorted(rows.values(), key=lambda r: -r["sf"])
    for r in out:
        r["sf"] = round(r["sf"], 1)
    return {"groups": out,
            "inScopeSF": round(sum(r["sf"] for r in out if r["inScope"]), 1),
            "parkedSF": round(sum(r["sf"] for r in out if not r["inScope"]), 1),
            "parkedGroups": [r["name"] for r in out if not r["inScope"]]}


def _apply_scope(j, smap, psmap=None):
    """Park every group she unticked; restore every group she re-ticked. A zone is
    IN scope only when its material group is ticked AND its page is ticked — one
    shared `out_of_scope` flag, two reasons (material-scope `smap` OR page-scope
    `psmap`). A zone parked for either reason only restores when BOTH the group and
    the page are back in scope — correct by the AND below.

    Zones MOVE as whole objects between `zones` and `parkedZones` — never rebuilt —
    and each parked zone remembers the index it left, so a restore puts it back
    where it was. The polygons take the same flag so the canvas can grey them and
    the evidence PDF can skip them (the parallel of the suggest_only exclusion at
    :1921). Returns (zones parked, zones restored).

    `psmap` is `{ "<pageNumber>": bool }` — an absent page defaults IN scope, exactly
    like an absent group in `smap`. An empty `smap` AND empty/absent `psmap` parks
    nothing and restores nothing: the identity path (both `.get(..., True)` calls
    return the True default for every zone, so this is byte-identical to today).
    """
    n_park = n_back = 0
    for el in (j.get("takeoffData") or []):
        _pg9 = str(el.get("pageNumber"))
        _page_in = (psmap or {}).get(_pg9, True)
        zones = list(el.get("zones") or [])
        parked = list(el.get("parkedZones") or [])
        keep = []
        for i, z in enumerate(zones):
            if smap.get(_scope_group_of_zone(z), True) and _page_in:
                keep.append(z)
            else:
                z["out_of_scope"] = True
                z["_scopeIdx"] = i
                parked.append(z); n_park += 1
        still, back = [], []
        for z in parked:
            if smap.get(_scope_group_of_zone(z), True) and _page_in:
                z.pop("out_of_scope", None)
                back.append((int(z.pop("_scopeIdx", len(keep)) or 0), z)); n_back += 1
            else:
                still.append(z)
        for idx, z in sorted(back, key=lambda t: t[0]):
            keep.insert(max(0, min(idx, len(keep))), z)
        el["zones"] = keep
        if still or parked:          # never ADD the key to a page nobody parked on
            el["parkedZones"] = still
    for _pg, polys in (j.get("polygons_by_page") or {}).items():
        _ppage_in = (psmap or {}).get(str(_pg), True)
        for p in (polys or []):
            if smap.get(_scope_group_of_poly(p), True) and _ppage_in:
                p.pop("out_of_scope", None)
            else:
                p["out_of_scope"] = True
    return n_park, n_back


def _page_scope_summary(j):
    """THE PAGE-SCOPE GATE, read side — one row per takeoff page with its detected SF,
    sheet ref, engine flags and its page tick. `pageScopeMap` is the single source of
    truth for the tick; the zones only follow it (parked pages MOVE their zones to
    `parkedZones`, exactly like the material-scope gate, so every total/export drops
    them for free). `detectedSF` sums zones+parkedZones so it is STABLE whether the
    page is in or out — the number the panel shows never jitters as she toggles.

    Pure read; books nothing. With no `pageScopeMap` every page answers inScope:true —
    the identity state. `siteScaleDemoted`/`threeDDemoted`/`needsReview` are the
    engine-flagged classes the panel may PRE-HIGHLIGHT (never pre-uncheck)."""
    psmap = j.get("pageScopeMap") or {}
    pbp = j.get("polygons_by_page") or {}
    pages = []
    in_sf = parked_sf = 0.0
    parked_pages = []
    for el in (j.get("takeoffData") or []):
        pn = el.get("pageNumber")
        zones = list(el.get("zones") or [])
        parked = list(el.get("parkedZones") or [])
        sf = sum(float(z.get("netArea") or 0) for z in (zones + parked))
        polys = pbp.get(pn) or pbp.get(str(pn)) or []
        site_demoted = any(p.get("site_scale_demoted") for p in polys)
        three_d = any(p.get("three_d_demoted") for p in polys)
        in_scope = bool(psmap.get(str(pn), True))
        pages.append({"pageNumber": pn, "sheetRef": el.get("sheetRef"),
                      "title": el.get("title"), "detectedSF": round(sf, 1),
                      "inScope": in_scope, "zonesCount": len(zones) + len(parked),
                      "flags": el.get("flags") or [], "scaleSource": el.get("scaleSource"),
                      "siteScaleDemoted": site_demoted, "threeDDemoted": three_d,
                      "needsReview": el.get("scaleSource") == "default"})
        if in_scope:
            in_sf += sf
        else:
            parked_sf += sf; parked_pages.append(pn)
    return {"pages": pages, "inScopeSF": round(in_sf, 1),
            "parkedSF": round(parked_sf, 1), "parkedPages": parked_pages}


# ── NET BY CONFIRMATION (STEP 4, PERFECT_ACCURACY_BLUEPRINT) ─────────────────
# She books openings-DEDUCTED area (~27% median off gross). Two independent
# programmes proved the machine cannot recover that number from pixels: the v18
# ladder and the netting ORACLE, which showed that even PERFECT wall selection
# plus the best CONSTANT ratio scores BELOW baseline with 27 confidently-wrong
# (fix2_DECISION.md). So this pass does not compute a net, does not suggest a
# net, and does not show a "job habit" — FIX_OPENINGS stands: a habit may be
# SHOWN, never APPLIED, and even the showing is a later pass. All that lands
# here is the PLUMBING: a place to PUT her number and a rule that everything
# downstream reads it.
#
# IDENTITY BY DEFAULT, same three rules as the scope gate:
#  1. `netOverrides` starts EMPTY. With no entry, `_apply_nets` touches nothing
#     and every zone dict is the one `process()` built, key for key.
#  2. THE MACHINE NUMBER IS NEVER DESTROYED. The first override stashes it in
#     `netAreaAuto`; clearing the override puts it back and removes the stash,
#     so a cleared job is byte-identical to one that was never touched.
#  3. THE KEY IS THE ZONE'S SCOPE IDENTITY (page + material_group stamp), not a
#     zone index and not a renameable display string — so an override survives
#     the /accept-suggestion mirror rebuilding that page's zones from scratch,
#     which is the exact way `reader_class` and `material_group` were lost
#     before they were re-derived from stamps.
#
# ⚠ NOTHING IN THE ENGINE READS THIS. `_apply_nets` runs only where a takeoff
# already exists (after /set-net, /set-scope and the /accept-suggestion
# rebuild); `process()` never calls it, so no detection, grading or bench number
# can move because of it.

def _net_key(page, zone_key):
    return "%s|%s" % (page, zone_key)


def _apply_nets(j):
    """Write her confirmed NET onto every zone that has one; restore the machine's
    number on every zone that doesn't. Returns the number of zones overridden.

    Parked zones are netted too — she may park a group AFTER netting it, and
    restoring it must not silently hand back the gross."""
    ov = j.get("netOverrides") or {}
    n = 0
    for el in (j.get("takeoffData") or []):
        pn = el.get("pageNumber")
        for z in (list(el.get("zones") or []) + list(el.get("parkedZones") or [])):
            rec = ov.get(_net_key(pn, _scope_group_of_zone(z)))
            if not isinstance(rec, dict):
                if "netAreaAuto" in z:          # override cleared -> exactly today's zone
                    z["netArea"] = z.pop("netAreaAuto")
                    z.pop("netOverride", None)
                    z.pop("netOverrideAt", None)
                continue
            if "netAreaAuto" not in z:          # keep the machine's draft, once
                z["netAreaAuto"] = z.get("netArea")
            try:
                z["netArea"] = round(float(rec.get("net")), 1)
            except Exception:
                continue
            z["netOverride"] = True
            z["netOverrideAt"] = rec.get("ts")
            n += 1
    return n


def _disp_name(z):
    """The label to SHOW for a zone: her confirmed material name when she has given
    the group one, else the machine's own. DISPLAY/EXPORT ONLY -- it reads the
    optional `materialNameHer` field and NOTHING else, so it is never an input to a
    scope, net or review KEY (those stay on the material_group stamp). With no name
    given it returns exactly `materialName`, so every export is byte-identical."""
    return z.get("materialNameHer") or z.get("materialName")


def _apply_material_names(j):
    """THE NAMING BRIDGE, apply side (the sibling of `_apply_nets`). Write her
    confirmed material NAME onto every zone whose group she named; strip it from
    every zone she did not. Returns the number of names applied.

    DISPLAY ONLY: sets `materialNameHer`, the field `_disp_name` prefers, and never
    touches materialName / material_group / material_type / netArea -- so not one SF
    moves and no scope, net or review KEY drifts. Keyed on the material_group STAMP
    (`_scope_group_of_zone`, the identity the scope gate and `_apply_nets` share) and
    written ONLY on zones that carry that stamp, so the write can never drift the key
    it was looked up under -- the exact trap that reverted the fill demotion.

    IDENTITY BY DEFAULT: an empty `materialNameMap` names nothing and removes any
    stale label a prior map left, so a job nobody named is byte-identical to today's.
    """
    nm = j.get("materialNameMap") or {}
    n = 0
    for el in (j.get("takeoffData") or []):
        for z in (list(el.get("zones") or []) + list(el.get("parkedZones") or [])):
            name = nm.get(_scope_group_of_zone(z)) if z.get("material_group") else None
            if name:
                z["materialNameHer"] = str(name)
                n += 1
            elif "materialNameHer" in z:
                z.pop("materialNameHer", None)
    return n


def _materials_summary(j):
    """GET /materials payload: every material group with its SF, pages, her current
    name and an is_unnamed flag for the generic / no-identity groups the drawing gave
    no name to (mat_canon classes 'generic'/'junk') -- the '[name it]' rows that are
    the flywheel's entry. Pure read; books nothing."""
    nm = j.get("materialNameMap") or {}
    rows = {}
    for el in (j.get("takeoffData") or []):
        pn = el.get("pageNumber")
        for z, parked in ([(z, False) for z in (el.get("zones") or [])] +
                          [(z, True) for z in (el.get("parkedZones") or [])]):
            g = _scope_group_of_zone(z)
            cls = z.get("material_class")
            no_identity = cls in ("generic", "junk")     # reuse mat_canon's classes
            r = rows.setdefault(g, {"group": g, "machineName": g,
                                    "name": nm.get(g) or g, "herName": nm.get(g),
                                    "isNamed": g in nm, "noIdentity": no_identity,
                                    "isUnnamed": no_identity and g not in nm,
                                    "sf": 0.0, "zones": 0, "pages": [],
                                    "category": z.get("category"), "family": z.get("family"),
                                    "reader": z.get("reader"), "readerClass": z.get("readerClass"),
                                    "readAs": z.get("readAs"), "materialClass": cls,
                                    "parked": False})
            r["sf"] += float(z.get("netArea") or 0)
            r["zones"] += 1
            r["parked"] = r["parked"] or parked
            if pn and pn not in r["pages"]:
                r["pages"].append(pn)
    out = sorted(rows.values(), key=lambda r: -r["sf"])
    for r in out:
        r["sf"] = round(r["sf"], 1)
        r["pages"] = sorted(r["pages"])
    return {"groups": out,
            "unnamedSF": round(sum(r["sf"] for r in out if r["isUnnamed"]), 1),
            "namedSF": round(sum(r["sf"] for r in out if r["isNamed"]), 1),
            "unnamedGroups": [r["group"] for r in out if r["isUnnamed"]]}


def _append_material_ledger(j, jid, group, name):
    """THE FLYWHEEL. Append (practice, job, engine group + how it read + reader class
    -> her name) to a durable ledger. It is read back as a SUGGESTION on the next job
    by the same GC/architect -- SHOWN, never auto-applied (FIX_OPENINGS / fill-revert).
    Best-effort, exactly like `_persist_job`."""
    rec = {}
    for el in (j.get("takeoffData") or []):
        for z in (list(el.get("zones") or []) + list(el.get("parkedZones") or [])):
            if _scope_group_of_zone(z) == group:
                rec = z
                break
        if rec:
            break
    row = {"jobId": jid, "projName": j.get("projName"), "group": group,
           "readAs": rec.get("readAs"), "readerClass": rec.get("readerClass"),
           "materialClass": rec.get("material_class"), "name": name,
           "ts": int(time.time() * 1000)}
    os.makedirs(os.path.dirname(MATERIAL_LEDGER), exist_ok=True)
    with open(MATERIAL_LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def jlog(job, msg, level="info"):
    job["log"].append({"msg": msg, "level": level})

# ── NET-ASSIST (Feature 1, 2026-08-18) ──────────────────────────────────────
# The net step (openings deducted per wall) is where most of her review clicks go and it is
# HER judgment. The vector reader already detects windows/doors and books `area_sf` NET of
# them, recording the breakdown in `sf_calc` {gross_sf, openings_sf, net_sf, n_openings}. That
# per-wall opening detection is COMPUTED but hidden at the zone level (grossArea==netArea,
# totalOpeningArea==0). These two helpers SURFACE it so the review card can show
# "gross X · ~N openings ≈ Y SF · [use net Z]" and one-click PRE-FILL the existing /set-net
# field. Read-only by construction: they return numbers to DISPLAY and add only NEW zone keys;
# they never touch area_sf/netArea/grossArea/totalOpeningArea (FIX_OPENINGS discipline — an
# opening deduction may be SHOWN, NEVER auto-applied to the booked total; /set-net still books).
def _poly_openings(p):
    """(detected_opening_sf, opening_count) for one piece, from the reader's own sf_calc. The
    reader books `area_sf` NET of these; when it computed no breakdown (markup / plain fill)
    there are 0 openings. Openings are counted here only to be DISPLAYED, never re-deducted."""
    sc = p.get("sf_calc") or {}
    osf = float(sc.get("openings_sf") or 0.0)
    if osf < 0:
        osf = 0.0
    on = int(sc.get("n_openings") or len(p.get("holes") or []) or 0)
    return osf, on

def _net_assist(d):
    """Additive net-assist fields for a zone from its aggregated opening breakdown (`openSF`,
    `openN`, `sf`). The booked net (`sf`) already has the reader's detected openings removed,
    so netSuggested == the booked net and grossSF == net + openings — i.e. gross - openings ==
    net EXACTLY (the task's Z, with no drift from a later rescale). Identity by construction:
    netArea / grossArea / totalOpeningArea are NOT among these keys, so no booked SF moves."""
    booked = round(float(d.get("sf") or 0.0), 1)
    osf = round(float(d.get("openSF") or 0.0), 1)
    if osf < 0:
        osf = 0.0
    return {
        "grossSF": round(booked + osf, 1),   # gross-of-detected-openings (consistent with net)
        "openingSF": osf,                    # SF the reader already cut out (shown, not re-applied)
        "openingsSuggested": int(d.get("openN") or 0),
        "netSuggested": booked,              # = gross - detected openings; one-click PRE-FILL for /set-net
    }

def extract_page_polygons(pg, pw, ph, ft_per_in):
    polys = []
    rot = pg.rotation_matrix                       # ROTATED pages (e.g. 270°) store annot vertices in unrotated coords;
    for a in (pg.annots() or []):                  # apply the page rotation so shapes align to the drawing (identity if rotation=0)
        if a.type[0] != 6:  # 6 = Polygon
            continue
        verts = a.vertices or []
        if len(verts) < 3:
            continue
        info = a.info or {}
        content = info.get("content", "") or ""
        subject = info.get("subject", "") or ""
        fill = a.colors.get("fill") if a.colors else None
        pts_pdf = []
        for v in verts:
            vx, vy = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v.x, v.y)
            p = fitz.Point(vx, vy) * rot
            pts_pdf.append((p.x, p.y))
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*sf", content, re.I)
        if m:
            sf = float(m.group(1).replace(",", "")); sf_exact = True   # estimator's own measured SF — ground truth
        else:
            sf = round(shoelace(pts_pdf) * (ft_per_in / 72.0) ** 2, 1); sf_exact = False  # geometry estimate (needs scale)
        norm = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in pts_pdf]
        cx = round(sum(p[0] for p in norm) / len(norm), 5)
        cy = round(sum(p[1] for p in norm) / len(norm), 5)
        polys.append({
            "id": len(polys), "points": norm, "area_sf": sf, "cx": cx, "cy": cy,
            "fill_color": [round(c, 3) for c in fill] if fill else None,
            "source": "bluebeam", "material": subject or None,
            "category": categorize(subject), "label": content, "sf_exact": sf_exact,
        })
    return polys

def _feet(s):
    """Parse feet-inches ('43\\'-3\"', '12\\'-0', '7\\'-2\"') → float feet."""
    m = re.search(r"(\d+)\s*'\s*-?\s*(\d+)?", s or "")
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 12.0

def _height_of(subj):
    m = re.search(r"H\s*[:=]\s*(\d+)\s*'\s*-?\s*(\d+)?", subj or "", re.I)
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 12.0

def _explicit_sf(s):
    m = re.search(r"-?\s*([\d,]+(?:\.\d+)?)\s*sf", s or "", re.I)
    return float(m.group(1).replace(",", "")) if m else None

def _clean_material(subj):
    return re.sub(r"\s*\(.*?\)\s*", " ", re.sub(r"-?\d+sf", "", subj or "", flags=re.I)).strip(" -:") or "Unlabeled"

def extract_page_polylines(pg, pw, ph):
    """Capture the estimator's LINEAR measurements (Bluebeam type-7 polylines) that the polygon path misses.
    Length is exact (in the label). Height in the subject '(H:..)' → SF for cladding runs; else LF (trim/soffit).
    Returns (sf_polys, lf_items): sf_polys look like polygon dicts (added to the takeoff SF total); lf_items are {material, lf}."""
    sf_polys, lf_items = [], []
    for a in (pg.annots() or []):
        if a.type[0] != 7:  # 7 = PolyLine (length measurement)
            continue
        info = a.info or {}
        content = info.get("content", "") or ""
        subject = info.get("subject", "") or ""
        length_ft = _feet(content)
        if not length_ft:  # not a length measurement (e.g. a plain leader line) → skip
            continue
        mat = _clean_material(subject)
        esf = _explicit_sf(subject) or _explicit_sf(content)
        h = _height_of(subject)
        stroke = a.colors.get("stroke") if a.colors else None
        if esf or h:  # a CLADDING run → SF (explicit, else length × wall height)
            sf = round(esf if esf else length_ft * h, 1)
            verts = a.vertices or []
            pts_pdf = [( (fitz.Point(v[0], v[1]) if isinstance(v, (list, tuple)) else fitz.Point(v.x, v.y)) * pg.rotation_matrix ) for v in verts]
            pts_pdf = [(p.x, p.y) for p in pts_pdf]
            norm = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in pts_pdf] or [[0, 0]]
            cx = round(sum(p[0] for p in norm) / len(norm), 5)
            cy = round(sum(p[1] for p in norm) / len(norm), 5)
            sf_polys.append({
                "points": norm, "area_sf": sf, "cx": cx, "cy": cy,
                "fill_color": [round(c, 3) for c in stroke] if stroke else None,
                "source": "bluebeam-linear", "material": mat, "category": categorize(subject),
                "label": f"{content} × {round(h,1) if h else '?'}' = {sf:,.0f} SF", "sf_exact": True,
                "length_ft": round(length_ft, 1),
            })
        else:  # trim / soffit / fascia → LINEAR feet (priced per LF, no SF)
            lf_items.append({"material": mat, "lf": round(length_ft, 1)})
    return sf_polys, lf_items

def flag_label_outliers(polys, pw, ph):
    """Money-safety: catch a typo'd/mislabeled SF before it reaches a bid.
    Every marked polygon on ONE sheet shares ONE scale, so back it out per-poly and flag any
    whose typed SF label disagrees with its geometry (area is rotation-invariant → safe).
    Non-destructive: only sets a warning, never changes an SF. Returns page-level warning strings."""
    scales = []
    for p in polys:
        if p.get("source") != "bluebeam" or not p.get("sf_exact") or p.get("area_sf", 0) <= 0:
            continue  # only true area polygons — linear runs have no meaningful shoelace area
        sh = shoelace(p["points"]) * pw * ph  # normalized-shoelace → PDF points^2
        if sh > 0:
            scales.append(((72.0 * (p["area_sf"] / sh) ** 0.5), p))  # implied ft-per-inch
    if len(scales) < 3:  # need a few labeled regions to establish the sheet's true scale
        return []
    med = sorted(s for s, _ in scales)[len(scales) // 2]
    if med <= 0:
        return []
    warns = []
    for s, p in scales:
        r = s / med
        if r > 1.32 or r < 0.76:  # area off by >~1.75x vs the sheet's consistent scale = almost certainly a mistake
            implied = p["area_sf"] * (med / s) ** 2
            p["sf_warn"] = True
            p["sf_warn_msg"] = f"labeled {p['area_sf']:,.0f} SF but this sheet's scale implies ≈{implied:,.0f} SF — verify"
            warns.append(f"{p.get('material') or p.get('category') or 'A region'}: " + p["sf_warn_msg"])
    return warns

_TB_MAX_OCR_PAGES = int(os.environ.get("TITLEBLOCK_MAX_OCR_PAGES", "150"))
_TB_OCR_BUDGET = {}   # id(doc) -> pages OCR'd (protects the request path on mega-sets)


def _tb_text_sheet(pg):
    """True when the title-block region carries a sheet-number-shaped token in the TEXT
    layer — i.e. the title block is readable without OCR."""
    try:
        W8, H8 = pg.rect.width, pg.rect.height
        rot8 = pg.rotation_matrix
        for w in pg.get_text("words"):
            q = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rot8
            if q.x >= W8 * 0.82 and q.y >= H8 * 0.80 \
                    and titleblock.SHEET_RE.match((w[4] or "").strip().upper()):
                return True
    except Exception:
        pass
    return False


def _titleblock_fallback(pg, t):
    """v3-ocr (2026-07-29 flattened re-audit): the 15 page-finding-blind jobs are NOT
    textless — their gold pages carry body text while the TITLE BLOCK is curves (no
    sheet number, no title in the text layer). When the text path cannot classify AND
    the title block is text-blind, OCR the corner (titleblock.py — prototype-graded
    60/60 sheet ids, 9/9 elevation titles, 0 FP on 26-222) and admit the page when the
    OCR'd TITLE says ELEVATION (or the sheet sits in the A4xx elevation series).
    conf<0.6 abstains → page stays non-elevation (flag-never-bind)."""
    try:
        if len(t) >= 60 and _tb_text_sheet(pg):
            return False        # title block IS textual — the text verdict stands
        dk = id(pg.parent)
        used = _TB_OCR_BUDGET.get(dk, 0)
        if used >= _TB_MAX_OCR_PAGES:
            return False
        if len(_TB_OCR_BUDGET) > 64:
            _TB_OCR_BUDGET.clear()
        _TB_OCR_BUDGET[dk] = used + 1
        v = titleblock.elevation_candidate(pg)
        return bool(v.get("admit"))
    except Exception:
        return False


_SITE_SHEET_PHRASES = ("SITE PLAN", "LANDSCAPE PLAN", "GRADING PLAN", "UTILITY PLAN",
                       "PAVING PLAN", "DRAINAGE PLAN", "EROSION CONTROL", "LIMIT OF WORK",
                       "PLANTING PLAN", "CIVIL SITE")
_WALL_MATERIAL_TOKENS = ("SIDING", "BRICK", "MASON", "EIFS", "STUCCO", "METAL PANEL",
                         "MTL PANEL", "FIBER CEMENT", "HARDIE", "NICHIHA", "CMU", "VENEER",
                         "CURTAIN WALL", "STOREFRONT", "SPANDREL", "CLADDING", "WALL TYPE",
                         "MATERIAL LEGEND", "MATERIAL SCHEDULE", "ALUCOBOND", "CLAPBOARD",
                         "SHINGLE", "SHAKE", "PRECAST")


def _is_site_civil_sheet(text):
    """True ONLY for an unambiguous site/civil sheet: it names itself a site / landscape /
    grading / utility / paving plan AND carries no wall-material keynote. Used to DEMOTE
    (never delete) a CONFIRMED coarse-scale page (>= SITE_BLINDBAND_FT_PER_IN, above every
    gold scale) to suggest_only on the AUTO path. Conservative by construction: a real wall
    elevation drawn coarse still carries material keynotes (fails clause 2); a sheet that does
    not name itself a site plan fails clause 1. Validated on the 5 admitted blind-band pages
    (demotes only the two Avalon site plans 26-063 p56/p57) and 30 gold pages (0 fired).
    """
    u = (text or "").upper()
    if not any(ph in u for ph in _SITE_SHEET_PHRASES):
        return False
    if any(tok in u for tok in _WALL_MATERIAL_TOKENS):
        return False
    return True


# ── 3D / AXONOMETRIC / RENDERING SHEET DETECTOR (WAVE 5+, SPEC 2, 2026-08-18) ──
# A page whose TITLE is a 3D view (axonometric / isometric / perspective / massing /
# rendering) is a picture of the building, not a flat wall elevation; auto-booking its
# vector/texture area measures MASSING as cladding and invents huge phantom SF (one
# 26-002 sheet alone books 346,085 SF). This is a TITLE-vs-TITLE test, NOT the class-c
# site/civil keynote test: an axonometric MASSING sheet DOES print material call-outs
# (it colours the massing by material), so _is_site_civil_sheet's "no wall keynote"
# clause wrongly SPARES those pages. Correct predicate: a STRONG 3D/rendering title is
# present AND no directional flat-elevation title. Material keynotes are irrelevant.
# "KEY PLAN" is titleblock/locator noise on 1,932 pages (every 26-002 sheet) and is
# DELIBERATELY EXCLUDED from the firing vocabulary. Census (rd_3d_detector_census.py,
# 122/122 jobs, corpus EE743811): fires on 109 pages, 33 currently ADMITTED across 16
# jobs, and ZERO of the 372 certified gold pages — a demoted real elevation would be
# confidently-wrong and there are none. Same demote-never-delete discipline as the
# class-c / blind-band fixes: out of zones/totals/Excel/evidence, still drawn, one click
# from real via /accept-suggestion. Scale-independent (3D sheets are admitted at any
# scale: 26-002 p129/26-182 p12 at 80 ft/in, 26-190 isometrics at 8 ft/in).
_3D_PHRASES = ("AXONOMETRIC", "AXONOMETRICS", "ISOMETRIC", "PERSPECTIVE",
               "3D VIEW", "3-D VIEW", "3D MASSING", "MASSING MODEL", "MASSING DIAGRAM",
               "MASSING STUDY", "AERIAL VIEW", "AERIAL PERSPECTIVE",
               "BIRD'S EYE", "BIRDS EYE", "BIRD'S-EYE", "EYE VIEW",
               "RENDERING", "RENDERINGS", "RENDERED VIEW", "EXTERIOR RENDERING",
               "3D RENDERING", "PHOTOREALISTIC", "MASSING")
# directional flat-elevation TITLE — the ONE exclusion (a real elevation says this)
_ELEVTITLE_RE = re.compile(r"\b(north|south|east|west|front|rear|left|right|side|"
                           r"partial|overall|exterior|building|typ(?:ical)?)\s+"
                           r"[\w\s]{0,18}elevation")


def _is_3d_massing_sheet(text):
    """True for a page whose TITLE is a 3D view / rendering / massing AND which carries
    NO directional flat-elevation title. Demote-only (suggest_only), never delete.
    Identical predicate to rd_3d_detector_census.demote_fires (the 0-gold-proven test)."""
    u = (text or "").upper()
    if not any(p in u for p in _3D_PHRASES):
        return False
    if _ELEVTITLE_RE.search((text or "").lower()):
        return False
    return True


def is_elevation_page(pg, ocr_fallback=True):
    """Auto-crop to the pages the estimator actually measures = elevations, returns, soffits.
    v2 (cold-run audit 2026-07-24: v1 text-only test found ALL gold pages on just 19/88
    archive jobs — blind on flattened sheets, excluded mixed sheets carrying a 'roof
    plan' note, over-included note-mentioning details): TITLE evidence outranks
    exclusions; geometric level-datum evidence rescues textless sheets.
    v3-ocr (2026-07-29): title-block OCR fallback for CURVE-TITLE sheets — the
    measured blind class has texty bodies but drawn title blocks; strictly additive
    (only ever ADDS admission; the doc-level gate stays text-only)."""
    import re as _re5
    try:
        t = (pg.get_text() or "").lower()
    except Exception:
        t = ""
    # 1) a real elevation TITLE anywhere = yes, regardless of plan notes on the sheet
    if _re5.search(r"\b(north|south|east|west|front|rear|left|right|side|partial|overall|exterior|building)\s+[\w\s]{0,18}elevation", t):
        return True
    # 2) plan-titled sheets without an elevation title = no — but a curve-title sheet
    # can carry plan NOTES in body text while its drawn title says ELEVATIONS: consult
    # the title-block OCR before rejecting (text-sheeted pages return False unchanged)
    if ("roof plan" in t) or ("floor plan" in t):
        return _titleblock_fallback(pg, t) if ocr_fallback else False
    if ("elevation" in t) or ("return" in t) or ("soffit" in t):
        return True
    # 3) textless/flattened sheets: LEVEL-DATUM geometry = elevation signature —
    #    but sections/details stack LEVELs too (v2 gate: 1,818 extras corpus-wide),
    #    so require the FACADE signature as well: several LONG horizontal structure
    #    lines (building base/story lines span elevations; sections are narrow cuts).
    if len(t) < 60:
        try:
            words = pg.get_text("words") or []
            lv = sum(1 for w in words if (w[4] or "").strip().upper() in ("LEVEL", "ROOF")
                     or (w[4] or "").strip().upper().startswith("T.O"))
            if lv >= 3:
                hl, _vl = _page_struct_lines(pg)
                W9 = pg.rect.width
                long_h = sum(1 for (x0, x1, _y) in hl if (x1 - x0) > 0.30 * W9)
                if long_h >= 3:
                    return True
        except Exception:
            pass
    return _titleblock_fallback(pg, t) if ocr_fallback else False

def _page_struct_lines(pg):
    """Long H/V vector lines of the page (same rule as /snap-points): the building's real geometry."""
    W, H = pg.rect.width, pg.rect.height
    Lmin = 0.06 * max(W, H)
    hlines, vlines = [], []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                segs = []
                if it[0] == "l":
                    segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
                elif it[0] == "re":
                    r = it[1]; segs = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                                       (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                for (x1, y1, x2, y2) in segs:
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 < Lmin ** 2:
                        continue
                    if abs(y2 - y1) < abs(x2 - x1) * 0.02:
                        hlines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
                    elif abs(x2 - x1) < abs(y2 - y1) * 0.02:
                        vlines.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
    except Exception:
        pass
    return hlines[:600], vlines[:600]

def snap_auto_polys(pg, polys, pw, ph):
    """The model paints a pixel mask — its outline is blobby and ignores the CAD geometry that is
    RIGHT THERE in the vector file. Make the AI's shapes look like an estimator drew them:
    simplify the contour, then snap each vertex to the drawing's real corners/lines (Bluebeam-style).
    SF is rescaled by the area ratio so the mask's net-of-openings measurement is preserved."""
    import numpy as np, cv2
    hlines, vlines = _page_struct_lines(pg)
    if not hlines and not vlines:
        return polys
    corners = []
    for (xa, xb, yh) in hlines:
        for (ya, yb, xv) in vlines:
            if xa - 2 <= xv <= xb + 2 and ya - 2 <= yh <= yb + 2:
                corners.append((xv, yh))
    corners = np.array(corners[:4000], dtype=np.float32) if corners else None
    hy = np.array([y for (_, _, y) in hlines], dtype=np.float32) if hlines else None
    vx = np.array([x for (_, _, x) in vlines], dtype=np.float32) if vlines else None
    tol = 0.012 * max(pw, ph)   # snap radius ~1.2% of the sheet
    out = []
    for p in polys:
        try:
            pts = np.array([(x * pw, y * ph) for x, y in p["points"]], dtype=np.float32)
            if len(pts) < 3:
                out.append(p); continue
            orig_area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
            appr = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 0.012 * cv2.arcLength(pts.reshape(-1, 1, 2), True), True).reshape(-1, 2)
            if len(appr) < 3:
                out.append(p); continue
            snapped = []
            for (x, y) in appr:
                nx, ny = float(x), float(y)
                hit = False
                if corners is not None and len(corners):
                    d = np.hypot(corners[:, 0] - nx, corners[:, 1] - ny)
                    i = int(d.argmin())
                    if d[i] < tol:
                        nx, ny = float(corners[i][0]), float(corners[i][1]); hit = True
                if not hit:  # no true corner nearby — align edges to the nearest structural line
                    if vx is not None and len(vx):
                        dv = np.abs(vx - nx); i = int(dv.argmin())
                        if dv[i] < tol: nx = float(vx[i])
                    if hy is not None and len(hy):
                        dh = np.abs(hy - ny); i = int(dh.argmin())
                        if dh[i] < tol: ny = float(hy[i])
                if not snapped or abs(snapped[-1][0] - nx) > 1 or abs(snapped[-1][1] - ny) > 1:
                    snapped.append((nx, ny))
            if len(snapped) < 3:
                out.append(p); continue
            new_area = abs(cv2.contourArea(np.array(snapped, dtype=np.float32).reshape(-1, 1, 2)))
            q = dict(p)
            q["points"] = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in snapped]
            if orig_area > 0 and new_area > 0:
                q["area_sf"] = round(p.get("area_sf", 0) * new_area / orig_area, 1)
            out.append(q)
        except Exception:
            out.append(p)
    return out

# ── VECTOR-PAGE SUGGESTIONS (STEP 3, PERFECT_ACCURACY_BLUEPRINT) ────────────
# The suggestion block below has two branches — DENSE hairline pages and RASTER
# underlays — and a CLEAN VECTOR elevation is neither, so those pages shipped
# ZERO suggestions (confirmed against live). That is the wrong half of the gap:
# r4_truly_missed_partition counted 869 walls / 246k SF with no engine ink on
# them at all, and a suggestion is the difference between "she draws it" and
# "she clicks it".
#
# `vector_hatch._v13_regions` already SELF-GATES onto vector sheets (>=15% of
# the page in non-white fill paths, else one cheap 384px probe that proceeds
# only when the model's wall-interior response covers >=8%), so calling it here
# costs nothing on sparse CAD line sheets and admits exactly the rendered-vector
# class it was widened for (Avalon / Rivers-Edge / Essex: 30/15/37 walls found
# standalone where every hand-written gate excluded them).

def _piece_bbox(p):
    pts = p.get("points") or []
    if not pts:
        return None
    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_cover(a, b):
    """Fraction of box `a` that box `b` covers (normalised page coords, 0..1)."""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    return (ix * iy) / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def _vector_page_suggestions(job, pdf_bytes, pi, polys, pw, ph, scale_val, max_new=40):
    """v13 pieces on a clean VECTOR elevation, minus everything the engine already booked.

    TWO THINGS PROTECT THE PRIMARY READ, because a suggestion that costs booked SF is
    not identity-by-default and would be the fill demotion all over again:

    1. THE BUDGET IS SEPARATE.  `vector_hatch._V13_BUDGET` is a per-DOCUMENT counter
       ({doc_fp: pages_inferred}) and on a hit `_v13_regions` WITHHOLDS work — it
       returns [] rather than a cached answer (that is the whole of
       FIX_V13_NONDETERMINISM: a spent budget silently cost 34% of a job's SF). The
       booked read spends that budget at vector_hatch.py:1509. If suggestions spent it
       too, a doc would reach V13_MAX_PAGES in HALF the pages and the primary reader
       would go dark on the rest — booked SF down, silently. So the shared counter is
       snapshotted and RESTORED around the call, and suggestions run against their own
       per-job page cap (V13_SUGGEST_MAX_PAGES, default = V13_MAX_PAGES). The primary
       read's budget is therefore provably untouched: it sees the same number after
       this function as before it, on every path including the failure paths.

    2. IF v13 ALREADY RAN ON THIS PAGE, DON'T RUN IT AGAIN.  Its own ownership filter
       ("virgin territory only", vector_hatch.py:~2280) already dropped every candidate
       overlapping a booked poly — which is precisely the subtraction below. A second
       full inference would spend ~40s to produce the empty set. Detected by the
       reader_class stamp, which is written once and never renamed.

    The pieces returned are SUBTRACTED against booked polys, so this surfaces only what
    the engine did not book, and the caller tags every one of them suggest_only.
    """
    if any((p.get("reader_class") or "") == "v13" for p in polys):
        return []
    bud = getattr(vector_hatch, "_V13_BUDGET", None)
    if not isinstance(bud, dict):
        # cannot prove isolation from the primary read's budget -> do not risk it
        return []
    try:
        cap = int(os.environ.get("V13_SUGGEST_MAX_PAGES") or os.environ.get("V13_MAX_PAGES") or 8)
    except Exception:
        cap = 8
    used = int(job.get("_v13SugPages") or 0)
    if used >= cap:
        return []
    fp = (len(pdf_bytes), hash(pdf_bytes[:4096]))
    had = fp in bud
    saved = bud.get(fp, 0)
    sugs, ran = [], False
    try:
        bud[fp] = 0                     # suggestions run on their OWN budget
        sugs = vector_hatch._v13_regions(pdf_bytes, pi, [], pw, ph,
                                         float(scale_val or 8.0) / 72.0, max_new=max_new) or []
        ran = bool(bud.get(fp, 0))      # the budget gate is past the cheap self-gates:
    except Exception:                   # a bump means inference actually ran on this page
        sugs = []                       # a reader that fails costs THIS page's suggestions,
    finally:                            # never the page's probe, its flags or a single SF
        if had:
            bud[fp] = saved
        else:
            bud.pop(fp, None)
    if ran:
        job["_v13SugPages"] = used + 1
    owned = [b for b in (_piece_bbox(p) for p in polys if not p.get("suggest_only")) if b]
    out = []
    for s in sugs:
        b = _piece_bbox(s)
        if b is None:
            continue
        if any(_bbox_cover(b, ob) > 0.3 for ob in owned):
            continue                    # the engine already booked this territory
        out.append(s)
    return out


def process(jid, pdf_bytes):
    job = jobs[jid]
    # FIX_V13_NONDETERMINISM (2026-08-06): scope the v13 inference budget to THIS analysis.
    # vector_hatch._V13_BUDGET is a process-lifetime dict {doc_fp: pages_inferred} that caps
    # v13 inference per document; on a HIT it WITHHOLDS work (returns []), it does not
    # memoise. On a long-lived worker the count survived between uploads, so re-analysing
    # the same PDF began with the budget already spent and _v13_regions returned [] for
    # every page -- the boundary reader silently dropped out (38,172.6 -> 25,112.9 SF,
    # -34.21%, measured live). Dropping THIS document's fingerprint makes every upload start
    # from a fresh budget. It does not raise the cap or the per-page limit, and for a
    # never-seen fingerprint (every cold single-pass grade) it is a no-op, so first-run SF
    # is unchanged by construction.
    vector_hatch.reset_v13_budget(pdf_bytes)
    try:
        job["status"] = "running"; job["phase"] = "loading"
        job["progress"] = {"label": "Loading PDF", "pct": 3}
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = doc.page_count
        job["pageCount"] = n   # Draw tab needs the full page list — photo/sparse sheets have no takeoffData entry
        jlog(job, f"PDF loaded — {n} page(s)", "ok")
        legend = {}
        auto_used = 0  # cap expensive texture renders so a big unmarked PDF can't blow memory
        # If the estimator marked ANY page, this is a DIGITIZE job — don't texture-auto the other
        # pages (they're plans/details, not cladding) or we inflate the bid with invented SF.
        # Count only annotations the digitize path actually READS as measurements: a polygon
        # with real vertices, or a polyline whose label carries a length. Stray leader lines,
        # clouds, or review stamps in an issued set must NOT disable auto-detect (that made a
        # "no markups" full set return a completely empty takeoff).
        doc_has_markup = False
        for pi in range(n):
            try:
                for a in (doc[pi].annots() or []):
                    t = a.type[0]
                    if t == 6 and len(a.vertices or []) >= 3:
                        doc_has_markup = True; break
                    if t == 7 and _feet((a.info or {}).get("content", "") or ""):
                        doc_has_markup = True; break
            except Exception:
                pass
            if doc_has_markup:
                break
        # auto-crop: which pages are elevations? (only mark those; if none detectable, fall back to all)
        # Text-based classification only works if the PDF actually HAS text — many issued sets
        # flatten CAD text to curves (a 77-page set had text on 1 page). If too few pages carry
        # text, the filter is meaningless: turn it OFF and try every page instead of page 6 only.
        text_pages = sum(1 for pi in range(n) if len(doc[pi].get_text() or "") > 150)
        text_reliable = text_pages >= max(3, int(n * 0.3))
        doc_any_elevation = text_reliable and any(is_elevation_page(doc[pi], ocr_fallback=False) for pi in range(n))  # doc gate stays text-only — OCR admission is per-page/additive
        if not text_reliable and n > 1:
            jlog(job, f"Drawing text not extractable ({text_pages}/{n} pages with text) — scanning every page for cladding", "warn")
        auto_tried = 0; auto_hits = 0
        # DUPLICATE-PAGE DEDUP (WAVE 5 Item 1, 2026-08-17). Some issued sets bind the same
        # physical sheet 2+ times (byte-identical /Contents stream — e.g. 26-171 p28==p16,
        # p30==p25; census w5_overdraw_census.json: 21 of 122 jobs, 58 elevation-ish extras).
        # On the AUTO path each copy auto-books the SAME SF independently = double-count. A
        # byte-identical page IS the same sheet, so the first occurrence is read normally and
        # every later copy contributes 0 new SF. The key is the FULL content-stream md5 —
        # identical to w5_overdraw_census.py:215, md5(page.read_contents()) — a FULL byte
        # match, never a near-match, and the FIRST occurrence is never skipped. GATED to the
        # AUTO path (`not doc_has_markup`): on the digitize/marked path SF comes from the
        # estimator's OWN annotations, not the /Contents, so two pages with identical /Contents
        # but different markup are legitimately different takeoffs and must never be deduped.
        _seen_page_content = {}   # content_md5 -> first 1-based page booked on it
        _job_t0 = time.monotonic()   # per-JOB wall-clock cap start (reliability guard)
        _ocr_spent = 0.0             # cumulative title-block OCR admission time (reliability guard)
        for pi in range(n):
            job["progress"] = {"label": f"Reading page {pi+1} of {n}", "pct": 5 + int(90 * pi / max(n, 1))}
            job["phase"] = "analyzing"
            # PER-JOB WALL-CLOCK CAP (2026-08-18): a single upload must never hold the single
            # prod worker long enough to starve every concurrent job. Once the budget is spent,
            # stop admitting pages, record how many were skipped, and finalize the pages already
            # done — return-or-fail-fast, never hang. The finished pages keep their exact booked
            # SF; the remainder are surfaced as needs-review, never as a silent zero.
            if (time.monotonic() - _job_t0) > T_JOB_SECONDS:
                _skipped = n - pi
                job["partialReview"] = {"skippedPages": _skipped, "reason": "job_time_budget",
                                        "budgetSeconds": T_JOB_SECONDS}
                jlog(job, f"⏱ Job time budget {T_JOB_SECONDS:.0f}s exceeded after {pi} page(s) — "
                          f"{_skipped} remaining page(s) skipped for manual review; returning a "
                          f"PARTIAL takeoff of what finished.", "warn")
                break
            pg = doc[pi]; pw, ph = pg.rect.width, pg.rect.height
            if not doc_has_markup:   # dedup only where booking is driven by /Contents (AUTO path)
                try:
                    _cmd5 = hashlib.md5(pg.read_contents()).hexdigest()
                except Exception:
                    _cmd5 = None
                if _cmd5 is not None:
                    _first_pg = _seen_page_content.get(_cmd5)
                    if _first_pg is not None:
                        # exact duplicate of an already-booked sheet -> 0 new SF
                        jlog(job, f"Page {pi+1}: duplicate of page {_first_pg} (identical "
                                  f"content) — skipped, contributes 0 SF", "info")
                        job["polygons_by_page"][pi + 1] = []
                        job["dims_by_page"][pi + 1] = {"width": pw, "height": ph}
                        continue
                    _seen_page_content[_cmd5] = pi + 1
            ft = 8.0  # scale fallback; digitize SF comes from the markup's own labels
            polys = extract_page_polygons(pg, pw, ph, ft)
            lin_sf_polys, lin_lf_items = extract_page_polylines(pg, pw, ph)  # estimator's linear measurements
            polys = polys + lin_sf_polys
            for i, p in enumerate(polys):
                p["id"] = i  # unique ids across polygons + linear runs
            auto = False; scale_conf = True; scale_val = None; auto_engine = None
            site_scale_flags = []   # survives even when the page ends up producing no auto regions
            review_flags = []       # per-page "too complex — review manually" (0 SF); reliability guard
            # RELIABILITY (2026-08-18): is_elevation_page(ocr_fallback=True) runs title-block OCR
            # on every non-text-elevation page, and that OCR is the measured #1 hang (26-232:
            # 128s of a 147s run — 21 pages, 14.4s max; 26-180A: 326s over 26 pages).
            #  * DIGITIZE path (doc_has_markup): page_is_elev is DEAD — its only readers (the
            #    auto-detect block and the raster-suggestion block below) are both gated on
            #    `not doc_has_markup` — so computing it, and paying the OCR, is pure waste. Skip
            #    it (provably behaviour-identical: every reader is already ANDed with that gate).
            #  * AUTO path: keep OCR admission, but stop once the cumulative OCR budget is spent
            #    (scanned/flattened sets) — remaining pages use TEXT-ONLY admission so the job
            #    can't be run for minutes discovering elevations one 14s OCR at a time. A NORMAL
            #    job never reaches the budget (OCR rarely fires on text sheets) so this is a
            #    no-op there; the value below is byte-identical to the old expression whenever
            #    the budget is not exhausted.
            if doc_has_markup:
                page_is_elev = False
            elif not doc_any_elevation:
                page_is_elev = True                    # inversion: all pages admitted (no OCR)
            elif _ocr_spent > T_OCR_BUDGET_S:
                page_is_elev = is_elevation_page(pg, ocr_fallback=False)   # text-only, budget spent
            else:
                _t_ocr = time.monotonic()
                page_is_elev = is_elevation_page(pg)
                _ocr_spent += time.monotonic() - _t_ocr
            if not polys and not doc_has_markup and page_is_elev:
                # RAW/clean page -> auto-markup. Engine order: VECTOR (reads the drawn pattern —
                # exact geometry, openings netted; cheap, so EVERY page gets it) -> trained MODEL
                # -> texture heuristics (heavy, capped so a 77-page set can't stall the worker).
                try:
                    auto_tried += 1
                    tpolys = []; sinfo = {}
                    try:
                        # RELIABILITY GUARD (2026-08-18): vector_hatch.detect cost scales with the
                        # page's drawing-segment count — legit elevations top out ~23s, but civil/
                        # detail mega-sheets (951k segs) run 120-213s and hang the single worker.
                        # PRE-CHECK skips a page too dense to finish in budget (no worker thread is
                        # even spawned); otherwise the call is bounded to T_PAGE_SECONDS and an
                        # over-budget page is ABANDONED and marked needs-review, never left with a
                        # half-computed SF. rel_guard absent (deploy window) => runs unbounded.
                        if _rg is not None and _rg.too_dense(pg)[0]:
                            raise _GuardTimeout("pre-check: page too dense to auto-read in budget")
                        tpolys, _, _, sinfo = (
                            _rg.call_with_timeout(lambda: vector_hatch.detect(pdf_bytes, pi))
                            if _rg is not None else vector_hatch.detect(pdf_bytes, pi))
                        auto_engine = "vector"
                        # ⭐ STAMP THE READER CLASS HERE, AND ONLY HERE (FIX_TWO_TIER_GATE).
                        # The class is read off the `material` PREFIX, and `material` is
                        # overwritten twice below — callouts.name_regions (callouts.py:560/
                        # 588/655) and the per-job code map (~:790) — so by the time a piece
                        # reaches the response its provenance is gone and every region would
                        # collapse into one bucket wearing the wrong risk. This is the same
                        # moment `gold_bench.py:370` reads the same strings, which is the
                        # ONLY reason the corpus-measured risks in review_rank.RISK apply to
                        # these pieces verbatim. Additive: no SF, no grouping, no total.
                        if _rr is not None:
                            for _p9 in tpolys:
                                _p9["reader_class"] = _rr.classify(_p9.get("material"))
                    except _GuardTimeout:
                        tpolys = []
                        if not review_flags:
                            review_flags.append(PAGE_REVIEW_MSG)
                        _nsg = _rg.page_segment_count(pg) if _rg is not None else -1
                        jlog(job, f"Page {pi+1}: auto-detect exceeded the {T_PAGE_SECONDS:.0f}s "
                                  f"page budget ({_nsg:,} drawing segments) — skipped for manual "
                                  f"review (0 SF booked)", "warn")
                    except Exception:
                        tpolys = []
                    if not tpolys and not review_flags and auto_used < MAX_AUTO_PAGES:
                        # Same reliability bound on the fallback engines: texture.detect calls
                        # texture._read_scale, which iterates the page geometry (or OCRs a scanned
                        # sheet) and can blow up (120-150s in the census). Skip if the page already
                        # failed the vector guard above; otherwise bound the call the same way.
                        try:
                            if model_infer.available():
                                tpolys, _, _, sinfo = (
                                    _rg.call_with_timeout(lambda: model_infer.detect(pdf_bytes, pi, zoom=2.0))
                                    if _rg is not None else model_infer.detect(pdf_bytes, pi, zoom=2.0))
                                auto_engine = "model"
                            else:
                                tpolys, _, _, sinfo = (
                                    _rg.call_with_timeout(lambda: texture.detect(pdf_bytes, pi, ft_per_in=ft, zoom=2.0))
                                    if _rg is not None else texture.detect(pdf_bytes, pi, ft_per_in=ft, zoom=2.0))
                                auto_engine = "texture"
                        except _GuardTimeout:
                            tpolys = []
                            if not review_flags:
                                review_flags.append(PAGE_REVIEW_MSG)
                            jlog(job, f"Page {pi+1}: fallback auto-detect exceeded the "
                                      f"{T_PAGE_SECONDS:.0f}s page budget — skipped for review (0 SF)", "warn")
                        # The fallback engines are NOT in the corpus — every graded wall came
                        # from vector_hatch.detect. They are stamped with their engine, which
                        # review_rank has no risk row for, so they rank FIRST (unmeasured is
                        # not safe). Giving them a vector class would be borrowing somebody
                        # else's measurement.
                        for _p9 in tpolys:
                            _p9["reader_class"] = auto_engine
                        if tpolys:
                            try:
                                tpolys = snap_auto_polys(pg, tpolys, pw, ph)  # lock AI shapes to the drawing's real corners
                            except Exception:
                                pass
                    # ---- SITE/CIVIL SCALE = NOT A CLADDING SHEET (2026-08-06) -----------
                    # The detector has now told us the scale IT believes, read from this
                    # page's own printed note and CONFIRMED. Every confidence mechanism in
                    # the engine asks "was the scale read correctly?" and none asks "what
                    # kind of drawing is this?", so a correctly-read site scale is the one
                    # error class that passes every guard (FIX_ENGINE_BLOWUP.md §4). Only
                    # a CONFIRMED scale disqualifies: an unconfirmed reading is already
                    # handled by the calibrate-before-trusting flag below, and guessing a
                    # page away on an unconfirmed number is how the 08-04 scale patch
                    # withheld 42.2% of gold SF and dropped the frozen exam 140 -> 107.
                    # AUTO PATH ONLY — this whole branch is inside
                    # `if not polys and not doc_has_markup and page_is_elev`, so the
                    # marked/digitize path (the exact 100%-SF tier) cannot reach it.
                    if tpolys:
                        try:
                            _ss_val = float(sinfo.get("ft_per_in") or 0.0)
                        except Exception:
                            _ss_val = 0.0
                        if bool(sinfo.get("scale_confirmed")) and _ss_val >= SITE_SCALE_FT_PER_IN:
                            _ss_sf = sum(float(p.get("area_sf") or 0) for p in tpolys)
                            site_scale_flags.append(
                                f"⚠ SITE SHEET EXCLUDED — this page prints 1\"={_ss_val:g}', a site/civil "
                                f"scale, not a wall elevation (an elevation is 1/8\"=1'-0\" = 8 ft/in). "
                                f"{len(tpolys)} auto-read region(s) totalling {_ss_sf:,.0f} SF were LEFT OUT "
                                f"of the takeoff: at this scale the reader is measuring LAND, not cladding. "
                                f"If this really is a wall elevation, calibrate the page and re-run it.")
                            jlog(job, f"Page {pi+1}: 1\"={_ss_val:g}' is a site/civil scale — "
                                      f"{len(tpolys)} region(s) / {_ss_sf:,.0f} SF excluded from the takeoff", "warn")
                            tpolys = []
                            page_is_elev = False   # also suppresses the raster-page AI suggestions below
                        elif _ss_val >= SITE_SCALE_FT_PER_IN:
                            # ---- GUARD SYMMETRY (2026-08-17) --------------------------------
                            # The branch above demands a CONFIRMED reading before it will act.
                            # `texture.py:239` demands nothing: `ft_per_in = sc` BOOKS the page
                            # at whatever that same call returned, confirmed or not. So the
                            # engine books on an unconfirmed site scale and only excludes on a
                            # confirmed one — measured on 26-232 p4, which prints 1"=400' (a
                            # civil sheet: "...WHERE NEW PAVEMENT MEETS EXISTING PAVEMENT..."),
                            # reads 400 UNCONFIRMED, clears the cut by 4x, and books 2.8-4.25M
                            # SF — 91% of that job's off-page total (w3c_guard_census.json).
                            #
                            # DEMOTE, NEVER DELETE. The branch above may delete because a
                            # CONFIRMED coarse read is positive evidence about the sheet. An
                            # unconfirmed one is an ABSENCE of evidence, and the 08-04 scale
                            # patch is the standing lesson about acting on those: it withheld
                            # 42.2% of gold SF and dropped the frozen exam 140 -> 107. So these
                            # regions stay in the payload as suggest_only — out of zones,
                            # totals, Excel and evidence, still drawn, one click from real
                            # through /accept-suggestion (they are stamped with an id at :1209
                            # like every other piece, so the index and the accept path both
                            # reach them). Nothing is asserted and nothing is destroyed.
                            #
                            # SHIP CONDITION, MEASURED BEFORE THE EDIT (w3c_guard_census.py,
                            # cross-checked 372/372 against r7x_scale_census): ZERO of the 372
                            # gold pages in the 122-job pin read >= this cut unconfirmed — the
                            # coarsest unconfirmed page carrying gold anywhere is 26.8 ft/in,
                            # 3.7x below it. So this cannot park a wall she marked. The battery
                            # fixtures are not in that pin and were censused separately
                            # (w4_guard_pregate_fixtures.json, 10/10): neither has a page that
                            # qualifies, so a moved battery pin here means this did something
                            # it was not asked to do.
                            #
                            # `material` is deliberately NOT renamed to the suggestion tier's
                            # label: she needs to see WHAT the engine thinks it found to judge
                            # it, and an accepted piece must keep the reader class it was
                            # stamped with at :993. `page_is_elev` is left alone too — the
                            # suggestion probe below already ran on this page today, and this
                            # is one change: booked -> suggested.
                            _ss_sf = sum(float(p.get("area_sf") or 0) for p in tpolys)
                            for _p9 in tpolys:
                                _p9["suggest_only"] = True
                                _p9["site_scale_demoted"] = True
                            site_scale_flags.append(
                                f"⚠ SITE/CIVIL SCALE, UNCONFIRMED — this page reads 1\"={_ss_val:g}', a "
                                f"site scale rather than a wall elevation (an elevation is 1/8\"=1'-0\" "
                                f"= 8 ft/in), but the reading is NOT confirmed by a printed note. "
                                f"{len(tpolys)} auto-read region(s) totalling {_ss_sf:,.0f} SF are SHOWN "
                                f"AS SUGGESTIONS instead of being counted: at this scale the reader "
                                f"would be measuring LAND, not cladding. If this really is a wall "
                                f"elevation, calibrate the page — or accept the regions individually.")
                            jlog(job, f"Page {pi+1}: 1\"={_ss_val:g}' is a site/civil scale and UNCONFIRMED — "
                                      f"{len(tpolys)} region(s) / {_ss_sf:,.0f} SF demoted to suggestions "
                                      f"(not counted in the takeoff, one click from real)", "warn")
                        elif (bool(sinfo.get("scale_confirmed"))
                              and SITE_BLINDBAND_FT_PER_IN <= _ss_val < SITE_SCALE_FT_PER_IN
                              and _is_site_civil_sheet(pg.get_text() or "")):
                            # ---- BLIND-BAND SITE/CIVIL SHEET (WAVE 5 Item 2, 2026-08-17) ------
                            # A CONFIRMED scale coarser than any elevation but under the 100 cut
                            # (e.g. an architectural SITE PLAN at 1"=30') passes both guards above
                            # and books phantom SF — measuring parking, courtyards and paving as
                            # cladding. Only demote when the page ALSO names itself a site/civil
                            # sheet AND carries no wall-material keynote (_is_site_civil_sheet), so
                            # a real large-building elevation drawn coarse is never touched. The cut
                            # (24 ft/in) is above the coarsest CONFIRMED gold scale (21.33; 0 of 372
                            # gold pages read >= 22 confirmed — r7x), so this can never park a wall
                            # she marked. DEMOTE, NEVER DELETE — same policy as the unconfirmed
                            # branch: out of zones/totals/Excel/evidence, still drawn, one click
                            # from real through /accept-suggestion.
                            _ss_sf = sum(float(p.get("area_sf") or 0) for p in tpolys)
                            for _p9 in tpolys:
                                _p9["suggest_only"] = True
                                _p9["site_scale_demoted"] = True
                            site_scale_flags.append(
                                f"⚠ SITE/CIVIL SHEET — this page reads a CONFIRMED 1\"={_ss_val:g}', "
                                f"coarser than any wall elevation, and its text names a site/civil "
                                f"plan with no wall-material callout. {len(tpolys)} auto-read "
                                f"region(s) totalling {_ss_sf:,.0f} SF are SHOWN AS SUGGESTIONS "
                                f"instead of being counted: at this scale the reader would be "
                                f"measuring LAND, not cladding. If this really is a wall elevation, "
                                f"calibrate the page — or accept the regions individually.")
                            jlog(job, f"Page {pi+1}: 1\"={_ss_val:g}' CONFIRMED site/civil sheet (blind band) — "
                                      f"{len(tpolys)} region(s) / {_ss_sf:,.0f} SF demoted to suggestions "
                                      f"(not counted in the takeoff, one click from real)", "warn")
                        elif _is_3d_massing_sheet(pg.get_text() or ""):
                            # ---- 3D / AXONOMETRIC / RENDERING SHEET (SPEC 2, 2026-08-18) -----
                            # This page's TITLE is a 3D view, not a flat wall elevation (see
                            # _is_3d_massing_sheet: a STRONG 3D/rendering title AND no directional
                            # flat-elevation title). SCALE-INDEPENDENT — a 3D sheet is admitted at
                            # any scale (26-002 p129 at 80 ft/in, 26-190 isometrics at 8 ft/in), so
                            # this branch runs whatever _ss_val is. Booking its vector/texture area
                            # measures MASSING as cladding = huge phantom SF (26-002 p129 = 346,085).
                            # DEMOTE, NEVER DELETE — out of zones/totals/Excel/evidence, still drawn,
                            # one click from real via /accept-suggestion (pieces are stamped with an
                            # id like every other, so index + accept both reach them). 0 of 372 gold
                            # pages fire the predicate (rd_3d_detector_census, zero_gold_movement) —
                            # this cannot park a real elevation she marked. `material` is NOT renamed
                            # (she must see WHAT it found to judge it); `page_is_elev` is left alone
                            # like the class-c branch so raster-page suggestions still surface.
                            _ss_sf = sum(float(p.get("area_sf") or 0) for p in tpolys)
                            for _p9 in tpolys:
                                _p9["suggest_only"] = True
                                _p9["three_d_demoted"] = True
                            site_scale_flags.append(
                                f"⚠ 3D / AXONOMETRIC / RENDERING sheet — this page's title is a 3D "
                                f"view, not a flat wall elevation. {len(tpolys)} region(s) / "
                                f"{_ss_sf:,.0f} SF are SHOWN AS SUGGESTIONS instead of counted (3D "
                                f"massing measured as flat cladding is not real SF). If this really "
                                f"is a flat elevation, accept the regions individually.")
                            jlog(job, f"Page {pi+1}: 3D / axonometric / rendering sheet — "
                                      f"{len(tpolys)} region(s) / {_ss_sf:,.0f} SF demoted to suggestions "
                                      f"(not counted in the takeoff, one click from real)", "warn")
                    if tpolys:
                        # ── SIZE-OUTLIER GATE (Feature 3, so_sizegate_census) — demote page-spanning
                        # phantoms, NEVER delete. Per REGION (unlike the whole-page site/3D branches):
                        # a piece covering > SIZE_OUTLIER_FRAC of the page is a boundary/background box,
                        # not a wall. Scale-independent (normalized frame). Carve-outs mirror the census
                        # safe predicate: never a v13 or markup read (the confirmed/high-trust classes),
                        # never one already demoted. 0 of 372 gold pages carry a region this large (floor
                        # 0.3515), so it cannot park a wall she marked. suggest_only => out of zones/
                        # totals/Excel/evidence, still drawn, one click from real via /accept-suggestion.
                        _so_hit = []
                        for _ps in tpolys:
                            if _ps.get("suggest_only") or (_ps.get("reader_class") or "") in ("v13", "markup"):
                                continue
                            _pts = _ps.get("points") or []
                            if len(_pts) < 3:
                                continue
                            try:
                                _frac = shoelace(_pts)   # normalized points => fraction of the page area
                            except Exception:
                                continue
                            if _frac > SIZE_OUTLIER_FRAC:
                                _ps["suggest_only"] = True
                                _ps["size_outlier_demoted"] = True
                                _ps["size_outlier_frac"] = round(_frac, 4)
                                _so_hit.append(_ps)
                        if _so_hit:
                            _so_sf = sum(float(p.get("area_sf") or 0) for p in _so_hit)
                            site_scale_flags.append(
                                f"⚠ OVERSIZED REGION(S) — {len(_so_hit)} region(s) / {_so_sf:,.0f} SF each cover "
                                f"more than {int(SIZE_OUTLIER_FRAC * 100)}% of the sheet, larger than any real wall "
                                f"on a drawing. SHOWN AS SUGGESTIONS instead of counted (a page-spanning box is "
                                f"usually a boundary/background, not cladding). If one really is a wall, accept it.")
                            jlog(job, f"Page {pi+1}: {len(_so_hit)} oversized region(s) / {_so_sf:,.0f} SF demoted "
                                      f"to suggestions (> {int(SIZE_OUTLIER_FRAC * 100)}% of the page — one click from real)", "warn")
                        try:  # read the architect's own labels: callout text + leader arrows -> region names
                            n_named = callouts.name_regions(pdf_bytes, pi, tpolys, pw, ph)
                            if n_named:
                                jlog(job, f"Page {pi+1}: {n_named} region(s) named from the drawing's callouts", "ok")
                        except Exception:
                            pass
                        polys = tpolys; auto = True; auto_used += 1; auto_hits += 1
                        scale_conf = bool(sinfo.get("scale_confirmed")); scale_val = sinfo.get("ft_per_in")
                except Exception as te:
                    jlog(job, f"Page {pi+1}: auto-detect skipped ({te})", "warn")
            sf_warns = flag_label_outliers(polys, pw, ph) if not auto else []  # catch typo'd markup labels
            # RASTER-PAGE v13 SUGGESTIONS (2026-07-21; the 5-iteration reclaim verdict:
            # no runtime signal separates junk floods from real walls, so the HUMAN is
            # the discriminator). On raster-underlay pages, run the boundary model with
            # EMPTY ownership and surface its pieces as suggest_only: never counted in
            # zones/totals/Excel/evidence until the estimator accepts each one.
            if not doc_has_markup and page_is_elev and not review_flags:
                # (runs even when auto-detect found NOTHING — the emptiest raster
                # pages need suggestions most; acid test: p19 had 0 pieces, 0 sugs)
                # `not review_flags`: a page the guard abandoned as too complex must not then
                # spend more of the budget running the v13/density suggestion probe on it.
                try:
                    _imga = 0.0
                    for _im9 in pg.get_images(full=True):
                        for _r9 in pg.get_image_rects(_im9[0]):
                            _imga += max(0, _r9.width) * max(0, _r9.height)
                    _sugs = []
                    # DENSITY FIRST (Callahan acid 2026-07-21: dense hairline pages often
                    # ALSO carry a backdrop image, so the raster test stole them and — via
                    # an indent bug — dropped the pieces; 40k hairlines is the more
                    # specific signature, so it wins the tie).
                    _dense9 = density_reader.is_dense_page(pg)
                    if _dense9:
                        # Callahan-class micro-texture: the density reader's fields
                        # become confirmable suggestions (probe-proven 22/114 ensemble)
                        _sugs = density_reader.suggest_pieces(pdf_bytes, pi, pw, ph, max_new=40)
                        for _s9 in _sugs:
                            _s9["reader_class"] = "density"   # not in the corpus — unmeasured
                    elif _imga >= 0.25 * pw * ph:
                        _ftpt9 = float(scale_val or 8.0) / 72.0
                        _sugs = vector_hatch._v13_regions(pdf_bytes, pi, [], pw, ph, _ftpt9, max_new=40)
                        # stamped BEFORE the rename below overwrites `material` with
                        # "AI suggestion (confirm to add)" — an accepted suggestion must
                        # keep the risk of the reader that drew it, not lose it at the click
                        if _rr is not None:
                            for _s9 in _sugs:
                                _s9["reader_class"] = _rr.classify(_s9.get("material"))
                    else:
                        # CLEAN VECTOR ELEVATION — the branch that did not exist (STEP 3).
                        # Everything protecting the booked read lives in the helper; the
                        # stamp goes on here for the same reason it does above.
                        _sugs = _vector_page_suggestions(job, pdf_bytes, pi, polys, pw, ph,
                                                         scale_val, max_new=40)
                        if _rr is not None:
                            for _s9 in _sugs:
                                _s9["reader_class"] = _rr.classify(_s9.get("material"))
                    # DEDUP SUGGESTIONS vs BOOKED WALLS (2026-08-19). The density and raster-v13
                    # branches above call their readers with EMPTY ownership, so a suggestion that
                    # merely re-outlines a wall the engine ALREADY booked was surfaced as new —
                    # census rp0818_tier_overlap: 85.8% of the tier (v13 59pc / 37,479 SF) was a
                    # re-listing, and one click of Accept would DOUBLE-BOOK its SF. Apply the SAME
                    # booked-territory subtraction the clean-vector helper already does
                    # (_vector_page_suggestions / _bbox_cover > 0.3), here at the COMMON append
                    # point so all three branches are guarded uniformly. Removes ONLY suggest_only
                    # pieces — booked totals sum non-suggest_only pieces only (app.py:1922/1982),
                    # so booked SF is untouched by construction; a genuinely NEW wall is <30% covered
                    # by any booked box and survives. Idempotent for the clean-vector branch.
                    _pre_dd9 = len(_sugs)
                    _owned_dd9 = [bb for bb in (_piece_bbox(p) for p in polys
                                                if not p.get("suggest_only")) if bb]
                    if _owned_dd9 and _sugs:
                        _kept_dd9 = []
                        for _sdd9 in _sugs:
                            _bdd9 = _piece_bbox(_sdd9)
                            if _bdd9 is not None and any(_bbox_cover(_bdd9, _odd9) > 0.3
                                                         for _odd9 in _owned_dd9):
                                continue      # the engine already booked this territory
                            _kept_dd9.append(_sdd9)
                        _sugs = _kept_dd9
                    if _pre_dd9 - len(_sugs):
                        jlog(job, f"Page {pi+1}: dropped {_pre_dd9 - len(_sugs)} suggestion(s) that "
                                  f"re-outline already-booked walls (dedup vs booked)", "info")
                    jlog(job, f"Page {pi+1}: suggestion probe — dense={_dense9}, "
                              f"img={_imga / max(pw * ph, 1):.0%}, suggestions={len(_sugs)}", "info")
                    # tag + append OUTSIDE the branches — BOTH readers' suggestions ship
                    # (the indent bug had raster suggestions computed then discarded)
                    for _s9 in _sugs:
                        _s9["suggest_only"] = True
                        _s9["material"] = "AI suggestion (confirm to add)"
                        _s9["category"] = "AI suggestion (confirm to add)"
                        _s9["group"] = "AI suggestion (confirm to add)"
                        _s9["sf_exact"] = False
                    if _sugs:
                        polys = polys + _sugs
                        auto_flags_pre = f"🤖 {len(_sugs)} AI wall suggestion(s) on this page — dashed outlines; click Accept on the ones that are real walls. NOT counted until accepted."
                    else:
                        auto_flags_pre = None
                except Exception as _se9:
                    jlog(job, f"Page {pi+1}: suggestion probe FAILED — {type(_se9).__name__}: {_se9}", "warn")
                    auto_flags_pre = None
            else:
                auto_flags_pre = None
            # LEGEND-FIRST SECOND PASS (owner directive 2026-07-25: naming is the lane).
            # Late-reader pieces (v13/band/rc) exist only AFTER the first naming pass —
            # re-run callout/legend binding on pieces still lacking a drawing identity.
            if auto:
                try:
                    _anon9 = [p for p in polys if not p.get("named_by")
                              and not p.get("suggest_only")
                              and p.get("area_sf", 0) > 100]
                    if _anon9:
                        _n2 = callouts.name_regions(pdf_bytes, pi, _anon9, pw, ph)
                        if _n2:
                            jlog(job, f"Page {pi+1}: {_n2} late region(s) named on second legend pass", "ok")
                    # PER-JOB CODE MAP (gate v2 2026-07-25: 35/35 = 1.000 precision):
                    # the set's own text defines CODE -> MATERIAL; expand bare-code
                    # names so family binding uses the job's vocabulary.
                    import re as _rc9
                    if "_code_map9" not in job:
                        _cm = {}
                        try:
                            _cd = fitz.open(stream=pdf_bytes, filetype="pdf")
                            for _cpi in range(len(_cd)):
                                _t = (_cd[_cpi].get_text() or "").upper()
                                for _m in _rc9.finditer(r"([A-Z][A-Z\s/&]{6,40})\s*[-–:]\s*\(?([A-Z]{1,4}-?\d{1,2}[A-Z]?)\)?", _t):
                                    _cm.setdefault(_m.group(2), _m.group(1).strip())
                                for _m in _rc9.finditer(r"\b([A-Z]{1,4}-?\d{1,2}[A-Z]?)\s*[:=–-]\s*([A-Z][A-Z\s/&]{6,40})", _t):
                                    _cm.setdefault(_m.group(1), _m.group(2).strip())
                            _cd.close()
                        except Exception:
                            pass
                        job["_code_map9"] = _cm
                    _cm = job["_code_map9"]
                    if _cm:
                        for p in polys:
                            _mt = str(p.get("material") or "")
                            if p.get("named_by") or p.get("suggest_only"):
                                continue
                            for _c in _rc9.findall(r"\b([A-Z]{1,4}-?\d{1,2}[A-Z]?)\b", _mt.upper()):
                                if _c in _cm:
                                    p["material"] = p["category"] = p["group"] = f"{_cm[_c]} ({_c})"
                                    p["named_by"] = "codemap:" + _c
                                    break
                except Exception:
                    pass
            # SELECT-TO-SUM FIX #1 — tag each AUTO polygon with its job-level material group.
            # The readers emit a distinct name per course width (vector_hatch:625), per seam
            # (:622) and per rgb (:1208), so the canvas — which keys a click-group on the
            # per-page fill_color — shows ~43 cryptic chips per job (measured, 103 jobs).
            # canon() collapses those families and folds sheet prose/OCR noise into ONE
            # confirm bucket. It relabels only; area_sf is never touched.
            # MARKED pages are deliberately NOT grouped: those names are the estimator's own
            # markup labels, the marked path is exact by constitution, and canon() would merge
            # two same-family labels (the battery pins the four marked SFs exactly).
            if auto and _mc is not None:
                try:
                    # SCOPE GATE: `out_of_scope` rides beside `suggest_only` at every site
                    # that decides what gets BOOKED. Inside process() it is provably never
                    # set (only POST /set-scope writes it, and only to an existing takeoff),
                    # so these five guards are no-ops on the analyze path by construction —
                    # they exist so a future re-process of a scoped job cannot resurrect a
                    # parked group's SF into the totals.
                    _cmap = _mc.build_color_name_map([q for q in polys if not q.get("suggest_only")
                                                      and not q.get("out_of_scope")])
                    for q in polys:
                        _g9, _c9 = _mc.material_group(q.get("material") or q.get("category") or "",
                                                      q.get("fill_color"), _cmap)
                        q["material_group"] = _g9
                        q["material_class"] = _c9
                        # ⚠ LOAD-BEARING, not cosmetic. App.jsx:1056 builds a Draw-tab
                        # polygon's category as `p.material_type || p.category`, and
                        # App.jsx:1084/1118 attribute a drawn CUT-OUT to the zone matching
                        # that name (`zz.category===hostCat || zz.materialName===hostCat`),
                        # falling back to the BIGGEST zone on the page when nothing matches.
                        # Renaming zones to material groups while polygons still answered with
                        # their raw name would have sent every opening on a merged group to
                        # the wrong material — the page total is unchanged but the per-material
                        # split (which is what gets priced) would be wrong, silently. Emitting
                        # the group here makes both sides speak the same name.
                        q["material_type"] = _g9
                except Exception as _ge:
                    jlog(job, f"Page {pi+1}: material grouping skipped ({_ge})", "warn")
            # ⭐ EVERY PIECE GETS AN ID (2026-08-17). Only the MARKUP path stamped one
            # (~:800); auto-read pieces and suggestions came out of vector_hatch with no
            # `id` at all, and `/accept-suggestion` matches `str(p["id"]) == str(pieceId)`
            # — so with every id None, the frontend's `pieceId: z.id` serialised to
            # ABSENT and "None" == "None" accepted the FIRST suggestion on the page
            # whichever one she clicked. Additive (a new key on pieces that had none, the
            # same value on pieces that had one — the list is only ever appended to, never
            # reordered) and no SF, zone, total or export reads it.
            for _i9, _p9 in enumerate(polys):
                _p9["id"] = _i9
            job["polygons_by_page"][pi + 1] = polys
            job["dims_by_page"][pi + 1] = {"width": pw, "height": ph}
            # keep pages that have linear (trim/LF) measurements even with no area polygons,
            # AND keep a page the site-scale guard emptied: dropping it here would make the
            # exclusion SILENT, which is the exact defect app.py:756's page-total SANITY flag
            # has (it fires on all six blowup pages and has no consumer, so it never becomes a
            # code). An excluded page must arrive at the estimator carrying its own reason.
            if not polys and not lin_lf_items and not site_scale_flags and not review_flags:
                # DETECTION-GAP (review-only; additive; books/removes NOTHING). This page is
                # about to be dropped for carrying no booked wall, no suggestion and no linear
                # run. When it is an AUTO page (the engine's job to draw) that PASSED elevation
                # admission on a set where elevations ARE positively identified
                # (doc_any_elevation), that blank is the high-precision "the engine drew nothing
                # on an elevation" signal — she cannot confirm a wall she cannot see is missing.
                # Record a "draw here" gap so the frontend can jump her to the page and arm
                # add-by-corners. It only APPENDS to a side list nothing else reads: no zone, no
                # polygon, no total, no SF moves, and the page is still dropped exactly as before
                # (a page with any booked/suggested wall never reaches here — confidently-wrong
                # impossible). Requiring doc_any_elevation keeps it high-precision: on a scanned
                # set where every page is admitted by inversion, a blank cover/plan is NOT flagged.
                if (not doc_has_markup) and page_is_elev and doc_any_elevation:
                    try:
                        job.setdefault("detectionGaps", []).append(
                            {"page": pi + 1, "sheetRef": f"p{pi+1}", "detectionGap": True,
                             "note": "elevation page — no cladding detected; draw the walls"})
                        jlog(job, f"Page {pi+1}: admitted elevation, engine detected no cladding "
                                  f"— flagged as a draw-here gap for review (0 SF booked)", "warn")
                    except Exception:
                        pass
                continue          # a guard-abandoned page (review_flags) is KEPT below so its
                                  # needs-review flag reaches the estimator, exactly like the
                                  # site-scale exclusion — an excluded page must never be silent.
            bymat = defaultdict(lambda: {"sf": 0.0, "n": 0, "category": None, "classes": [],
                                         "openSF": 0.0, "openN": 0})
            for p in polys:
                if p.get("suggest_only") or p.get("out_of_scope"):
                    continue          # suggestions NEVER enter zones/totals until accepted;
                                      # parked (out-of-scope) groups leave them the same way
                key = p.get("material") or p.get("category") or "Unlabeled"
                bymat[key]["sf"] += p["area_sf"]; bymat[key]["n"] += 1
                bymat[key]["category"] = p.get("category")
                bymat[key]["classes"].append(p.get("reader_class"))
                _o9o, _n9o = _poly_openings(p)   # net-assist: aggregate the reader's opening breakdown
                bymat[key]["openSF"] += _o9o; bymat[key]["openN"] += _n9o
            zones = []
            src_txt = ({"vector": "drawing vectors (exact geometry)", "model": "AI model (confirm)"}.get(auto_engine, "AI texture (confirm)")) if auto else "markup"
            # FAMILY LAYER v1 (weekend lane #43, 26-020 probe): normalize each zone's
            # family from its name; junk names and non-BFS materials get FLAGGED,
            # anonymous masses get FLAGGED — never guessed (DoD: flag beats guess).
            def _fam9(m):
                u = (m or "").upper()
                import re as _rf
                if _rf.match(r"^[AS]-?\d{2,4}\b", u) or u.startswith("GROUP ") or len(u) <= 2:
                    return "junk-name"
                if any(k in u for k in ("BRICK", "MASON", "STONE", "EIFS", "STUCCO", "CONCRETE",
                                         "GLAZ", "GLASS", "CURTAIN", "STOREFRONT", "GWB", "ROOF MEMBRANE")):
                    return "non-bfs"
                if "SOFFIT" in u: return "soffit"
                if "ACM" in u or "MCM" in u or "COMPOSITE" in u or "ALUCOBOND" in u: return "acm"
                if "NICHIHA" in u or "NCH" in u: return "nichiha"
                if "PERF" in u: return "perforated"
                # archive vocabulary (128 markup files / 775 terms, 2026-07-25):
                # his code system is prefix-stable across GCs
                import re as _rv
                if _rv.match(r"^(FCP|FCL|FCS|FCBB|FSV|FSH|FC)[-\s:]", u): return "fc"
                if _rv.match(r"^(VCS|EL0?\d)[-\s:]", u) or "VINYL" in u: return "vinyl"
                if _rv.match(r"^AP[-\s:]", u): return "metal"
                if any(k in u for k in ("SHINGLE", "SHAKE", "CEDAR")): return "shingle"
                if any(k in u for k in ("STANDING", "SEAM", "CORRUGAT", "METAL PANEL", "MTL", "ALUM")): return "metal"
                if any(k in u for k in ("FC", "FIBER", "HARDIE", "LAP", "SIDING", "CEMENT", "CLAPBOARD", "PLANK")): return "fc"
                return "unassigned"
            def _fam9_outer(m):
                # READER-generic vocabulary is not a drawing identity — always
                # unassigned (miner 2026-07-25: 'Panel wall'/'Vertical rib'/'Rendered
                # siding' token-leaked into fc and caused 7 of the 8 family errors)
                u = (m or "").upper()
                if any(k in u for k in ("WALL AREA", "(CONFIRM)", "WALL BAND", "PANEL WALL",
                                         "VERTICAL RIB", "RENDERED SIDING", "HATCHED AREA",
                                         "LAP / HORIZONTAL", "COLOR FILL", "AI SUGGESTION")):
                    return "unassigned"
                return _fam9(m)
            # ---- SELECT-TO-SUM FIX #1: the CLICK-GROUP rollup (auto only) -------------------
            # `bymat` above stays keyed on the RAW reader names and keeps driving the family /
            # scope FLAGS below, byte-identically — a display label must never become the input
            # to a scope warning. Zones are emitted per MATERIAL GROUP instead, and each group
            # takes its category, family and reader provenance from its DOMINANT raw name, so
            # every derived field still comes from a real drawing name.
            _grouped = bool(auto and _mc is not None and
                            any(p.get("material_group") for p in polys
                                if not p.get("suggest_only") and not p.get("out_of_scope")))
            bygrp = defaultdict(lambda: {"sf": 0.0, "n": 0, "cls": None, "classes": [],
                                         "openSF": 0.0, "openN": 0})
            _rawsf9 = defaultdict(float)
            if _grouped:
                for p in polys:
                    if p.get("suggest_only") or p.get("out_of_scope"):
                        continue
                    g9 = p.get("material_group") or "Unlabeled"
                    bygrp[g9]["sf"] += p["area_sf"]; bygrp[g9]["n"] += 1
                    bygrp[g9]["cls"] = p.get("material_class")
                    # the reader classes that BUILT this group — the group's own name is a
                    # display label by now, so the risk can only come from the pieces
                    bygrp[g9]["classes"].append(p.get("reader_class"))
                    _o9o, _n9o = _poly_openings(p)   # net-assist: same opening breakdown, per group
                    bygrp[g9]["openSF"] += _o9o; bygrp[g9]["openN"] += _n9o
                    _rawsf9[(g9, p.get("material") or p.get("category") or "Unlabeled")] += p["area_sf"]
            _dom9 = {}
            for (g9, raw9), sf9 in _rawsf9.items():
                if sf9 > _dom9.get(g9, (None, -1.0))[1]:
                    _dom9[g9] = (raw9, sf9)

            _fam_flags9 = set()
            for mat, d in bymat.items():
                # FLAGS ALWAYS RUN OVER THE RAW NAMES — grouped or not — so the scope warnings
                # the estimator relies on are unchanged by a display relabel.
                cat = d["category"] or "Other"
                _f9 = _fam9_outer(mat) if auto else None
                if _f9 == "non-bfs" and d["sf"] > 30:
                    _fam_flags9.add(f"⚠ NON-BFS MATERIAL detected: '{str(mat)[:36]}' ({d['sf']:,.0f} SF) — "
                                    "brick/masonry/EIFS/glazing class is never BFS scope; confirm exclusion.")
                if _f9 == "unassigned" and d["sf"] > 300:
                    _fam_flags9.add(f"⚠ FAMILY UNASSIGNED: '{str(mat)[:36]}' ({d['sf']:,.0f} SF) has no material "
                                    "identity from the drawing — name it before pricing (never priced on a guess).")
                if _grouped:
                    continue          # this page's zones are emitted per material group below
                zones.append(dict({
                    "materialName": mat, "material_type": mat, "category": cat,
                    "family": _f9,
                    "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
                    "totalOpeningArea": 0, "description": f"{d['n']} region(s) from {src_txt}",
                }, **_net_assist(d), **_review_fields(d["classes"], auto=auto, raw_name=mat)))
                legend[mat] = {"id": mat, "name": mat, "category": cat}
            for g9, d in bygrp.items():
                raw9 = _dom9.get(g9, (g9, 0.0))[0]        # dominant raw name carries provenance
                cat = (bymat.get(raw9) or {}).get("category") or "Other"
                zones.append(dict({
                    "materialName": g9, "material_type": g9, "category": cat,
                    "family": _fam9_outer(raw9),
                    "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
                    "totalOpeningArea": 0, "description": f"{d['n']} region(s) from {src_txt}",
                    "material_group": g9, "material_class": d["cls"],
                    "readAs": (str(raw9)[:60] if str(raw9) != str(g9) else None),
                }, **_net_assist(d), **_review_fields(d["classes"], auto=auto, raw_name=raw9)))
                legend[g9] = {"id": g9, "name": g9, "category": cat}
            auto_flags = []
            if auto_flags_pre:
                auto_flags.append(auto_flags_pre)
            page_levels = []
            if auto:
                # SOFFIT/RETURN SENTINEL — the historical money-loser was FORGETTING these
                # (they hide in section views where area readers don't reach). If the sheet
                # mentions them, the takeoff refuses to let them be forgotten.
                try:
                    _t = (pg.get_text() or "").upper()
                    _hits = [k for k in ("SOFFIT", "CANOPY", "RETURN") if k in _t]
                    if _hits:
                        auto_flags.append("⚠ " + "/".join(h.title() + "s" for h in _hits) +
                                          " on this job — often drawn in SECTION views the auto-read"
                                          " doesn't measure yet. Verify their SF before bidding"
                                          " (missed soffits/returns have cost real money).")
                    # REWORK-CASCADE sentinel (93 Bennington loss: 'REMOVE AND REFABRICATE
                    # panels that INTERFACE with the shifted wall' — scope written in notes,
                    # never measured). Rework verbs = scope beyond the highlighted limits.
                    # REWORK TRIGGER CORRECTED 2026-08-05 (FIX_TYPOF.md — instructional-text
                    # audit).  The trigger was `"REFABRICATE" in _t`, a BARE SUBSTRING test,
                    # and the ordinary drawing word PRE-FABRICATED contains it.  Re-running
                    # the shipped trigger over all 5,959 pages of the 121-job archive:
                    # 157 pages fire, only 27 carry a genuine rework verb, 125 fire ONLY on
                    # "PREFABRICATED" and 5 ONLY on "LIMITS OF WORK" -> 17.2% precision, and
                    # 43 of the 56 flagged jobs contain no rework verb anywhere.  Telling an
                    # estimator a prefab-panel job is a REWORK job is a confidently-wrong
                    # assertion.  \bREFABRICAT cannot match PREFABRICATED (the preceding P is
                    # a word char, so there is no boundary).  "LIMITS OF WORK" is dropped
                    # outright: it is the phrase that MARKS the limits, so firing "scope
                    # extends beyond the marked limits" on it asserts the opposite of the
                    # note.  The flag now QUOTES the matched note so the claim is checkable
                    # against the sheet.  FLAG TEXT ONLY: auto_flags reaches only
                    # "flags" at app.py:847 — no SF, no scale, no polygon.
                    _rw_m = re.search(r"\bREFABRICATE|\bREMOVE AND REINSTALL\b|\bREWORK", _t)
                    if _rw_m:
                        auto_flags.append(f"⚠ REWORK job — this sheet carries the note"
                                          f" '{_rw_m.group(0)}'. Rework scope lives in the NOTES,"
                                          " outside the highlighted limits (interfacing panels,"
                                          " returns, make-good), so it is easy to miss."
                                          " Read every note on this sheet before pricing.")
                except Exception:
                    pass
                auto_flags.append("Read from the drawing's pattern vectors — confirm which walls are in your scope"
                                  if auto_engine == "vector" else "AI suggestion — verify SF before bidding")
                # JOB-LEVEL SANITY (audit item 5 — flags only, SF never touched):
                # a merged/leaked region announces itself as a size outlier.
                try:
                    _sfs = sorted(p.get("area_sf", 0) for p in polys
                                  if not p.get("suggest_only") and not p.get("out_of_scope")
                                  and p.get("area_sf", 0) > 0)
                    if len(_sfs) >= 4:
                        _med = _sfs[len(_sfs) // 2]
                        _big = _sfs[-1]
                        if _med > 0 and _big > 3 * _med and _big > 1500:
                            auto_flags.append(f"⚠ SANITY: largest piece ({_big:,.0f} SF) is "
                                              f"{_big/_med:.0f}× the page median — check for a merged or "
                                              f"leaked region before trusting its SF.")
                    _ptot = sum(_sfs)
                    if _ptot > 40000:
                        auto_flags.append(f"⚠ SANITY: page total {_ptot:,.0f} SF is unusually large for "
                                          f"one sheet — verify the scale and check for double-counted faces.")
                except Exception:
                    pass
                if not scale_conf:
                    # scale could NOT be read → SF used a default 8.0 and is unreliable. Force a calibrate.
                    auto_flags.append("Scale could not be read on this sheet — calibrate before trusting SF (SF may be off several ×)")
                else:
                    auto_flags.append(f"Scale read as 1\"={scale_val}' — verify")
                # PER-VIEW SCALES (owner rule: never measure at a scale the drawing didn't
                # state): pieces re-measured at their view's own printed scale say so; a
                # piece in an AMBIGUOUS multi-scale zone is flagged, never silently priced.
                try:
                    n_vs = sum(1 for p in polys if p.get("view_scale"))
                    if n_vs:
                        vss = sorted({p["view_scale"] for p in polys if p.get("view_scale")})
                        auto_flags.append(f"✓ {n_vs} region(s) measured at their view's own printed scale "
                                          + ", ".join(f"1\"={v}'" for v in vss))
                    if any(p.get("scale_risk") for p in polys):
                        auto_flags.append("⚠ Multiple view scales printed near some regions — verify those "
                                          "SF against the view's own scale before pricing")
                except Exception:
                    pass
                # ELEVATION MARKERS (blueprint 1a): "T.O. STEEL 139'-0\"" texts give exact heights.
                # When ≥2 markers agree they yield the sheet's TRUE vertical scale — the strongest
                # possible check. Conservative: only speaks when the fit has real consensus.
                try:
                    lv = callouts.read_levels_pg(pg)
                    if len(lv) < 2 and len((pg.get_text() or "")) < 250 and ocr_text.available():
                        lv = ocr_text.read_levels(pdf_bytes, pi)
                    if lv:
                        page_levels = [{"ft": l["ft"], "label": l["label"]} for l in lv[:8]]
                        fit = callouts.vertical_scale_from_levels(lv)
                        if fit:
                            m_ftpi = fit[0] * 72.0
                            used_ftpi = float(scale_val) if (scale_conf and scale_val) else 8.0
                            agree = min(m_ftpi, used_ftpi) / max(m_ftpi, used_ftpi)
                            if agree >= 0.93:
                                auto_flags.append(f"✓ Elevation markers confirm the scale ({m_ftpi:.1f} ft/in from {fit[2]} markers)")
                                jlog(job, f"Page {pi+1}: elevation markers CONFIRM scale ({m_ftpi:.1f} ft/in)", "ok")
                            else:
                                scale_conf = False   # trips the existing calibrate-before-trusting safety
                                auto_flags.append(f"⚠ Elevation markers imply {m_ftpi:.1f} ft/in but {used_ftpi:.1f} was used — calibrate before trusting SF")
                                jlog(job, f"Page {pi+1}: markers imply {m_ftpi:.1f} ft/in vs {used_ftpi:.1f} used — flagged", "warn")
                except Exception:
                    pass
                # DIMENSION-STRING SELF-CALIBRATION (blueprint 1b): the drawing's own "24'-0\""
                # strings paired with their measured dim lines = the sheet's true scale.
                # Cross-check only (same safety pattern as markers): confirm, or force calibrate.
                try:
                    ds = dim_scale.sheet_scale(pg)
                    if ds:
                        d_ftpi = ds[0] * 72.0
                        used_ftpi = float(scale_val) if (scale_conf and scale_val) else 8.0
                        agree = min(d_ftpi, used_ftpi) / max(d_ftpi, used_ftpi)
                        if agree >= 0.95:
                            auto_flags.append(f"✓ {ds[1]} dimension strings confirm the scale ({d_ftpi:.2f} ft/in)")
                            jlog(job, f"Page {pi+1}: {ds[1]} dimension strings CONFIRM scale ({d_ftpi:.2f} ft/in)", "ok")
                        else:
                            scale_conf = False   # trips the existing calibrate-before-trusting safety
                            auto_flags.append(f"⚠ {ds[1]} dimension strings imply {d_ftpi:.2f} ft/in but {used_ftpi:.1f} was used — calibrate before trusting SF")
                            jlog(job, f"Page {pi+1}: dims imply {d_ftpi:.2f} ft/in vs {used_ftpi:.1f} used — flagged", "warn")
                except Exception:
                    pass
            # aggregate the estimator's LINEAR (LF) measurements — trim/soffit/fascia (priced per LF)
            lin_by_mat = defaultdict(float)
            for it in lin_lf_items:
                lin_by_mat[it["material"]] += it["lf"]
            linear_items = [{"material": m, "lf": round(v, 1)} for m, v in sorted(lin_by_mat.items(), key=lambda x: -x[1])]
            n_lin_sf = len(lin_sf_polys)
            # AUTO-TRIM (blueprint 1c): derive corner/base-top/opening LF straight from the
            # detected faces. SUGGESTIONS only — kept in a separate field so they never touch
            # the estimator's confirmed linearItems or the money total. Needs a trusted scale.
            auto_trim = []
            if auto and scale_conf and scale_val:
                try:
                    auto_trim = auto_trim_mod.compute(polys, pw, ph, float(scale_val) / 72.0)
                    if auto_trim:
                        auto_flags.append("Suggested trim (auto) below — verify each line against your scope before pricing")
                except Exception as _te:
                    jlog(job, f"Page {pi+1}: auto-trim skipped ({_te})", "warn")
            job["takeoffData"].append({
                "pageNumber": pi + 1, "title": f"Sheet page {pi+1}", "sheetRef": f"p{pi+1}",
                "scale": (f"1\"={scale_val}'" if (auto and scale_conf) else "auto (calibrate)") if auto else "from markup",
                "scaleSource": ("default" if (auto and not scale_conf) else ("AI auto — verify" if auto else "estimator markup")),
                "verifiedScale": bool(auto and scale_conf),
                "building": "Building",
                "zones": zones,
                "linearItems": linear_items,
                "autoTrim": auto_trim,
                "flags": auto_flags + sf_warns + sorted(_fam_flags9) + site_scale_flags + review_flags,
                "levels": page_levels,
                "source": "texture-auto" if auto else "digitize",
                # window/door COUNT surface (count-only takeoffs are a whole bid class):
                # openings the readers detected and cut out of the SF on this page
                "openingsCount": (sum(len(p.get("holes") or []) for p in polys) if auto else None),
            })
            # SECOND READER (confidence signal; two-tier -- moves no booked SF): an area
            # read DIRECTLY from this page's printed W/H dimension strings
            # (dim_area.area -> W_ft*H_ft), attached ALONGSIDE the vector SF so their
            # agreement/disagreement is measurable per page. It never touches a polygon,
            # an area_sf, a netArea or a grossArea; it abstains (None) unless both axes
            # close a dimension chain. See dim_area.py + SECOND_READER_TRAIN_LEG.md.
            try:
                _dra = dim_area.area(pg) if auto else None
            except Exception:
                _dra = None
            job["takeoffData"][-1]["areaFromDims"] = (round(_dra["area_sf"], 1) if (_dra and _dra.get("area_sf")) else None)
            job["takeoffData"][-1]["areaFromDimsBasis"] = (_dra.get("basis") if _dra else None)
            extra = (f" + {n_lin_sf} linear run(s)" if n_lin_sf else "") + (f", {len(linear_items)} trim/LF item(s)" if linear_items else "")
            jlog(job, f"Page {pi+1}: " + (f"{len(polys)} AI-suggested zone(s)" if auto else f"{len(polys)} marked region(s){extra}") + f", {len(zones)} material(s)", "warn" if auto else "ok")
            for w in sf_warns:
                jlog(job, f"Page {pi+1}: ⚠ {w}", "warn")
        doc.close()
        # CROSS-PAGE SCALE-OUTLIER review flag (additive; touches no SF): a page read at a
        # confidently-wrong scale won't trip the default-scale warning, yet inflates SF by
        # scale-squared. Flag any confirmed page whose scale is an outlier vs this set's shared
        # scale so she can verify/calibrate before trusting its SF. Never books or removes SF.
        try:
            _flag_scale_outliers(job["takeoffData"], jlog=jlog, job=job)
        except Exception:
            jlog(job, "Cross-page scale-outlier check skipped — takeoff unaffected", "warn")
        job["legend"] = list(legend.values())
        # THE ARCHITECT'S OWN MATERIAL SCHEDULE — read it off any text-bearing page so the
        # estimator can sanity-check our takeoff against the drawing's stated quantities.
        try:
            sched_best = {}
            sdoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pi in range(sdoc.page_count):
                if len(sdoc[pi].get_text() or "") < 60:   # skip textless pages fast
                    continue
                for row in callouts.read_schedule_pg(sdoc[pi]):
                    k = row["key"]
                    if k not in sched_best or row["sf"] > sched_best[k]["sf"]:
                        sched_best[k] = row
            sdoc.close()
            if len(sched_best) >= 2:
                sched = sorted(sched_best.values(), key=lambda r: -r["sf"])
                job["drawingSchedule"] = {"items": sched, "total": sum(r["sf"] for r in sched)}
                jlog(job, f"Read the drawing's own material schedule — {len(sched)} materials, "
                          f"{job['drawingSchedule']['total']:,} SF stated", "ok")
        except Exception:
            pass
        # OCR FALLBACK for FLATTENED sets — auto-detected pages with cladding but NO base text
        # (lettering exported as curves). Reads the material spec off the rendered image so the
        # estimator knows WHAT the cladding is even when the file carries no text. Lazy + capped.
        try:
            odoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            flat_pages = [e["pageNumber"] for e in job["takeoffData"]
                          if e.get("source") == "texture-auto"
                          and len((odoc[e["pageNumber"] - 1].get_text() or "")) < 250][:4]
            odoc.close()
            if flat_pages and not ocr_text.available():
                jlog(job, f"OCR unavailable on server ({len(flat_pages)} flattened page(s) would benefit)", "warn")
            if flat_pages and ocr_text.available():
                seen = set(); ocr_mats = []
                for pn in flat_pages:
                    for m in ocr_text.read_materials(pdf_bytes, pn - 1):
                        k = m["text"].upper()
                        if k not in seen:
                            seen.add(k); ocr_mats.append(m)
                if ocr_mats:
                    job["ocrMaterials"] = ocr_mats[:10]
                    jlog(job, f"Read {len(ocr_mats)} material spec(s) off the drawing with OCR "
                              f"(flattened set): {', '.join(m['text'] for m in ocr_mats[:3])}", "ok")
        except Exception:
            pass
        # ANNOT-INTEGRITY GUARD (assisted-path robustness, 2026-07-23): corrupt or
        # partially-stripped PDFs can carry annot chains that BREAK mid-iteration —
        # the reader then silently reads SOME markups and produces a wrong total.
        # Compare the xref-declared annot count per page against what iteration
        # reaches; any shortfall = LOUD flag (SF untouched — the human re-exports).
        try:
            _idoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            _bad_pages = []
            for _pi8 in range(len(_idoc)):
                _pg8 = _idoc[_pi8]
                _decl = 0
                try:
                    _av8 = _idoc.xref_get_key(_pg8.xref, "Annots")
                    if _av8 and _av8[0] in ("array", "xref"):
                        import re as _re8
                        for _x8 in _re8.findall(r"(\d+) 0 R", str(_av8[1])):
                            try:
                                _o8 = _idoc.xref_object(int(_x8), compressed=True) or ""
                                if not _o8.strip() or _o8.strip() == "null":
                                    _decl += 1   # freed/null object behind a live ref
                                    continue
                            except Exception:
                                _decl += 1   # unresolvable ref = exactly what we're hunting
                                continue
                            # Popup/Link/Widget live in /Annots but are NOT takeoff
                            # markups and are rightly skipped by annot iteration
                            if any(t in _o8 for t in ("/Subtype/Popup", "/Subtype /Popup",
                                                       "/Subtype/Link", "/Subtype /Link",
                                                       "/Subtype/Widget", "/Subtype /Widget")):
                                continue
                            _decl += 1
                except Exception:
                    pass
                if not _decl:
                    continue
                _seen = 0
                try:
                    _a8 = _pg8.first_annot
                    while _a8 is not None:
                        _seen += 1
                        _a8 = _a8.next
                except Exception:
                    pass          # chain broke mid-iteration — _seen holds the shortfall
                if _seen < _decl:
                    _bad_pages.append((_pi8 + 1, _seen, _decl))
            _idoc.close()
            if _bad_pages and job["takeoffData"]:
                _pp = ", ".join(f"p{p} ({s}/{d})" for p, s, d in _bad_pages[:5])
                job["takeoffData"][-1].setdefault("flags", []).append(
                    f"⚠ MARKUP INTEGRITY: this PDF's annotations are partially unreadable on "
                    f"{len(_bad_pages)} page(s) — read/declared: {_pp}. The totals may be missing "
                    "markups. Re-export the PDF from Bluebeam (no flatten) and re-upload.")
        except Exception:
            pass
        # TYPICAL-OF-n FLAG.  PREMISE CORRECTED 2026-08-05 (FIX_TYPOF.md; first
        # measured in FIX_CONVENTIONS.md, then INDEPENDENTLY RE-DERIVED).
        # The 2026-07-23 census that motivated the old text was a co-occurrence table;
        # the within-job control collapses TYP to -1.9 found points = zero effect.
        # typof_verify.py then re-scanned ALL pages of all 121 jobs with THIS regex on
        # THIS text source, and compared the stated n against k = the instances the
        # estimator actually traced, under four independent k definitions:
        #   26-214 p51 'TYP OF 4' k=10 | 26-021 p12 'TYP. OF (3)' k=4
        #   26-214 p70 'TYP OF 4' k=3  | 26-109 p1  'TYPICAL OF 5' k=1
        # k == n on 0 of 4, and k spans 1..10 against n=3..5 in BOTH directions, so
        # no count rule is fundable.  The repeated units ARE drawn and he traces each
        # one: on p51 thirteen garage-screen panels are traced (ten within 1% of
        # 77.92 SF) on a sheet whose note says "TYP OF 4".  So "one instance is drawn
        # but n exist" was false and "multiply" would have over-billed that page 4x.
        # A TYP-OF callout means repeats EXIST but the drawing does not establish how
        # many are in scope -> it is a FLAG TO CHECK, never a multiplier.
        # 42 pages / 21 of 121 jobs fire this; only 4 carry gold and 33 carry no
        # markup at all, so the archive is silent (not supportive) on the rest.
        # FLAG ONLY: SF is never auto-multiplied.
        try:
            import re as _re7
            _tdoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for _pi7 in range(len(_tdoc)):
                _t7 = (_tdoc[_pi7].get_text() or "").upper()
                for _m7 in _re7.finditer(r"TYP(?:ICAL)?\.?\s*(?:OF|X)\s*\(?(\d+)\)?", _t7):
                    _n7 = int(_m7.group(1))
                    if 2 <= _n7 <= 30:
                        for _e7 in job["takeoffData"]:
                            if _e7["pageNumber"] == _pi7 + 1:
                                _e7.setdefault("flags", []).append(
                                    f"⚠ REPETITION: this sheet says '{_m7.group(0).strip()}'."
                                    " That means repeats exist — but the drawing does NOT establish"
                                    " how many are in your scope, so this is a flag to CHECK, never a"
                                    f" multiplier. Do NOT multiply the measured area by {_n7}: in the"
                                    " 121-job archive the repeated units are each DRAWN and the traced"
                                    " count never equalled the stated n (a sheet marked 'TYP OF 4'"
                                    " carries 10 traced instances of one 78 SF unit). Confirm every"
                                    " drawn instance is in your takeoff before bidding.")
                                break
                        break   # one flag per page
                if "MATCH LINE" in _t7:
                    for _e7 in job["takeoffData"]:
                        if _e7["pageNumber"] == _pi7 + 1:
                            _e7.setdefault("flags", []).append(
                                "⚠ MATCH LINE on this sheet — the view continues on another sheet."
                                " Confirm the continued portion is in the takeoff. Archive census:"
                                " 69 match-line pages in 19 jobs, and on 32 of them NO sheet"
                                " reference is printed beside the match line, so which sheet"
                                " continues it cannot be resolved from the drawing.")
                            break
            _tdoc.close()
        except Exception:
            pass
        # JOB-LEVEL SOFFIT/RETURN CROSS-CHECK (the $500K recall guard — BOTH paths):
        # if ANY page of the set mentions soffits (or panel/canopy returns) but the
        # finished takeoff contains no zone named for them, flag it loudly. Flags
        # only — SF is never touched.
        try:
            _sr_pages = []
            _sdoc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
            for _pi2 in range(len(_sdoc2)):
                _t2 = (_sdoc2[_pi2].get_text() or "").upper()
                if "SOFFIT" in _t2 or ("RETURN" in _t2 and ("PANEL" in _t2 or "CANOPY" in _t2 or "ACM" in _t2)):
                    _sr_pages.append(_pi2 + 1)
            _sdoc2.close()
            if _sr_pages and job["takeoffData"]:
                _named = any(("soffit" in str(z.get("materialName") or z.get("category") or "").lower())
                             or ("return" in str(z.get("materialName") or z.get("category") or "").lower())
                             for e in job["takeoffData"] for z in e["zones"])
                if not _named:
                    job["takeoffData"][-1].setdefault("flags", []).append(
                        "⚠ JOB CHECK: the drawings mention soffits/returns on page(s) "
                        + ", ".join(map(str, _sr_pages[:6])) + ("…" if len(_sr_pages) > 6 else "")
                        + " but NO soffit/return zone exists in this takeoff — confirm they are"
                          " measured or excluded before bidding (missed soffits have cost real money).")
        except Exception:
            pass
        # ── COVERAGE LAYER: compose flags + reconcile — report only, SF never touched
        try:
            import coverage as _cov
            _cov.run(job, pdf_bytes, jlog=lambda m, lv="info": jlog(job, m, lv))
        except Exception:
            jlog(job, "Coverage layer error — takeoff unaffected", "warn")
        # ── GROSS-vs-NET CONVENTION FLAG (Lane 6, 2026-08-05): the engine measures the
        # wall the architect drew (GROSS); the archive books NET. Two opening detectors
        # were refuted against pre-registered bars (FIX_OPENINGS.md), so the product
        # NAMES the convention instead of guessing a deduction. Metadata + flag text
        # only — the module fingerprints every SF number before/after and self-reverts
        # if one moved.
        try:
            import openings_convention as _oc
            _oc.annotate(job, jlog=lambda m, lv="info": jlog(job, m, lv))
        except Exception:
            jlog(job, "Gross-vs-net convention flag skipped — takeoff unaffected", "warn")
        total = sum(z["netArea"] for e in job["takeoffData"] for z in e["zones"])
        if auto_tried:  # never fail silently: say what the AI actually looked at
            jlog(job, f"Auto-detect scanned {auto_tried} page(s), found cladding on {auto_hits}", "warn" if auto_hits == 0 else "ok")
        if not job["takeoffData"]:
            jlog(job, "No measurements found — no readable markup, and the AI saw no cladding on the pages it scanned.", "warn")
        else:
            jlog(job, f"Done — {len(job['takeoffData'])} page(s), {round(total):,} SF total", "success")
        job["status"] = "done"; job["phase"] = "done"
        job["progress"] = {"label": "Complete", "pct": 100}
        _persist_job(jid)  # durable: this takeoff now survives a restart/redeploy
    except Exception as e:
        import traceback; traceback.print_exc()
        job["status"] = "error"; job["phase"] = "error"; job["error"] = str(e)
        jlog(job, f"Error: {e}", "error")
        _persist_job(jid)

# ── endpoints ──────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    jid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"  # collision-proof (no same-ms overwrite)
    # STREAM the upload to disk in chunks — never hold a whole giant set in RAM.
    # (await pdf.read() on a 1GB file was the OOM vector for heavy documents.)
    d = _job_dir(jid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.pdf")
    with open(path, "wb") as fh:
        while True:
            chunk = await pdf.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    # MEMORY GUARD (2026-08-18): reject an oversized set BEFORE it is pulled into RAM. The
    # stream-to-disk above bounds the UPLOAD, but _load_pdf_bytes reads the whole file back
    # into memory and process() then renders every page — a 442MB/396pp set (26-002) drove
    # the single worker into an OOM restart that killed every concurrent job. BIG_PDF_BYTES
    # was defined at module load but NEVER checked (that was the bug); enforce it here with a
    # clear, polled-through error instead of an OOM. The job is created in the error state so
    # the existing GET /status contract surfaces the reason; nothing is queued to process().
    _sz = os.path.getsize(path)
    if _sz > BIG_PDF_BYTES:
        _msg = ("This PDF is %.0f MB, over the %.0f MB limit for automatic processing. Split "
                "it (e.g. just the architectural elevation sheets) and re-upload, or raise "
                "BIG_PDF_BYTES if the server has the memory." % (_sz / 1e6, BIG_PDF_BYTES / 1e6))
        jobs[jid] = {"status": "error", "phase": "error", "log": [], "error": _msg,
                     "progress": {"label": "File too large", "pct": 0},
                     "legend": [], "takeoffData": [], "scheduleData": None,
                     "polygons_by_page": {}, "dims_by_page": {}, "pdf": None}
        jlog(jobs[jid], _msg, "error")
        try:
            os.remove(path)          # free the oversized file immediately; nothing will read it
        except Exception:
            pass
        _persist_job(jid); _gc_disk()
        return {"jobId": jid}
    data = _load_pdf_bytes(path)
    jobs[jid] = {"status": "queued", "phase": "idle", "log": [], "progress": {"label": "Queued", "pct": 0},
                 "legend": [], "takeoffData": [], "scheduleData": None, "error": None,
                 "polygons_by_page": {}, "dims_by_page": {}, "pdf": data}
    _persist_job(jid); _evict_mem(); _gc_disk()  # durable + bounded RAM + bounded disk
    _enqueue_job(process, jid, data)  # bounded FIFO queue (was background_tasks) — never OOM the worker
    return {"jobId": jid}

@app.get("/status/{jid}")
def status(jid: str):
    j = get_job(jid)
    if not j: raise HTTPException(404, "job not found")
    return {"status": j["status"], "phase": j.get("phase", ""), "log": j["log"], "progress": j["progress"],
            "legend": j.get("legend", []), "takeoffData": j.get("takeoffData", []),
            "scheduleData": j.get("scheduleData"), "drawingSchedule": j.get("drawingSchedule"),
            "ocrMaterials": j.get("ocrMaterials"), "error": j.get("error"), "pageCount": j.get("pageCount", 0),
            "coverageReport": j.get("coverageReport"),
            # grouped GROSS-vs-NET convention summary (counts + archive deduction RANGE).
            # Estimator information only — every SF in it is a convention estimate and is
            # never added to or subtracted from any total (openings_convention.py).
            "openingsConvention": j.get("openingsConvention"),
            # her review walk + her confirmed NETs, echoed so a reload/second device
            # resumes where she left off instead of starting the hour again. Both are
            # written ONLY by their own POST handlers; `reviewConfirmed` is read by
            # nothing that computes SF, and `netOverrides` is already baked into the
            # `netArea` above by `_apply_nets` (this is the audit trail, not the source).
            "reviewConfirmed": j.get("reviewConfirmed") or {},
            "netOverrides": j.get("netOverrides") or {},
            "materialNameMap": j.get("materialNameMap") or {}}

@app.get("/polygons/{jid}/{page}")
def polygons(jid: str, page: int):
    j = get_job(jid)
    if not j: raise HTTPException(404, "job not found")
    dims = j.get("dims_by_page", {}).get(page) or {"width": 612, "height": 792}
    return {"polygons": j.get("polygons_by_page", {}).get(page, []),
            "width": dims["width"], "height": dims["height"]}

@app.get("/page-image/{jid}/{page}")
def page_image(jid: str, page: int):
    j = get_job(jid)
    if not j or not j.get("pdf"): raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2))  # ~144 dpi
    png = pix.tobytes("png"); doc.close()
    return Response(content=png, media_type="image/png")

@app.get("/page-crop/{jid}/{page}")
def page_crop(jid: str, page: int, x0: float = 0, y0: float = 0, x1: float = 1, y1: float = 1, px: int = 1800):
    """Bluebeam-style deep zoom: render ONLY the viewport rect (normalized coords) at high DPI.
    The frontend swaps this in when the estimator zooms past ~2.5x so textures stay crisp."""
    j = get_job(jid)
    if not j or not j.get("pdf"): raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pg = doc[page - 1]
    x0 = max(0.0, min(1.0, x0)); y0 = max(0.0, min(1.0, y0))
    x1 = max(x0 + 0.01, min(1.0, x1)); y1 = max(y0 + 0.01, min(1.0, y1))
    # pg.rect is already the rotated/display rect and clip takes display coords (verified on a 270° page)
    W, H = pg.rect.width, pg.rect.height
    clip = fitz.Rect(x0 * W, y0 * H, x1 * W, y1 * H)
    z = min(8.0, max(1.0, min(2200, px) / max(1, clip.width)))
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip, alpha=False)
    png = pix.tobytes("png"); doc.close()
    return Response(content=png, media_type="image/png")

@app.post("/material-groups")
def material_groups_route(payload: dict = Body(...)):
    """A PREVIEW of material groups (within-job texture clustering, elevation-fenced). The estimator
    SELECTS the groups she's bidding — her selection carries the accuracy; the preview may be imperfect.
    SF returned is a PREVIEW estimate (approx), clearly flagged. Additive; digitize-markup untouched.
    payload: {jobId, page (1-indexed)}."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    page = int(payload.get("page", 1)) - 1
    # VECTOR-FIRST (estimator-confirmed: the geometry path beats texture clustering by a mile).
    # Each vector region is emitted as grid patches so the existing select + exact-on-select
    # flow works unchanged; the texture clustering stays as the fallback for scanned pages.
    try:
        vpolys, _, _, _ = vector_hatch.detect(j["pdf"], page)
        if vpolys:
            import numpy as np
            import cv2 as _cv2
            GX, GY = 96, 64
            groups = []
            for gi, p in enumerate(vpolys[:24]):
                pts = np.array([[x * GX, y * GY] for x, y in p["points"]], np.float32).astype(np.int32)
                m = np.zeros((GY, GX), np.uint8)
                _cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
                ys, xs = np.where(m > 0)
                patches = [[round(x / GX, 4), round(y / GY, 4), round(1 / GX, 4), round(1 / GY, 4)]
                           for x, y in zip(xs.tolist(), ys.tolist())][:1500]
                if not patches:
                    continue
                groups.append({"group": f"vec{gi}", "color": p.get("fill_color") or [0, 0.7, 0.85],
                               "patches": patches, "approx_sf": p.get("area_sf", 0),
                               "material": p.get("material"), "source": "vector"})
            if groups:
                return {"status": "ok", "groups": groups, "engine": "vector"}
    except Exception:
        pass
    try:
        r = material_groups.groups(j["pdf"], page)
    except Exception as e:
        return {"groups": [], "error": str(e)}
    return r

@app.post("/refine-group")
def refine_group_route(payload: dict = Body(...)):
    """Exact-on-select: preview-group patches → corner-snapped exact shapes (+opening deductions).
    payload: {jobId, page (1-indexed), patches:[[nx,ny,nw,nh],...], min_opening_sf}. Additive."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    try:
        return snap_fill.refine_group(j["pdf"], int(payload.get("page", 1)) - 1, payload.get("patches") or [],
                                      min_opening_sf=float(payload.get("min_opening_sf") or 0))
    except Exception as e:
        return {"status": "error", "shapes": [], "error": str(e)}

@app.post("/snap-fill")
def snap_fill_route(payload: dict = Body(...)):
    """Coloring-book bucket / corner-snap → exact polygon + SF from the drawing's vector geometry.
    payload: {jobId, page (1-indexed), point:[nx,ny]}  OR  {jobId, page, corners:[[nx,ny],...]}.
    Bucket returns status 'ok' (points+area_sf) or 'leak' (caller switches to corner mode). Additive —
    does not touch the digitize-markup pipeline."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    page = int(payload.get("page", 1)) - 1
    dims = j.get("dims_by_page", {}).get(page + 1) or {"width": 612, "height": 792}
    try:
        if payload.get("corners"):
            r = snap_fill.corners(j["pdf"], page, payload["corners"],
                                  min_opening_sf=float(payload.get("min_opening_sf") or 0))
        else:
            r = snap_fill.bucket(j["pdf"], page, payload.get("point", [0.5, 0.5]))
    except Exception as e:
        return {"status": "error", "error": str(e)}
    r["width"] = dims["width"]; r["height"] = dims["height"]
    return r

@app.post("/split-shape")
def split_shape(payload: dict = Body(...)):
    """ESTIMATOR'S KNIFE: split a face along the structural line nearest the click — where
    HIS material boundary lives (the drawing often can't show it; he knows it). Each half
    keeps its share of the (already net) SF by area ratio; holes follow by containment.
    payload: {jobId, page (1-based), points:[[nx,ny]..], area_sf, holes, click:[nx,ny]}"""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    try:
        from shapely.geometry import Polygon as SPoly, LineString, Point as SPoint
        from shapely.ops import split as sh_split
        page = int(payload.get("page", 1)) - 1
        doc = fitz.open(stream=j["pdf"], filetype="pdf")
        pg = doc[page]
        W, H = pg.rect.width, pg.rect.height
        geo = vector_hatch.page_geometry(pg)
        doc.close()
        pts = [(float(x) * W, float(y) * H) for x, y in payload.get("points", [])]
        poly = SPoly(pts).buffer(0)
        if poly.is_empty:
            return {"status": "error", "error": "bad polygon"}
        cx, cy = float(payload["click"][0]) * W, float(payload["click"][1]) * H
        minx, miny, maxx, maxy = poly.bounds
        # candidate structural lines near the click: long V (split left/right) or H (split bands)
        best = None
        for (x1, y1, x2, y2) in geo.get("segs", []):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            L = (dx * dx + dy * dy) ** 0.5
            if L < 25:
                continue
            if dx < 0.05 * L:                      # vertical line: x ~ const
                x = (x1 + x2) / 2
                if minx + 4 < x < maxx - 4 and L >= 0.35 * (maxy - miny):
                    d = abs(x - cx)
                    if best is None or d < best[0]:
                        best = (d, "v", x)
            elif dy < 0.05 * L:                    # horizontal line: y ~ const
                y = (y1 + y2) / 2
                if miny + 4 < y < maxy - 4 and L >= 0.35 * (maxx - minx):
                    d = abs(y - cy)
                    if best is None or d < best[0]:
                        best = (d, "h", y)
        if best is None or best[0] > 0.06 * max(W, H):
            # no structural line near the click — split exactly AT the click (still useful)
            best = (0, "v", cx) if (maxy - miny) > (maxx - minx) else (0, "h", cy)
        _, axis, c = best
        cutter = LineString([(c, miny - 10), (c, maxy + 10)]) if axis == "v" else LineString([(minx - 10, c), (maxx + 10, c)])
        parts = [g for g in sh_split(poly, cutter).geoms if g.area > 1e-6]
        if len(parts) < 2:
            return {"status": "error", "error": "line does not split the face"}
        parts.sort(key=lambda g: -g.area)
        parts = parts[:2] if len(parts) == 2 else [parts[0], max(parts[1:], key=lambda g: g.area)]
        total_a = sum(g.area for g in parts)
        sf = float(payload.get("area_sf", 0))
        holes = payload.get("holes") or []
        out = []
        for g in parts:
            ext = list(g.exterior.coords)
            hp = []
            for hh in holes:
                try:
                    hc = SPoly([(float(x) * W, float(y) * H) for x, y in hh]).centroid
                    if g.contains(hc):
                        hp.append(hh)
                except Exception:
                    pass
            out.append({"points": [[round(x / W, 5), round(y / H, 5)] for x, y in ext[:-1]],
                        "area_sf": round(sf * g.area / max(total_a, 1e-9), 1),
                        "holes": hp})
        return {"status": "ok", "shapes": out, "axis": axis,
                "cut_at": round((c / W) if axis == "v" else (c / H), 5)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/learn")
def learn(payload: dict = Body(...)):
    """Capture a manual/corrected takeoff as labeled training data (the flywheel).
    payload: {jobId, page, shapes:[{points,name,color,type}], source}"""
    jid = payload.get("jobId")
    job = get_job(jid)
    try:
        # disk-fill guard: a single correction payload is KBs; anything huge is abuse
        if len(json.dumps(payload)) > 40_000_000:
            return {"ok": False, "error": "payload too large"}
        os.makedirs(CORR_DIR, exist_ok=True)
        ts = str(int(time.time() * 1000))
        d = os.path.join(CORR_DIR, ts)
        os.makedirs(d, exist_ok=True)
        if job and job.get("pdf") and payload.get("shapes"):   # pdf copy only for shape-labeled corrections
            with open(os.path.join(d, "drawing.pdf"), "wb") as fh:
                fh.write(job["pdf"])
        with open(os.path.join(d, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump({**payload, "at": ts}, fh)               # keep the WHOLE payload (final-* answer keys carry takeoffData)
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))])
        return {"ok": True, "saved": ts, "total": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/recall")
def recall():
    """Shared hatch learning -- the READ half of the flywheel (the write half is /learn).
    The frontend calls this on every job load (App.jsx) to pre-identify repeat hatches, so a
    fresh browser or a second estimator inherits what others already labeled. Reads back the
    hatch signatures /learn persisted to CORR_DIR and returns {hatches:{sig:{category,
    materialName, materialId}}} -- exactly the shape App.jsx merges into its learned store.
    READ-ONLY: opens no job, books/removes no SF, and cannot be confidently-wrong (CW=0)."""
    hatches = {}
    try:
        if os.path.isdir(CORR_DIR):
            for d in sorted(os.listdir(CORR_DIR)):   # oldest -> newest so the latest label wins
                lp = os.path.join(CORR_DIR, d, "labels.json")
                if not os.path.isfile(lp):
                    continue
                try:
                    with open(lp, encoding="utf-8") as fh:
                        rec = json.load(fh)
                except Exception:
                    continue
                for h in (rec.get("hatches") or []):
                    sig = h.get("signature")
                    if not sig:
                        continue
                    hatches[sig] = {"category": h.get("category"),
                                    "materialName": h.get("materialName") or h.get("materialId"),
                                    "materialId": h.get("materialId")}
    except Exception:
        pass
    return {"hatches": hatches}

@app.get("/autonomy-status")
def autonomy_status():
    """THE AUTONOMY METER: for every exported bid (answer key = final-* capture), compare the
    system's FIRST AUTO OUTPUT (job.json as processed, before any human edit) against the
    human-confirmed final. agreement = min/max of total SF. This is the dial that shows the
    learning working — when it lives near 100%, estimators stop reviewing and only price."""
    jobs_seen = {}
    try:
        for d in sorted(os.listdir(CORR_DIR)):
            lp = os.path.join(CORR_DIR, d, "labels.json")
            if not os.path.isfile(lp):
                continue
            try:
                with open(lp, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if not str(rec.get("source", "")).startswith("final"):
                continue
            jid = rec.get("jobId")
            if jid:
                jobs_seen[jid] = rec        # latest final per job wins
    except Exception:
        pass
    out = []
    for jid, rec in jobs_seen.items():
        j = get_job(jid)
        if not j:
            continue
        auto_sf = sum(z.get("netArea", 0) for e in (j.get("takeoffData") or []) for z in (e.get("zones") or []))
        fin_sf = sum(z.get("netArea", 0) for e in (rec.get("takeoffData") or []) for z in (e.get("zones") or []))
        if fin_sf <= 0:
            continue
        agree = round(100 * min(auto_sf, fin_sf) / max(auto_sf, fin_sf, 1))
        out.append({"jobId": jid, "projName": rec.get("projName") or j.get("projName") or "Project",
                    "auto_sf": round(auto_sf), "final_sf": round(fin_sf), "agreement": agree})
    out.sort(key=lambda r: r["jobId"], reverse=True)
    avg = round(sum(r["agreement"] for r in out) / len(out)) if out else None
    return {"jobs": out[:40], "avg_agreement": avg, "n": len(out)}

def process_compare(jid, data):
    """THE KILLER DEMO: his marked takeoff vs the AI's blank-sheet takeoff on the SAME
    drawing. Reads his Bluebeam polygons EXACTLY, strips them to a synthetic clean set,
    runs the auto engine on it, grades every wall with the bench's assembled scoring
    (pieces >=50%-inside the wall, summed SF + union coverage) — the frozen-exam
    machinery, live in production, on the estimator's own job."""
    j = get_job(jid)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        his_by_pg = {}
        for pi in range(min(doc.page_count, 8)):
            pg = doc[pi]
            pw, ph = pg.rect.width, pg.rect.height
            his = [p for p in extract_page_polygons(pg, pw, ph, 8.0)
                   if p.get("sf_exact") and 40 <= p["area_sf"] <= 60000]
            if his:
                his_by_pg[pi] = his
        if not his_by_pg:
            j["status"] = "error"
            j["error"] = "No labeled takeoff polygons found — upload the estimator's marked Bluebeam set."
            _persist_job(jid); return
        for pi in range(doc.page_count):
            pg = doc[pi]
            for _ in range(400):
                a = pg.first_annot
                if not a:
                    break
                try:
                    pg.delete_annot(a)
                except Exception:
                    break
        clean = doc.tobytes()
        doc.close()
        j["pdf"] = clean                    # /page-image now serves the CLEAN drawing
        import vector_hatch
        from shapely.geometry import Polygon as _CP
        from shapely.ops import unary_union as _cu
        pages = []
        for pi, his in sorted(his_by_pg.items()):
            j["progress"] = {"label": f"AI takeoff on page {pi + 1}…", "pct": 20 + int(70 * len(pages) / max(1, len(his_by_pg)))}
            _persist_job(jid)
            try:
                pieces, W, H, sinfo = vector_hatch.detect(clean, pi)
            except Exception:
                pieces, sinfo = [], {}
            pp = []
            for p in pieces:
                try:
                    q = _CP([(x, y) for x, y in p["points"]]).buffer(0)
                    if not q.is_empty:
                        pp.append((q, p))
                except Exception:
                    pass
            walls = []
            for g in his:
                try:
                    gq = _CP([(x, y) for x, y in g["points"]]).buffer(0)
                except Exception:
                    continue
                mine = []
                for q, p in pp:
                    try:
                        if q.intersection(gq).area >= 0.5 * q.area:
                            mine.append((q, p))
                    except Exception:
                        pass
                asf = sum(p.get("area_sf", 0) for _, p in mine)
                cov = 0.0
                try:
                    if mine:
                        cov = _cu([q for q, _ in mine]).intersection(gq).area / max(1e-9, gq.area)
                except Exception:
                    pass
                walls.append({"sf": g["area_sf"], "mat": g.get("material") or "Wall",
                              "got": round(asf, 1), "cov": round(cov, 2),
                              "money": bool(cov >= 0.7 and abs(asf - g["area_sf"]) <= 0.15 * g["area_sf"])})
            pages.append({
                "page": pi + 1,
                "his": [{"points": g["points"], "sf": g["area_sf"],
                         "mat": (g.get("material") or "")[:40]} for g in his],
                "ours": [{"points": p["points"], "sf": p.get("area_sf", 0),
                          "mat": (p.get("material") or "")[:40],
                          "holes": (p.get("holes") or [])[:12]} for p in pieces],
                "hisTotal": round(sum(g["area_sf"] for g in his), 1),
                "ourTotal": round(sum(p.get("area_sf", 0) for p in pieces), 1),
                "walls": walls,
                "scale_confirmed": bool(sinfo.get("scale_confirmed")),
            })
        nw = sum(len(p["walls"]) for p in pages)
        nm = sum(1 for p in pages for w in p["walls"] if w["money"])
        nf = sum(1 for p in pages for w in p["walls"] if w["cov"] >= 0.3)
        j["compare"] = {"pages": pages,
                        "summary": {"walls": nw, "found": nf, "money": nm,
                                    "hisTotal": round(sum(p["hisTotal"] for p in pages), 1),
                                    "ourTotal": round(sum(p["ourTotal"] for p in pages), 1)}}
        j["status"] = "done"
        j["progress"] = {"label": "Comparison ready", "pct": 100}
        _persist_job(jid)
    except Exception as e:
        try:
            j["status"] = "error"; j["error"] = f"compare failed: {e}"
            _persist_job(jid)
        except Exception:
            pass

@app.post("/compare")
async def compare(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    jid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    d = _job_dir(jid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.pdf")
    with open(path, "wb") as fh:
        while True:
            chunk = await pdf.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    data = _load_pdf_bytes(path)
    jobs[jid] = {"status": "processing", "phase": "compare", "log": [],
                 "progress": {"label": "Reading the estimator's takeoff…", "pct": 5},
                 "legend": [], "takeoffData": [], "scheduleData": None, "error": None,
                 "polygons_by_page": {}, "dims_by_page": {}, "pdf": data}
    _persist_job(jid); _evict_mem(); _gc_disk()
    _enqueue_job(process_compare, jid, data)  # bounded FIFO queue (was background_tasks)
    return {"jobId": jid}

@app.get("/compare-result/{jid}")
def compare_result(jid: str):
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return {"status": j["status"], "progress": j.get("progress"),
            "error": j.get("error"), "compare": j.get("compare")}

@app.get("/learn-status")
def learn_status():
    try:
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))]) if os.path.isdir(CORR_DIR) else 0
    except Exception:
        n = 0
    return {"corrections": n}

@app.get("/evidence-pdf/{jid}")
def evidence_pdf(jid: str, materials: str = ""):
    """Marked-up evidence PDF: cover summary + each page with the regions OUTLINED in their color and
    labeled with SF. Optional ?materials=A,B filters to only what the estimator selected."""
    j = get_job(jid)
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    sel = set(m.strip().lower() for m in materials.split(",") if m.strip()) if materials else None
    def _match(name, cat):
        return (str(name or "").lower() in sel) or (str(cat or "").lower() in sel)
    # safety: if the selection matches nothing (e.g. renamed groups), mark everything — never a blank PDF
    if sel is not None and not any(_match(_disp_name(z), z.get("category"))
                                   for el in j.get("takeoffData", []) for z in el.get("zones", [])):
        sel = None
    def keep(name, cat):
        return True if sel is None else _match(name, cat)
    src = fitz.open(stream=j["pdf"], filetype="pdf")
    out = fitz.open()
    # summary cover page (filtered to her selection when given)
    # SPEC 3 (a): the column SUMS netArea (the booked number — reader-detected openings deducted,
    # her net overrides applied), so label it "SF (net)" and show gross alongside so the receipt
    # is unambiguous once overrides apply. gross = the zone's true gross (net-assist grossSF), or
    # netArea when the reader computed no breakdown.
    mats = {}; tot = 0.0; tgross = 0.0
    for el in j.get("takeoffData", []):
        for z in el.get("zones", []):
            if not keep(_disp_name(z), z.get("category")):
                continue
            k = _disp_name(z) or "Material"
            _n = z.get("netArea", 0) or 0
            _g = z.get("grossSF", None)
            _g = float(_g) if _g is not None else _n
            m = mats.setdefault(k, {"net": 0.0, "gross": 0.0})
            m["net"] += _n; m["gross"] += _g
            tot += _n; tgross += _g
    cov = out.new_page(width=612, height=792)
    cov.insert_text((50, 60), "Boston Facade Systems — Takeoff Evidence", fontsize=17, color=(0.05, 0.11, 0.18))
    cov.insert_text((50, 84), (j.get("projName") or "Project"), fontsize=11, color=(0.4, 0.45, 0.5))
    y = 130
    cov.insert_text((50, y), "Material", fontsize=10, color=(0.4, 0.45, 0.5))
    cov.insert_text((360, y), "SF (net)", fontsize=10, color=(0.4, 0.45, 0.5))
    cov.insert_text((470, y), "SF (gross)", fontsize=10, color=(0.4, 0.45, 0.5)); y += 8
    cov.draw_line((50, y), (560, y), color=(0.8, 0.83, 0.87)); y += 20
    for k, mv in sorted(mats.items(), key=lambda x: -x[1]["net"]):
        cov.insert_text((50, y), str(k)[:44], fontsize=11)
        cov.insert_text((360, y), f"{mv['net']:,.0f}", fontsize=11)
        cov.insert_text((470, y), f"{mv['gross']:,.0f}", fontsize=11, color=(0.4, 0.45, 0.5)); y += 22
    y += 6; cov.draw_line((50, y), (560, y), color=(0.8, 0.83, 0.87)); y += 22
    cov.insert_text((50, y), "TOTAL (net)", fontsize=12, color=(0.05, 0.11, 0.18))
    cov.insert_text((360, y), f"{tot:,.0f} SF", fontsize=12, color=(0.05, 0.11, 0.18))
    cov.insert_text((470, y), f"{tgross:,.0f}", fontsize=11, color=(0.4, 0.45, 0.5))
    # BID-READINESS block: provenance + verify-first — the questions a GC exec asks.
    y += 34
    cov.insert_text((50, y), "How these numbers were measured", fontsize=11, color=(0.05, 0.11, 0.18)); y += 16
    cov.insert_text((50, y), "Every region is read from the drawing's own geometry (seams, fills, tags,", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 12
    cov.insert_text((50, y), "trained boundary model). SF (net) is the booked number: openings the reader", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 12
    cov.insert_text((50, y), "detected are deducted and your net overrides applied. SF (gross) is the wall", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 12
    cov.insert_text((50, y), "as drawn, before any opening deduction.", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 18
    nflag = 0
    flagged = []
    for el in j.get("takeoffData", []):
        for fl in (el.get("flags") or []):
            nflag += 1
            if len(flagged) < 6 and not str(fl).startswith("✓"):
                flagged.append((el.get("pageNumber"), str(fl)[:86]))
    unver = [el.get("pageNumber") for el in j.get("takeoffData", [])
             if el.get("scaleSource") == "default"]
    cov.insert_text((50, y), "Verify first:", fontsize=10, color=(0.7, 0.35, 0.05)); y += 14
    if unver:
        cov.insert_text((58, y), f"- Pages {', '.join(str(p) for p in unver[:8])}: scale unconfirmed - calibrate before trusting SF", fontsize=8.5, color=(0.55, 0.3, 0.05)); y += 12
    for pn9, fl in flagged:
        cov.insert_text((58, y), f"- p{pn9}: {fl}", fontsize=8.5, color=(0.55, 0.3, 0.05)); y += 12
    if not unver and not flagged:
        cov.insert_text((58, y), "- No open flags. All scales confirmed.", fontsize=8.5, color=(0.15, 0.5, 0.25)); y += 12
    # PER-WALL TABLE page(s): every zone with its page, SF, and reader provenance —
    # the wall-by-wall receipt a reviewer can check against the drawing in seconds.
    def _reader_of(z):
        # prefer the provenance the zone carries (grouped zones are named for the material,
        # not the reader, so the name prefixes are no longer available here)
        return z.get("reader") or _reader_from_name(z.get("materialName"), z.get("source"))
    rows = []
    for el in j.get("takeoffData", []):
        for z in el.get("zones", []):
            if keep(_disp_name(z), z.get("category")):
                rows.append((el.get("pageNumber"), str(_disp_name(z) or "Material")[:38],
                             z.get("netArea", 0), _reader_of(z)))
    rows.sort(key=lambda r: -r[2])
    ty = 792
    tp = None
    for i9, (pn9, mat9, sf9, rd9) in enumerate(rows[:120]):
        if ty > 720:
            tp = out.new_page(width=612, height=792)
            tp.insert_text((50, 50), "Wall-by-wall detail", fontsize=13, color=(0.05, 0.11, 0.18))
            tp.insert_text((50, 72), "Page", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((90, 72), "Material / wall", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((360, 72), "SF (net)", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((430, 72), "Measured by", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.draw_line((50, 80), (562, 80), color=(0.8, 0.83, 0.87))
            ty = 96
        tp.insert_text((50, ty), f"p{pn9}", fontsize=9)
        tp.insert_text((90, ty), mat9, fontsize=9)
        tp.insert_text((360, ty), f"{sf9:,.0f}", fontsize=9)
        tp.insert_text((430, ty), rd9, fontsize=8.5, color=(0.35, 0.4, 0.45))
        ty += 15
    # each page: OUTLINE each kept region in its color + label the SF (a real marked-up sheet)
    for el in j.get("takeoffData", []):
        pn = el.get("pageNumber")
        if not pn or pn < 1 or pn > src.page_count:
            continue
        page_polys = [p for p in j.get("polygons_by_page", {}).get(pn, []) if keep(p.get("material"), p.get("category"))]
        if not page_polys:
            continue
        # Feature 4 (export cosmetic): a page whose kept regions are ALL suggestions or
        # parked (out_of_scope) draws nothing in the per-region loop below (it skips exactly
        # those), so appending it here would emit a blank background sheet. Skip it. Render-
        # only: no zone, total, cover or CSV moves -- the cover already excludes these regions.
        if not any(not (p.get("suggest_only") or p.get("out_of_scope")) for p in page_polys):
            continue
        # RENDER the page to a size-capped JPEG background (not the huge vector page) → small, emailable.
        # The region outlines + SF labels are drawn as crisp VECTOR on top, so they stay sharp.
        srcpage = src[pn - 1]; pw, ph = srcpage.rect.width, srcpage.rect.height
        zoom = min(2.5, 2400.0 / max(pw, ph, 1))  # cap long side ~2400px regardless of sheet size
        pix = srcpage.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pg = out.new_page(width=pw, height=ph)
        pg.insert_image(pg.rect, stream=pix.tobytes("jpg", jpg_quality=80))  # JPEG → colored sheets shrink hugely
        # SPEC 3 (b): the per-region labels below are the SF measured per region (as drawn); the
        # cover TOTAL is the booked net (net overrides + reader openings applied), so the sum of
        # sheet labels need not equal the cover once nets are applied. Say so on the sheet.
        pg.insert_text((8, 14), "Region labels = SF measured per region. Cover TOTAL = booked net.",
                       fontsize=7, color=(0.5, 0.0, 0.0))
        for p in page_polys:
            if p.get("suggest_only") or p.get("out_of_scope"):
                continue      # unaccepted AI suggestions never appear on the evidence PDF,
                              # and neither does a group she parked as out of scope (its
                              # zones already left `zones`, so the cover/detail tables above
                              # drop it too — the receipt shows the bid, not the building)
            col = p.get("fill_color") or [0.85, 0.1, 0.1]
            col = tuple(float(c) for c in col[:3])
            pts = [(float(x) * pw, float(y) * ph) for x, y in (p.get("points") or [])]
            if len(pts) >= 3:
                pg.draw_polyline(pts + [pts[0]], color=col, width=1.3)
            cx, cy = p.get("cx", 0.5) * pw, p.get("cy", 0.5) * ph
            pg.insert_text((cx - 18, cy), f"{p.get('area_sf', 0):,.0f} SF", fontsize=8, color=(0.55, 0.0, 0.0))
    data = out.tobytes(); out.close(); src.close()
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="BFS_Evidence_{jid}.pdf"'})

@app.get("/takeoff-csv/{jid}")
def takeoff_csv(jid: str):
    """SPEC 3 (c): the server-authoritative canonical takeoff — one file, straight from job state
    (post-scope, post-net, post-name). PURE READ: it books nothing and moves no total. Parked
    (out_of_scope) and suggest_only zones already left `zones`, so this emits exactly the bid set
    the evidence-PDF cover and bid-excel read. Keyed on material_group (`_scope_group_of_zone`),
    the same identity scope / net / materials use, so a second consumer or a re-export reconciles
    against the review total without re-assembling per_page on the client."""
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    import csv as _csv, io as _io
    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["page", "sheetRef", "material_group", "name", "category",
                "grossSF", "netSF", "openingSF", "openingsDetected", "netOverride", "reader"])
    tot_net = 0.0; tot_gross = 0.0
    for el in j.get("takeoffData", []):
        pn = el.get("pageNumber"); sref = el.get("sheetRef") or ""
        for z in el.get("zones", []):
            net = float(z.get("netArea", 0) or 0)
            _g = z.get("grossSF", None)
            gross = float(_g) if _g is not None else net
            osf = float(z.get("openingSF", 0) or 0)
            w.writerow([pn, sref, _scope_group_of_zone(z), _disp_name(z) or "",
                        z.get("category") or "", round(gross, 1), round(net, 1), round(osf, 1),
                        int(z.get("openingsSuggested") or 0),
                        "yes" if z.get("netOverride") else "no",
                        z.get("reader") or z.get("readerClass") or ""])
            tot_net += net; tot_gross += gross
    w.writerow([])
    w.writerow(["TOTAL", "", "", "", "", round(tot_gross, 1), round(tot_net, 1), "", "", "", ""])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="BFS_Takeoff_{jid}.csv"'})

@app.post("/accept-suggestion")
def accept_suggestion(payload: dict = Body(...)):
    """Flip a suggest_only v13 piece into a REAL zone (the human said it's a wall).
    payload: {jobId, page (1-based), pieceId}. Rebuilds that page's zone rows,
    persists, logs the acceptance as flywheel training data."""
    jid = payload.get("jobId"); page = int(payload.get("page") or 0)
    pid = payload.get("pieceId")
    j = get_job(jid)
    if not j or page < 1:
        raise HTTPException(404, "job/page not found")
    polys = (j.get("polygons_by_page") or {}).get(page) or (j.get("polygons_by_page") or {}).get(str(page))
    if polys is None:
        raise HTTPException(404, "no polygons for page")
    hit = next((p for p in polys if str(p.get("id")) == str(pid) and p.get("suggest_only")), None)
    if hit is None and pid is not None:
        # ID FALLBACK (2026-08-17): jobs analysed BEFORE pieces were stamped with an id
        # (~:1030) are still on the volume and still rehydrate, and every piece in them
        # answers `id: None`. Without this, her click on suggestion #4 of an old job
        # 404s; with the old matcher it accepted #1. The array index IS the id this
        # analyse path now assigns, so an index lookup is the same answer.
        try:
            _cand = polys[int(pid)]
            if _cand.get("suggest_only") and _cand.get("id") is None:
                hit = _cand
        except Exception:
            hit = None
    if hit is None:
        raise HTTPException(404, "suggestion not found (already accepted?)")
    _accepted_cls = hit.get("reader_class")
    hit["suggest_only"] = False
    hit["material"] = "AI wall (accepted)"
    hit["category"] = "AI wall (accepted)"
    hit["group"] = "AI wall (accepted)"
    # rebuild the page's zones from non-suggestion polys (mirror the analyze path).
    # ⚠ THIS MIRROR IS LOAD-BEARING: it overwrites the page's zones wholesale, so if it keyed
    # on the raw name while /analyze keyed on the material group, accepting one suggestion
    # would silently shatter that page back into the ~43 unnamed chips FIX #1 just removed —
    # and only on the pages the estimator interacted with. Keyed the same way, from the same
    # tag the analyze path already wrote onto each polygon.
    from collections import defaultdict as _dd
    # ⚠ SCOPE GATE + THE MIRROR: this handler REPLACES the page's zones wholesale, so a
    # parked group would be resurrected into the totals by the first suggestion she
    # accepts on that page — visible only on the pages she touched, which is the exact
    # failure mode this block's own header warns about. Parked pieces are excluded here
    # for the same reason suggestions are; their zones stay in `parkedZones` untouched.
    _live9 = [p for p in polys if not p.get("suggest_only") and not p.get("out_of_scope")]
    _grouped9 = bool(_mc is not None and any(p.get("material_group") for p in _live9))
    if _grouped9 and _mc is not None:
        try:                      # the accepted region is new — it has no group tag yet
            _cmap9 = _mc.build_color_name_map(_live9)
            for p in _live9:
                if not p.get("material_group"):
                    _g9, _c9 = _mc.material_group(p.get("material") or p.get("category") or "",
                                                  p.get("fill_color"), _cmap9)
                    p["material_group"] = _g9; p["material_class"] = _c9
        except Exception:
            _grouped9 = False
    bymat = _dd(lambda: {"sf": 0.0, "n": 0, "category": None, "raw": None, "cls": None,
                         "classes": []})
    for p in _live9:
        key = ((p.get("material_group") or "Unlabeled") if _grouped9
               else (p.get("material") or p.get("category") or "Unlabeled"))
        bymat[key]["sf"] += p.get("area_sf", 0); bymat[key]["n"] += 1
        bymat[key]["category"] = p.get("category")
        bymat[key]["raw"] = p.get("material") or p.get("category") or "Unlabeled"
        bymat[key]["cls"] = p.get("material_class")
        # ⚠ THE MIRROR AGAIN: this handler REPLACES the page's zones wholesale, so a field
        # the analyze path writes and this one forgets vanishes from exactly the pages the
        # estimator touched — the failure mode this block's own header warns about. The
        # review fields are rebuilt here from the same stamps, including the ACCEPTED
        # piece's own (stamped at suggestion time, before its name became "AI wall
        # (accepted)"): a click confirms the wall EXISTS, it does not re-measure its area,
        # so the region keeps the risk of the reader that drew it.
        bymat[key]["classes"].append(p.get("reader_class"))
    zones = [dict({"materialName": m, "material_type": m, "category": d["category"] or "Other",
                   "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
                   "totalOpeningArea": 0, "description": f"{d['n']} region(s)",
                   "material_group": (m if _grouped9 else None), "material_class": d["cls"]},
                  **_review_fields(d["classes"], auto=True, raw_name=d["raw"]))
             for m, d in bymat.items()]
    for e in j.get("takeoffData") or []:
        if e.get("pageNumber") == page:
            e["zones"] = zones
    # ⚠ THE MIRROR, ONE MORE TIME. This handler REBUILDS the page's zones, so a zone she
    # had already netted comes back wearing the machine's gross. `netOverrides` is keyed
    # on (page, material_group stamp) exactly so it survives that — re-apply it here or
    # accepting a suggestion silently reverts her confirmed number on that page.
    try:
        # ⚠ THE MIRROR + PAGE SCOPE: the rebuild above put this page's zones back into
        # `zones`, so a page (or group) she PARKED would be resurrected into the totals by
        # the first suggestion she accepts on it — the exact mirror-bug class this handler's
        # header warns of. Re-park before netting/naming so page-park survives the rebuild.
        _apply_scope(j, j.get("scopeMap") or {}, j.get("pageScopeMap"))
        _apply_nets(j)
        _apply_material_names(j)   # keep her name across the page rebuild
    except Exception:
        pass
    try:
        _persist_job(jid)
    except Exception:
        pass
    try:  # flywheel: an accepted suggestion is gold-grade boundary supervision
        ts = int(time.time() * 1000)
        os.makedirs(CORR_DIR, exist_ok=True)
        with open(os.path.join(CORR_DIR, f"{ts}_suggest-accept.json"), "w", encoding="utf-8") as fh:
            json.dump({"jobId": jid, "page": page, "source": "suggest-accept",
                       "shapes": [{"points": hit.get("points"), "area_sf": hit.get("area_sf")}],
                       "_ts": ts}, fh)
    except Exception:
        pass
    # `readerClass` rides back so the accept button can say WHICH reader drew the wall she
    # just booked, and the review queue can rank it, without a second round trip. It is the
    # stamp taken BEFORE the rename above — the piece is now called "AI wall (accepted)",
    # and re-deriving a class from that name would give every accepted wall the same wrong
    # risk. A click confirms the wall EXISTS; it does not re-measure it or re-classify it.
    return {"ok": True, "zones": zones, "accepted_sf": hit.get("area_sf"),
            "readerClass": _accepted_cls,
            "reader": (_rr.label(_accepted_cls) if _rr is not None else None),
            "reviewRisk": (_rr.risk(_accepted_cls) if _rr is not None else None)}

@app.get("/scope/{jid}")
def scope_get(jid: str):
    """THE SCOPE GATE, read side: every material group this job found, with its SF,
    the pages it lives on, and whether it is ticked into THIS bid.

    Pure read — it books nothing, changes nothing and cannot fail a takeoff. With no
    ticklist set every group answers `inScope: true`, which is the identity state.
    """
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return dict(_scope_summary(j), jobId=jid, scopeMap=(j.get("scopeMap") or {}),
                note=("Untick what another trade is bidding. Unticked groups leave the "
                      "totals, the Excel and the evidence PDF but stay on the drawing, "
                      "parked and one-click restorable — nothing is ever deleted."))


@app.post("/set-scope")
def set_scope(payload: dict = Body(...)):
    """THE SCOPE GATE, write side — the 30-second human input that replaces six dead
    autonomous levers (PERFECT_ACCURACY_BLUEPRINT STEP 1).

    payload: {jobId, groups:{name: bool}}         explicit ticks (the panel sends this)
             {jobId, outOfScope:[name,...]}       park these
             {jobId, inScope:[name,...]}          restore these
             {jobId, reset:true}                  clear the ticklist -> back to identity

    Persistence follows /accept-suggestion exactly: mutate the dict `get_job` handed
    back (which IS `jobs[jid]` — get_job rehydrates into the hot cache), then
    `_persist_job`. `scopeMap` is in that function's whitelist, without which the
    ticklist would evaporate on the next eviction.

    Returns the fresh takeoffData so the panel updates the job total in one round trip.
    """
    jid = payload.get("jobId")
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    smap = {} if payload.get("reset") else dict(j.get("scopeMap") or {})
    if isinstance(payload.get("groups"), dict):
        for k, v in payload["groups"].items():
            smap[str(k)] = bool(v)
    for k in (payload.get("outOfScope") or []):
        smap[str(k)] = False
    for k in (payload.get("inScope") or []):
        smap[str(k)] = True
    j["scopeMap"] = smap
    n_park, n_back = _apply_scope(j, smap, j.get("pageScopeMap"))
    try:
        _apply_nets(j)   # a restored group comes back with HER net, not the gross
        _apply_material_names(j)   # ... and with HER name
    except Exception:
        pass
    jobs[jid] = j
    try:
        _persist_job(jid)
    except Exception:
        pass
    return dict(_scope_summary(j), ok=True, jobId=jid, scopeMap=smap,
                zonesParked=n_park, zonesRestored=n_back,
                takeoffData=j.get("takeoffData") or [])


@app.get("/page-scope/{jid}")
def page_scope_get(jid: str):
    """THE PAGE-SCOPE GATE, read side: every takeoff page in this job with its detected
    SF, sheet ref, engine flags and whether it is ticked into THIS bid. The sibling of
    /scope — material-scope removes a family, page-scope removes a whole sheet (or the
    off-elevation residual of a family that also lives on her pages).

    Pure read — it books nothing, changes nothing and cannot fail a takeoff. With no
    page ticklist every page answers `inScope: true`, which is the identity state.
    """
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return dict(_page_scope_summary(j), jobId=jid, pageScopeMap=(j.get("pageScopeMap") or {}),
                note=("Untick a whole sheet another trade owns, or a civil / 3D / needs-review "
                      "page the engine flagged. Unticked pages leave the totals, the Excel and "
                      "the evidence PDF but stay on the drawing, parked and one-click restorable "
                      "— nothing is ever deleted. All pages are in by default."))


@app.post("/set-page-scope")
def set_page_scope(payload: dict = Body(...)):
    """THE PAGE-SCOPE GATE, write side — mirrors /set-scope exactly, but keyed on
    pageNumber instead of material group.

    payload: {jobId, pages:{"<pn>": bool}}        explicit ticks (the panel sends this)
             {jobId, outOfScope:[pn,...]}          park these pages
             {jobId, inScope:[pn,...]}             restore these pages
             {jobId, reset:true}                   clear the ticklist -> back to identity

    Persistence follows /set-scope: mutate the dict `get_job` handed back (which IS
    `jobs[jid]`), re-apply BOTH scope maps (so page-park composes with material-park),
    re-bake HER net + HER name onto the restored zones, then `_persist_job`.
    `pageScopeMap` is in that function's whitelist, without which the page ticklist
    would evaporate on the next eviction. Returns the fresh page summary + takeoffData
    so the panel updates the job total in one round trip.
    """
    jid = payload.get("jobId")
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    psmap = {} if payload.get("reset") else dict(j.get("pageScopeMap") or {})
    if isinstance(payload.get("pages"), dict):
        for k, v in payload["pages"].items():
            psmap[str(k)] = bool(v)
    for k in (payload.get("outOfScope") or []):
        psmap[str(k)] = False
    for k in (payload.get("inScope") or []):
        psmap[str(k)] = True
    j["pageScopeMap"] = psmap
    n_park, n_back = _apply_scope(j, j.get("scopeMap") or {}, psmap)
    try:
        _apply_nets(j)   # a restored page comes back with HER net, not the gross
        _apply_material_names(j)   # ... and with HER name
    except Exception:
        pass
    jobs[jid] = j
    try:
        _persist_job(jid)
    except Exception:
        pass
    return dict(_page_scope_summary(j), ok=True, jobId=jid, pageScopeMap=psmap,
                zonesParked=n_park, zonesRestored=n_back,
                takeoffData=j.get("takeoffData") or [])


@app.get("/materials/{jid}")
def materials_get(jid: str):
    """THE NAMING BRIDGE, read side: every material group with its SF, the pages it
    lives on, the name she has given it (if any) and an is_unnamed flag for the
    generic / no-identity groups the drawing gave no name to -- the '[name it]' rows.

    Pure read -- it books nothing and cannot fail a takeoff. With no materialNameMap
    every group answers with the machine's own label, the identity state."""
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return dict(_materials_summary(j), jobId=jid,
                materialNameMap=(j.get("materialNameMap") or {}),
                note=("Name the 'unnamed cladding' groups the drawing gave no label to -- "
                      "the engine measured the SF, you give it the product name. A name "
                      "moves the label on the page, the ledger and the exports; it moves not "
                      "one SF, and clearing it returns the job byte-identical."))


@app.post("/set-material-name")
def set_material_name(payload: dict = Body(...)):
    """THE NAMING BRIDGE, write side -- the 1-2 min/job flywheel entry.

    payload: {jobId, group, name}          name this material group
             {jobId, group, name:null}     clear the name (back to the machine's label)
             {jobId, reset:true}           clear every name on the job

    `group` is the material_group STAMP -- the same identity the scope gate and
    /set-net key on (see `_scope_group_of_zone`), never a renameable display string.
    The name lands ONLY on `materialNameHer`, a display field, so no scope, net or
    review KEY moves and not one SF changes. Identity by default: an empty map renames
    nothing and a cleared name removes every field this added, byte-identical to today.

    A confirmed pair also appends to a durable per-practice ledger (the flywheel),
    which is SHOWN on the next job by the same GC/architect, NEVER auto-applied.
    Returns fresh takeoffData + the materials summary so the panel updates in one trip.
    """
    jid = payload.get("jobId")
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    nm = {} if payload.get("reset") else dict(j.get("materialNameMap") or {})
    grp = None
    if not payload.get("reset"):
        grp = payload.get("group")
        if grp is None:
            raise HTTPException(400, "group required")
        grp = str(grp)
        known = {r["group"] for r in _materials_summary(j)["groups"]}
        if grp not in known:                 # never let an arbitrary key into the map
            raise HTTPException(404, "unknown material group: %s" % grp[:60])
        nm_name = payload.get("name")
        if nm_name is None or str(nm_name).strip() == "":
            nm.pop(grp, None)
        else:
            nm[grp] = str(nm_name).strip()[:80]
    j["materialNameMap"] = nm
    napplied = _apply_material_names(j)
    jobs[jid] = j
    try:
        _persist_job(jid)
    except Exception:
        pass
    if grp is not None and nm.get(grp):      # a confirmed pair feeds the flywheel
        try:
            _append_material_ledger(j, jid, grp, nm[grp])
        except Exception:
            pass
    return dict(_materials_summary(j), ok=True, jobId=jid, group=grp,
                materialNameMap=nm, namesApplied=napplied,
                takeoffData=j.get("takeoffData") or [])


@app.get("/review-queue/{jid}")
def review_queue(jid: str):
    """THE RANKING (FIX_TWO_TIER_GATE's replacement for the assert tier).

    Returns this job's regions ordered by the MEASURED confidently-wrong rate of the
    reader that drew each one — riskiest first — so the estimator spends her first clicks
    where the corpus says the engine is most often materially over.

    ⭐ IT BOOKS NOTHING. No SF changes, no region is asserted, no region is hidden, and
    every zone in `takeoffData` is still here: this endpoint re-ORDERS, it does not filter
    (`order_zones` is required to return every zone it was given). That is the whole
    reason this could ship when the two-tier ASSERT gate could not — a gate that books
    money on a 15.9%-wrong class breaks the zero-confidently-wrong promise, and an
    ordering cannot, whatever the risk numbers turn out to be.

    The risk table is a MEASUREMENT (review_rank.TABLE names the corpus and its md5), not
    a judgement about materials, and it is re-derived by
    `bfs_overnight\\probe_review_rank.py` / checked by `test_review_rank_stamp.py`.
    """
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    if _rr is None:                      # visible, never a silently unordered queue
        raise HTTPException(503, "review_rank unavailable: %s" % (_RR_ERR or "not imported"))
    polys_by = j.get("polygons_by_page") or {}

    def _page_polys(pn):
        return polys_by.get(pn) or polys_by.get(str(pn)) or []

    confirmed = j.get("reviewConfirmed") or {}
    rows = []
    for el in (j.get("takeoffData") or []):
        pn = el.get("pageNumber")
        _pp = _page_polys(pn)
        for z in (el.get("zones") or []):
            cls = z.get("readerClass")
            # ⭐ PER-PIECE ROWS — the 355k SF that looked like a blank page.
            # r4_absorbed_census: 1,598 of her 2,659 uncredited walls (355k SF) HAVE
            # ENGINE INK ON THEM. They were invisible here because a row is per
            # (page, material) and a rollup of nine regions is one row wearing one
            # SF — she could not see, let alone confirm, the individual walls the
            # engine had already drawn. This is the STEP-2 "credit what's drawn"
            # surface and it books NOTHING: the pieces are read straight off
            # `polygons_by_page` with the zone's own rollup key (the material_group
            # STAMP, the same identity the scope gate and _apply_nets key on, never
            # a renameable display string), suggestions and parked pieces excluded
            # exactly as the zone builder excludes them.
            _zkey = _scope_group_of_zone(z)
            pieces = []
            for _idx, p in enumerate(_pp):
                if p.get("suggest_only") or p.get("out_of_scope"):
                    continue
                if _scope_group_of_poly(p) != _zkey:
                    continue
                pieces.append({"id": (p.get("id") if p.get("id") is not None else _idx),
                               "sf": round(float(p.get("area_sf") or 0), 1),
                               "cx": p.get("cx"), "cy": p.get("cy"),
                               "readerClass": p.get("reader_class")})
            pieces.sort(key=lambda q: -(q["sf"] or 0))
            rows.append({"page": pn,
                         "materialName": z.get("materialName"),
                         "displayName": _disp_name(z),
                         "netArea": z.get("netArea", 0),
                         "zoneKey": _zkey,
                         "grossArea": z.get("grossArea"),
                         "netAreaAuto": z.get("netAreaAuto"),
                         "netOverride": bool(z.get("netOverride")),
                         "confirmed": bool(confirmed.get(_net_key(pn, z.get("materialName")))),
                         "pieces": pieces,
                         "pieceCount": len(pieces),
                         "readerClass": cls,
                         "reader": z.get("reader") or _rr.label(cls),
                         "reviewRank": z.get("reviewRank", _rr.rank(cls)),
                         "reviewRisk": z.get("reviewRisk", _rr.risk(cls)),
                         "reviewRate": z.get("reviewRate", _rr.rate(cls)),
                         "reviewWhy": z.get("reviewWhy") or _rr.why(cls)})
    ordered = _rr.order_zones(rows)
    # pages, ordered by their own riskiest region — the queue the estimator walks is a
    # page at a time, and a page is exactly as urgent as the worst thing on it
    pages = {}
    for r in ordered:
        p = pages.setdefault(r["page"], {"page": r["page"], "sf": 0.0, "zones": 0,
                                         "topClass": r["readerClass"],
                                         "topRank": r["reviewRank"],
                                         "topReader": r["reader"]})
        p["sf"] += float(r.get("netArea") or 0)
        p["zones"] += 1
    page_list = sorted(pages.values(), key=lambda p: _rr.sort_key(p["topClass"], p["sf"]))
    # ⭐ THE SUGGESTIONS INDEX. Every suggest_only piece in the job, in one payload,
    # ranked by the same measured risk order as the queue. It replaces the frontend's
    # N-per-page /polygons scan (one request per page, every page, to find out whether
    # there was anything to suggest at all) — which is why suggestions were effectively
    # invisible on long sets. Suggestions are NOT zones: none of this SF is booked, is
    # in `zones` above, or is in any total; accepting one is a separate human click
    # through /accept-suggestion.
    suggestions = []
    for _pn, _polys in (polys_by or {}).items():
        try:
            _pni = int(_pn)
        except Exception:
            continue
        for _idx, p in enumerate(_polys or []):
            if not p.get("suggest_only"):
                continue
            suggestions.append({"page": _pni,
                                "id": (p.get("id") if p.get("id") is not None else _idx),
                                "sf": round(float(p.get("area_sf") or 0), 1),
                                "cx": p.get("cx"), "cy": p.get("cy"),
                                "readerClass": p.get("reader_class")})
    suggestions.sort(key=lambda s: _rr.sort_key(s["readerClass"], s["sf"]))
    # ⭐ DETECTION-GAP ROWS. Admitted ELEVATION pages the auto engine drew nothing on — no
    # booked wall, no suggestion, no linear run (recorded by process() at the page-drop point).
    # She cannot accept a wall she cannot see is missing; these turn a silently-dropped blank
    # elevation into an actionable "draw here" row. Additive and books NOTHING: none of this is
    # a zone, is in `zones`, or is in any total. Empty (renders nothing) on a job with no gaps.
    gaps = sorted(
        ({"page": g.get("page"), "sheetRef": g.get("sheetRef"),
          "note": g.get("note"), "detectionGap": True}
         for g in (j.get("detectionGaps") or [])),
        key=lambda g: (g["page"] if g["page"] is not None else 0))
    return {"jobId": jid, "zones": ordered, "pages": page_list,
            "suggestions": suggestions,
            "suggestionSF": round(sum(s["sf"] for s in suggestions), 1),
            "gaps": gaps,
            "reviewConfirmed": confirmed,
            "table": dict(_rr.TABLE, order=_rr.ORDER, risk=_rr.RISK),
            "note": ("ORDERING ONLY — every region is still confirmed by the estimator and "
                     "no SF is booked, asserted or hidden by this ranking. The order is the "
                     "reader class's measured share of walls where the engine booked more "
                     "than 1.15x the wall's true SF, upper-bounded for support.")}

@app.post("/review-confirm")
def review_confirm(payload: dict = Body(...)):
    """SHE LOOKED AT IT — the review walk, moved off the browser and onto the job.

    payload: {jobId, page, materialName, confirmed:bool, pieceId?}

    Until now the walker's progress lived in the frontend's own state and its saved
    recall record, so it died with the tab and never existed for a second device or a
    second person. The queue she is walking is the product's core loop (STEP 2/4:
    "she confirms every number fast"), and its progress belongs to the JOB.

    ⭐ IT BOOKS NOTHING AND IT IS READ BY NOTHING THAT COMPUTES SF. This flag is
    display state: no zone, total, Excel, evidence PDF, bench or grade reads it. That
    is deliberate — a confirmation is not a measurement, and the moment "confirmed"
    starts changing quantities it becomes an assert tier, which is the gate
    FIX_TWO_TIER_GATE refused to ship. Her NUMBER goes through /set-net.

    `pieceId` is optional and extends the key, so the per-piece rows the queue now
    returns can be ticked individually without colliding with the row's own tick.
    """
    jid = payload.get("jobId")
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    try:
        page = int(payload.get("page"))
    except Exception:
        raise HTTPException(400, "page required")
    mat = payload.get("materialName")
    if mat is None:
        raise HTTPException(400, "materialName required")
    key = _net_key(page, mat)
    if payload.get("pieceId") is not None:
        key += "|#%s" % payload.get("pieceId")
    conf = dict(j.get("reviewConfirmed") or {})
    if payload.get("confirmed", True):
        conf[key] = {"at": int(time.time() * 1000)}
    else:
        conf.pop(key, None)
    j["reviewConfirmed"] = conf
    jobs[jid] = j
    try:
        _persist_job(jid)
    except Exception:
        pass
    return {"ok": True, "jobId": jid, "key": key,
            "confirmed": key in conf, "confirmedCount": len(conf),
            "reviewConfirmed": conf,
            "note": ("Display state only — no SF, total, Excel or evidence quantity reads "
                     "this. The confirmed NUMBER is /set-net.")}


@app.post("/set-net")
def set_net(payload: dict = Body(...)):
    """STEP 4 — NET BY CONFIRMATION. Her number, on one zone, and it is THE number.

    payload: {jobId, page, zoneKey, net}          set the override
             {jobId, page, zoneKey, net: null}    clear it (back to the machine's draft)
             {jobId, reset: true}                 clear every override on the job

    `zoneKey` is the zone's scope identity — its `material_group` stamp, or its
    materialName on ungrouped/marked pages (see `_scope_group_of_zone`). The review
    queue hands it back on every row as `zoneKey`, so the caller never has to guess.

    ⭐ WHAT THIS DOES NOT DO. It does not compute a net, does not suggest one, does not
    show the job's habit, and does not apply a ratio to anything. Two programmes (the
    v18 ladder, then the netting oracle) proved that even PERFECT wall selection with
    the best constant ratio scores BELOW baseline at 27 confidently-wrong, and
    FIX_OPENINGS stands: a habit may be SHOWN, never APPLIED. This pass is the
    plumbing and nothing else — the display of openings/habit is a later pass.

    IDENTITY BY DEFAULT: with no override the takeoff is byte-identical to today's,
    and clearing an override restores the machine's own number from `netAreaAuto` and
    removes every field this ever added.
    """
    jid = payload.get("jobId")
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    ov = {} if payload.get("reset") else dict(j.get("netOverrides") or {})
    key = None
    if not payload.get("reset"):
        try:
            page = int(payload.get("page"))
        except Exception:
            raise HTTPException(400, "page required")
        zk = payload.get("zoneKey")
        if zk is None:
            raise HTTPException(400, "zoneKey required")
        key = _net_key(page, zk)
        raw = payload.get("net", None)
        if raw is None or raw == "":
            ov.pop(key, None)
        else:
            try:
                val = float(raw)
            except Exception:
                raise HTTPException(400, "net must be a number")
            if val < 0:
                raise HTTPException(400, "net cannot be negative")
            ov[key] = {"net": round(val, 1), "ts": int(time.time() * 1000),
                       "by": str(payload.get("by") or "estimator")[:40]}
    j["netOverrides"] = ov
    n = _apply_nets(j)
    jobs[jid] = j
    try:
        _persist_job(jid)
    except Exception:
        pass
    return {"ok": True, "jobId": jid, "key": key, "netOverrides": ov,
            "zonesOverridden": n,
            "jobTotal": round(sum(float(z.get("netArea") or 0)
                                  for e in (j.get("takeoffData") or [])
                                  for z in (e.get("zones") or [])), 1),
            "takeoffData": j.get("takeoffData") or [],
            "note": ("Her confirmed net IS the delivered number — totals, the Excel and the "
                     "evidence PDF all read it. Nothing here computes or suggests a net.")}


@app.post("/bid-excel")
def bid_excel_endpoint(payload: dict = Body(...)):
    """Template-v2 BFS proposal workbook: clones the estimator's real letterhead
    template (bid_template_v2.xlsx) and fills it her way — per-page quantity
    addition formulas (=2049+321), Amount=Qty*Conv*Rate, TOTAL linked. Validated
    13/13 cell-identical against the submitted 26-262 Malden bid."""
    import bid_excel as _be
    try:
        data = _be.fill_bid(payload or {})
    except Exception as ex:
        raise HTTPException(500, f"bid excel failed: {ex}")
    name = ((payload.get("job_number") or "bid") + " - " + (payload.get("job_name") or "proposal"))
    safe = "".join(ch for ch in name if ch not in '\\/:*?"<>|')[:80]
    return Response(content=data,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.xlsx"'})

@app.post("/scope-read")
async def scope_read(pdf: UploadFile = File(...)):
    """Extract text (+ estimator's checkmarks/notes) from a scope PDF so the Scope tab can read PDFs, not just Excel."""
    data = await pdf.read()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(400, "not a readable PDF")
    npages = doc.page_count
    parts = []
    for i in range(min(npages, 40)):
        t = doc[i].get_text() or ""
        anns = [a.info.get("content", "") for a in (doc[i].annots() or []) if a.info.get("content", "")]
        block = t
        if anns:
            block += "\n[estimator marks/notes on page: " + " | ".join(anns[:60]) + "]"
        if block.strip():
            parts.append(f"# Page {i+1}\n{block}")
    doc.close()
    txt = "\n\n".join(parts)
    return {"text": txt, "chars": len(txt), "pages": len(parts)}

@app.get("/snap-points/{jid}/{page}")
def snap_points(jid: str, page: int, rect: str = "", min_len: float = 0.0):
    """Bluebeam-style corner snap: the drawing's real CAD geometry → structural snap corners (endpoints +
    intersections of the LONG horizontal/vertical lines = building outline, floor lines, major openings).
    The estimator's clicks snap to these (pixel-perfect like Bluebeam); auto-markup can snap its edges too.

    ⭐ DETAIL TIER (2026-08-17): `?rect=x0,y0,x1,y1[&min_len=pt]`.

    `Lmin = 0.06 * max(W, H)` is a WHOLE-SHEET threshold — "structural, not
    window-mullion/brick/detail noise". It is also why the 869 walls with no engine ink
    on them (246k SF, r4_truly_missed_partition) cannot be clicked in: they are small
    trims and friezes, every corner they own is shorter than 6% of a D-size sheet, and
    a snap layer with no corners on the thing she is trying to trace is a blank page.
    Zoomed in, "detail noise" is the subject — so when a viewport is supplied the SAME
    6% rule is measured against the VIEWPORT instead of the sheet, which makes the
    threshold shrink exactly as fast as she zooms. `min_len` (PDF points) overrides it
    outright.

    `rect` is in NORMALISED page coordinates (0..1, the frame every other polygon on
    this API speaks); values above 1 are read as raw PDF points. Points are returned in
    the same normalised page frame as the full-sheet call, so the caller's transform is
    unchanged. The 1500-point cap is unchanged, and it now spends all 1500 inside the
    viewport instead of on the sheet's outline.

    IDENTITY: with neither parameter this is byte-for-byte the endpoint it was."""
    j = get_job(jid)
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pg = doc[page - 1]; W, H = pg.rect.width, pg.rect.height
    box = None
    if rect:
        try:
            v = [float(x) for x in str(rect).replace(" ", "").split(",")]
            if len(v) != 4:
                raise ValueError("rect needs 4 numbers")
            if max(abs(x) for x in v) <= 1.0:      # normalised -> PDF points
                v = [v[0] * W, v[1] * H, v[2] * W, v[3] * H]
            x0, x1 = sorted((v[0], v[2])); y0, y1 = sorted((v[1], v[3]))
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                raise ValueError("rect is empty")
            box = (x0, y0, x1, y1)
        except Exception as ex:
            doc.close(); raise HTTPException(400, "bad rect (want x0,y0,x1,y1): %s" % ex)
    if min_len and min_len > 0:
        Lmin = float(min_len)
    elif box is not None:
        Lmin = 0.06 * max(box[2] - box[0], box[3] - box[1])   # the same rule, on the viewport
    else:
        Lmin = 0.06 * max(W, H)  # "long" = structural, not window-mullion/brick/detail noise
    hlines = []; vlines = []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                segs = []
                if it[0] == "l":
                    segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
                elif it[0] == "re":
                    r = it[1]; segs = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                                       (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                for (x1, y1, x2, y2) in segs:
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 < Lmin ** 2:
                        continue
                    if box is not None and (max(x1, x2) < box[0] or min(x1, x2) > box[2] or
                                            max(y1, y2) < box[1] or min(y1, y2) > box[3]):
                        continue      # segment never enters the viewport
                    if abs(y2 - y1) < abs(x2 - x1) * 0.02:
                        hlines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
                    elif abs(x2 - x1) < abs(y2 - y1) * 0.02:
                        vlines.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
    except Exception:
        pass
    doc.close()
    hlines = hlines[:600]; vlines = vlines[:600]  # bound compute
    pts = []
    for (xa, xb, y) in hlines: pts += [(xa, y), (xb, y)]
    for (ya, yb, x) in vlines: pts += [(x, ya), (x, yb)]
    for (xa, xb, yh) in hlines:  # H×V intersections = true corners
        for (ya, yb, xv) in vlines:
            if xa - 2 <= xv <= xb + 2 and ya - 2 <= yh <= yb + 2:
                pts.append((xv, yh))
    if box is not None:
        # spend the 1500 on the thing she is looking at (a 2pt pad keeps a corner that
        # sits exactly on the viewport edge, which is where a trim usually starts)
        pts = [p for p in pts if box[0] - 2 <= p[0] <= box[2] + 2 and box[1] - 2 <= p[1] <= box[3] + 2]
        dedup = min(6.0, 0.006 * max(box[2] - box[0], box[3] - box[1]))
    else:
        dedup = 6.0
    keep = []
    for pt in pts[:9000]:
        if not any(abs(pt[0] - k[0]) < dedup and abs(pt[1] - k[1]) < dedup for k in keep):
            keep.append(pt)
        if len(keep) >= 1500:
            break
    norm = [[round(x / W, 5), round(y / H, 5)] for (x, y) in keep]
    out = {"points": norm, "width": W, "height": H, "count": len(norm)}
    if box is not None or (min_len and min_len > 0):
        out["rect"] = ([round(box[0] / W, 5), round(box[1] / H, 5),
                        round(box[2] / W, 5), round(box[3] / H, 5)] if box else None)
        out["minLen"] = round(Lmin, 3)
        out["tier"] = "detail"
    return out

@app.post("/admin/upload-model")
async def upload_model(model: UploadFile = File(...), key: str = "", slot: str = ""):
    """One-time: load a trained ONNX model onto the volume. FAIL-CLOSED behind
    ADMIN_KEY: with no key configured this endpoint is DISABLED and refuses every
    request — the model file it writes is what every later inference loads, so an
    open door here is silent poisoning of every SF number the system produces.
    slot="" -> the extent model (/data/model.onnx, v11 path); slot="v13" -> the
    boundary model (/data/model_v13.onnx) — its PRESENCE activates the v13 reader."""
    ok, code, msg = admin_auth.authorize(key, "/admin/upload-model")
    if not ok:
        raise HTTPException(code, msg)
    data = await model.read()
    if slot == "v13":
        path = os.environ.get("V13_ONNX", "/data/model_v13.onnx")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return {"ok": True, "bytes": len(data), "slot": "v13"}
    path = os.environ.get("MODEL_ONNX", "/data/model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    model_infer.reset()
    return {"ok": True, "bytes": len(data), "model_available": model_infer.available()}

@app.post("/split-suggest")
def split_suggest(payload: dict = Body(...)):
    """✂ SUGGESTIONS: v13's boundary channel proposes cut lines inside a selected
    piece (probe-proven: boundary ink sits ON the estimator's wall edges at median
    1.00). The estimator accepts with one click — and every accepted cut becomes
    boundary supervision for v14. payload: {jobId, page(1-based), points([[x,y]..]
    normalized)} → {cuts: [{axis:'v'|'h', pos: normalized, span:[a,b]}]}"""
    jid = payload.get("jobId")
    page = int(payload.get("page") or 0)
    pts = payload.get("points") or []
    j = get_job(jid)
    if not j or not j.get("pdf") or page < 1 or len(pts) < 3:
        raise HTTPException(400, "bad request")
    import numpy as np, cv2
    mp = os.environ.get("V13_ONNX", "/data/model_v13.onnx")
    if not os.path.isfile(mp):
        return {"cuts": [], "note": "boundary model not present"}
    try:
        doc = fitz.open(stream=j["pdf"], filetype="pdf")
        pg = doc[page - 1]
        W, H = pg.rect.width, pg.rect.height
        z = 1536.0 / max(W, H)
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        doc.close()
        import vector_hatch as _vh
        if _vh._V13_SESS is None:
            import onnxruntime as ort
            _vh._V13_SESS = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        ih, iw = img.shape[:2]
        T = 768
        prob = np.zeros((3, ih, iw), np.float32)
        cnt = np.zeros((ih, iw), np.float32)
        for y0 in sorted(set(list(range(0, max(1, ih - T + 1), T // 2)) + [max(0, ih - T)])):
            for x0 in sorted(set(list(range(0, max(1, iw - T + 1), T // 2)) + [max(0, iw - T)])):
                tile = img[y0:y0 + T, x0:x0 + T]
                th, tw = tile.shape[:2]
                pad = np.zeros((T, T, 3), np.uint8)
                pad[:th, :tw] = tile
                x = (pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
                lg = _vh._V13_SESS.run(None, {"input": x})[0][0]
                e = np.exp(lg - lg.max(0, keepdims=True))
                sm = e / e.sum(0, keepdims=True)
                prob[:, y0:y0 + th, x0:x0 + tw] += sm[:, :th, :tw]
                cnt[y0:y0 + th, x0:x0 + tw] += 1
        prob /= np.maximum(cnt, 1)[None]
        bnd = (prob.argmax(0) == 2).astype(np.uint8)
        m = np.zeros((ih, iw), np.uint8)
        cv2.fillPoly(m, [np.array([[int(px * W * z), int(py * H * z)] for px, py in pts], np.int32)], 1)
        inside = bnd & m
        col = inside.sum(0); hgt = m.sum(0)
        row = inside.sum(1); wid = m.sum(1)
        xs = [px * W * z for px, py in pts]; ys = [py * H * z for px, py in pts]
        x0b, x1b, y0b, y1b = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        raw = []
        for cx in range(x0b + 4, min(x1b - 4, iw)):
            if hgt[cx] > 8 and col[cx] >= 0.6 * hgt[cx]:
                raw.append(("v", cx))
        for cy in range(y0b + 4, min(y1b - 4, ih)):
            if wid[cy] > 8 and row[cy] >= 0.6 * wid[cy]:
                raw.append(("h", cy))
        cuts = []
        for ax in ("v", "h"):
            ps = sorted(c for a, c in raw if a == ax)
            i = 0
            while i < len(ps):
                k = i
                while k + 1 < len(ps) and ps[k + 1] - ps[k] <= 3:
                    k += 1
                c = (ps[i] + ps[k]) / 2
                if ax == "v":
                    cuts.append({"axis": "v", "pos": round(c / z / W, 5),
                                 "span": [round(y0b / z / H, 5), round(y1b / z / H, 5)]})
                else:
                    cuts.append({"axis": "h", "pos": round(c / z / H, 5),
                                 "span": [round(x0b / z / W, 5), round(x1b / z / W, 5)]})
                i = k + 1
        return {"cuts": cuts[:8]}
    except Exception as e:
        return {"cuts": [], "note": f"suggest failed: {e}"}

@app.get("/admin/export-corrections")
def export_corrections(key: str = "", since: str = ""):
    """Flywheel export: every /learn label captured by the live app (renames, deletes,
    splits, bucket confirms, final-* answer keys) as one JSON payload — the raw
    material for the v14 dataset. Labels only (PDFs stay on the volume; fetch a
    specific correction's drawing separately if needed). FAIL-CLOSED behind ADMIN_KEY
    exactly like upload-model: this payload IS the training moat, so with no key
    configured the endpoint is DISABLED rather than open."""
    ok, code, msg = admin_auth.authorize(key, "/admin/export-corrections")
    if not ok:
        raise HTTPException(code, msg)
    out = []
    try:
        for d in sorted(os.listdir(CORR_DIR)):
            if since and d <= since:
                continue
            lp = os.path.join(CORR_DIR, d, "labels.json")
            if not os.path.isfile(lp):
                continue
            try:
                with open(lp, encoding="utf-8") as fh:
                    rec = json.load(fh)
                rec["_ts"] = d
                rec["_has_pdf"] = os.path.isfile(os.path.join(CORR_DIR, d, "drawing.pdf"))
                out.append(rec)
            except Exception:
                continue
    except Exception:
        pass
    return {"n": len(out), "corrections": out}

@app.get("/health")
def health():
    try:
        on_disk = len([d for d in os.listdir(JOBS_DIR) if os.path.isdir(os.path.join(JOBS_DIR, d))]) if os.path.isdir(JOBS_DIR) else 0
    except Exception:
        on_disk = 0
    return {"status": "ok", "engine": "digitize-markup", "deps": "pymupdf+onnx",
            "auto_engine": "model" if model_infer.available() else "texture",
            "ocr": ocr_text.available(), "ocr_err": ocr_text.last_error(),
            "jobs_in_mem": len(jobs), "jobs_on_disk": on_disk, "mem_cap": MAX_MEM_JOBS,
            # BOUNDED JOB QUEUE: report the live concurrency cap + how many jobs are
            # waiting/running, so a batch backing up (or a hung slot) is visible. depth
            # > max_concurrency means uploads are queued behind the RAM-safe worker cap.
            "job_queue": {"max_concurrency": MAX_CONCURRENCY, "depth": _queue_depth,
                          "workers_started": _workers_started},
            # FIX #1 grouping: a deploy that lands app.py without mat_canon.py keeps serving
            # takeoffs with today's per-name rollup. That must be VISIBLE — a silent fallback
            # is indistinguishable from "the grouping never worked".
            "mat_canon": (False if _mc is None else True), "mat_canon_err": _MC_ERR,
            # Same move as mat_canon above, for a BIGGER silent fallback. `_v13_regions`
            # reads V13_ONNX with the production default /data/model_v13.onnx and returns []
            # SILENTLY when it is absent, so a prod running six readers instead of seven
            # looks exactly like one running seven. `v13_status()` is pure, is called by
            # nothing in the detect path, and never raises. It reports LOADABILITY, not that
            # inference ran -- the page budget it reports beside it can still withhold work.
            "v13": vector_hatch.v13_status(),
            # REVIEW RANKING: same visibility rule a third time. A missing review_rank.py
            # degrades to an unordered queue, which looks exactly like a queue nobody
            # prioritised — so the module's presence AND the corpus its risk table was
            # derived from are reported, because a table silently frozen at an old corpus
            # is the failure this ships with a guard against.
            "review_rank": (False if _rr is None else dict(_rr.TABLE, order=_rr.ORDER)),
            "review_rank_err": _RR_ERR,
            # RELIABILITY GUARDS: same visibility rule. If rel_guard.py failed to land, the
            # per-page pre-check/timeout silently no-ops (auto-detect runs unbounded again), so
            # the deploy confirms the running app.py imported it and reports the live thresholds.
            "rel_guard": (False if _rg is None else _rg.health()), "rel_guard_err": _RG_ERR,
            # SIZE-OUTLIER GATE (Feature 3): report the live threshold + gold floor, same
            # visibility rule as the guards above — a demote gate whose threshold silently
            # drifted (or a build that landed without it) is otherwise invisible.
            "size_gate": {"frac": SIZE_OUTLIER_FRAC, "gold_floor": SIZE_OUTLIER_GOLD_FLOOR}}
